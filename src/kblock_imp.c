/* kblock_imp.c -- the DSP side of the engine, on the HMX and HVX with every activation in the HMX layout: first Anima's DiT block
 * (self-attention with RoPE, cross-attention to the text, MLP; adaLN shift / scale / gate around each; RMS norms on q and k), then
 * the Krea 2 / MOSS-Image / FLUX.2 Klein blocks, YuE2's layers and the plain products; the VAEs, SDXL and YuE2's decoder in the .inc files. The residual stream is carried / 16 (the gates are divided by 16, the LayerNorm eps by 256), as the host lays out the weights and parameters.
 *
 * HMX layout: activations [rows x W]: per 32-row tile its W/32 blocks of 32 x 32 fp16; inside a block element (r, c)
 * at 32 (r & ~1) + 2c + (r & 1), so HVX vector k (0..15) of a block holds rows 2k (even lanes) and 2k + 1 (odd lanes), column c in
 * lanes 2c and 2c + 1. Widening a vector to fp32 puts the even lanes (row 2k) in the low vector and the odd lanes in the high one.
 * Weights [K x N] (= W^T): per 64-column group its two 32-column halves, each the K/32 row blocks.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kdsp.h"
#include "hvx_hexagon_protos.h"
#include "hexagon_types.h"
#include "kv69.h"
#include "kpool.inc"
#include "kblock.h"

void hmx_mpy_row(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t wgt_step, uint8_t *bias, uint32_t count);
void hmx_mpy_group(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t tile_bytes, uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks, uint32_t tiles, uint32_t half);
void hmx_mpy_chunks(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t out_ctrl,
                    uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks);
void hmx_mpy_chunks_lorad(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t out_ctrl, uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks, uint8_t *lact, uint8_t *lwgt, uint32_t lchunks);
void hmx_mpy_chunks_lora(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t out_ctrl,
                         uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks, uint8_t *lact, uint8_t *lwgt, uint32_t lchunks);

static uint32_t VTCM_BYTES = 8 * 1024 * 1024;   /* set from the chip's VTCM at the start of a run (8 MB on v73) */
/* an entry whose buffers are laid out for 8 MB of VTCM sets this before eng_begin; the blocks' products and attention size their tiles
   from VTCM_BYTES and run on 4 MB (chips with a reduced VTCM, such as some v73 parts; not tried on one) */
static int g_need8;
#define BLK 2048                         /* bytes of a 32 x 32 fp16 block */
#define BH 1024                          /* halfwords of a block */
#define KC 512
#define MAXD 6144                        /* widest model (Krea 2) */
static uint32_t g_D = 2048; static int g_NH = 16;   /* the model's width and query heads, set per run */
#define D g_D
#define DH 128
#define NH g_NH
#define RS 16.f

static worker_pool_context_t g_pool;
static int g_hmxclk = -1;   /* the HMX clock vote's answer (eng_begin, v75 on): 0 taken, else the error; -1 not asked */
/* the engine's optimisation switches, one bit each, for A/B on the phone without a rebuild: the host sets them
 * for the session with kblock_probe(h, 0x40000000 | bits) */
enum { OPT_SMSOLO = 1,      /* a softmax tile of <= g_smsolo key blocks done by the main thread alone (no check-in of every worker) */
       OPT_KEEPPOOL = 2,    /* the worker pool kept for the session instead of made and joined each call */
       OPT_POSTALL = 4,     /* post jobs on every HVX context (the weight copier's too), off: g_nw - WCOPY */
       OPT_XHVXSUM = 8,     /* SDXL's softmax row sums on the HVX (its head 64 makes the ones block +50% P V), off: XHMXSUM */
       OPT_VDA = 16,        /* the VAE attention's V laid out once, not once a query slab */
       OPT_ACTX = 32,       /* the activations (GELU, MOSS's situ, SwiGLU) two vectors at a time, their chains side by side */
       OPT_RESPOST = 64,    /* the gated residuals in the products' post (OUT_RESG, OUT_ADDRESG), not a pass over x after */
       OPT_HEADPAR = 128,   /* attention()'s per-head q, v copies and K transposes shared by the workers */
       OPT_KEEPVOTE = 256,  /* the power votes (DCVS, HMX, HVX) once a session, not once a call (~0.28 ms a call) */
       OPT_VSL = 512,       /* the VAE attention's query slab as large as the scratch allows (up to 4096 rows), not 512 */
       OPT_CREW = 1024,     /* attention()'s softmax workers kept for a head (SDXL's xsm crew), not submitted again each 32 query rows */
       OPT_RRPOST = 2048,   /* Anima's q / k RMS norm and RoPE in the q k v products' post (a head's two groups paired), not a pass after */
       OPT_PREFILL = 4096,  /* the next batch's activation tiles made while the HMX runs this one (into a second VTCM buffer, or a DDR staging buffer) */
       OPT_XHP = 8192,      /* SDXL's attention: a head's preparation shared by the softmax crew */
       OPT_XRES = 16384,    /* SDXL's out and down products add their bias and x in the post */
       OPT_HMXG = 32768,    /* hmx_mpy_group: every tile of a batch and both halves of a weight group in one call */
       OPT_LORAD = 65536,   /* the LoRA's blocks in one deep load each side (not on v81) */
       OPT_WDMA = 131072,   /* the weight copy by the DSP's user DMA */
       OPT_AVCREW = 262144, /* the P V finish taken by the softmax crew */
       OPT_XDN = 524288,    /* SDXL's down op, the stride-2 convolution on the kept pixels only */
       OPT_VRES = 1048576,  /* the VAE resnet's residual added in its second convolution's post (OUT_ADD into a), not vk_add after */
       OPT_XTEMB = 2097152, /* SDXL's resnet time embedding folded into conv 1's bias, not a vk_bias pass after (one rounding fewer) */
       OPT_VKC = 4194304 }; /* the VAE attention's P V key chunk 64 blocks, not 512 (the partial sums add up in fp16: values change a little) */
static uint64_t g_tatt[2];   /* attention(): P V (with the division) and QK time, us (times[20], [21]) */
static int g_voted;       /* the votes made this session (kblock_close clears it) */
#define KDEF_WDMA OPT_WDMA   /* the DMA weight copy: v73 block -2.8%, v81 with 7 workers -4% */
#if __HVX_ARCH__ >= 81
#define KDEF_ACTX OPT_ACTX   /* v81 -3.4% (Anima block 65.6 -> 63.4 ms); v73 no change */
#else
#define KDEF_ACTX 0
#endif
static int g_opt = OPT_KEEPPOOL | OPT_VDA | OPT_RESPOST | OPT_HEADPAR | OPT_KEEPVOTE | OPT_VSL | OPT_CREW | OPT_RRPOST | OPT_XHP | OPT_VRES | OPT_VKC | KDEF_WDMA | KDEF_ACTX;   /* defaults: the ones measured faster and bit-identical */
static int g_smsolo = 16;   /* OPT_SMSOLO's bound in key blocks (Anima's cross attention: 16; its self attention's tiles are 32) */
static uint8_t *g_vtcm, *g_bias, *g_bias_av;   /* the HMX output blocks: identity; and P V's, which scales by 2^-PSH (see softmax_rows) */
/* the probabilities are made 2^PSH times too large: down to 2^-(15 + PSH) of the row's largest survive (at 1, all below 2^-15 were
   zero: over 5120 keys that dropped a share of the weight and doubled the self-attention error); P V's HMX output and the
   row sums take the factor back out */
#define PSH 8

int kblock_open(const char *uri, remote_handle64 *h) { *h = (remote_handle64)malloc(1); return *h ? 0 : AEE_ENOMEMORY; }
/* Klein's scratch: only the DSP uses it, and as an RPC buffer every call paid for the cache maintenance of all of it (a third of the calls' 20 ms overhead a block) */
static uint16_t *g_dscr; static size_t g_dscrn;
static void pool_close(void);
int kblock_close(remote_handle64 h) { pool_close(); g_voted = 0; free(g_dscr); g_dscr = 0; g_dscrn = 0; free((void *)h); return 0; }

#if __HEXAGON_ARCH__ >= 79
/* v79 / v81: a value past fp16's range comes out of qf16 -> hf and out of the HMX as inf (v73 saturates), and one inf in the residual
 * stream spreads through attention; with real pictures' activations it happens: clamp to +-65504 as v73 does */
#define HF_CLAMP 1
static inline HVX_Vector clamp_hf(HVX_Vector h) { return Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(h, Q6_V_vsplat_R(0xfbfffbff)), Q6_V_vsplat_R(0x7bff7bff)); }
/* the residual stream's NaN to 0 and +-inf to +-65504, as ComfyUI's nan_to_num on fp16 FLUX blocks: a value past fp16 (Klein 9B's double
   blocks put two there) made v73 an inf that stayed where it was, but on v79 / v81 the HMX met it as +inf + -inf in the next product and
   gave a NaN row, which attention spread to every row (every Klein 9B picture was NaN from step 1) */
static inline HVX_Vector fin_hf(HVX_Vector h)
{
    const HVX_VectorPred nan = Q6_Q_vcmp_gt_VhVh(Q6_V_vand_VV(h, Q6_V_vsplat_R(0x7fff7fff)), Q6_V_vsplat_R(0x7c007c00));
    return clamp_hf(Q6_V_vmux_QVV(nan, Q6_V_vzero(), h));
}
/* hfr: qf16 -> hf without v79 / v81's clamp, for a value that a later step clamps anyway (an inf there comes out of it as +-65504) */
static inline HVX_Vector hfr(HVX_Vector q16) { return Q6_Vhf_equals_Vqf16(q16); }
static inline HVX_Vector hf(HVX_Vector q16) { return clamp_hf(Q6_Vhf_equals_Vqf16(q16)); }
#else
static inline HVX_Vector hf(HVX_Vector q16) { return Q6_Vhf_equals_Vqf16(q16); }
static inline HVX_Vector hfr(HVX_Vector q16) { return Q6_Vhf_equals_Vqf16(q16); }
#endif
#ifdef HF_CLAMP
#define FIN(v_) fin_hf(v_)
#else
#define FIN(v_) (v_)
#endif
static inline HVX_Vector sfv(HVX_Vector q32) { return Q6_Vsf_equals_Vqf32(q32); }
static HVX_Vector splat_hf(float f) { __fp16 h = (__fp16)f; uint16_t b; memcpy(&b, &h, 2); return Q6_Vh_vsplat_R(b); }
/* completion tokens: the worker pool's synctoken wait takes ~40 us to wake the waiter on v81, so the waiter spins instead */
#if __HVX_ARCH__ >= 79
typedef struct { int n; int done; } spintok_t;
static inline void st_init(spintok_t *t, int n) { t->n = n; __atomic_store_n(&t->done, 0, __ATOMIC_RELEASE); }
static inline void st_done(spintok_t *t) { __atomic_fetch_add(&t->done, 1, __ATOMIC_RELEASE); }
static inline void st_wait(spintok_t *t) { while (__atomic_load_n(&t->done, __ATOMIC_ACQUIRE) < t->n) { } }
#else                                       /* v73: the synctoken is cheaper there than a spinning thread */
typedef worker_synctoken_t spintok_t;
#define st_init worker_pool_synctoken_init
#define st_done worker_pool_synctoken_jobdone
#define st_wait worker_pool_synctoken_wait
#endif
/* parallel HVX jobs per split: the HVX contexts the chip has (4 on v73; more threads than contexts only wait) */
#define MAXW 12

#define MAIN_TAKE sm_take(&sa)
/* softmax jobs: 4 for Anima (more only slowed the main thread's HMX feeding on v81), 6 for long sequences (Krea's 4224 keys, Klein), whose
 * softmax is the longer side: Krea's attention went 67 -> 60 ms a block on v81 with 6 jobs, Anima's did not change */
static int g_smj = 4;
#define SMJ (g_nw < g_smj ? g_nw : g_smj)
static int g_nw = 4;
/* IEEE sf lanes to fp16 bits in the low halfword (rounded, flushed below 2^-14, clamped at 65504), integer ops only */
static inline HVX_Vector sf_to_hbits(HVX_Vector f)
{
    HVX_Vector a = Q6_Vw_vadd_VwVw(Q6_V_vand_VV(f, Q6_V_vsplat_R(0x7fffffff)), Q6_V_vsplat_R(0x1000));
    HVX_Vector h = Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(a, 13), Q6_V_vsplat_R(112 << 10));
    h = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(h, Q6_V_vzero()), Q6_V_vsplat_R(0x7bff));
    return Q6_V_vor_VV(h, Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(f, 16), Q6_V_vsplat_R(0x8000)));
}
/* two sf vectors (even halfword lanes e, odd lanes o) to one hf vector. qf32 values are never paired or moved: on v81 a qf
 * register read by a non-qf instruction (a vcombine too) loses its extended state */
static inline HVX_Vector sf2_to_hf(HVX_Vector e, HVX_Vector o) { return Q6_V_vor_VV(sf_to_hbits(e), Q6_Vw_vasl_VwR(sf_to_hbits(o), 16)); }
/* hf a * b and a + b with one rounding to nearest: the qf16 ops drop the low bits (a bias toward zero that piles up over a
 * block: the error was four times fp16's), so these go through qf32 and sf_to_hbits */
static inline HVX_Vector mul_hf(HVX_Vector a, HVX_Vector b)
{ HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(a, b); return sf2_to_hf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p))); }
static inline HVX_Vector add_hf(HVX_Vector a, HVX_Vector b)
{
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
    HVX_VectorPair pa = Q6_Wqf32_vmpy_VhfVhf(a, one), pb = Q6_Wqf32_vmpy_VhfVhf(b, one);
    return sf2_to_hf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(pa), Q6_V_lo_W(pb))), Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(pa), Q6_V_hi_W(pb))));
}
static HVX_Vector splat_sf(float f) { uint32_t b; memcpy(&b, &f, 4); return Q6_V_vsplat_R(b); }
/* a vector with column c of a block (lanes 2c, 2c + 1) set to v[c] */
static HVX_Vector colvec(const float *v) { __fp16 t[64] __attribute__((aligned(128))); for (int c = 0; c < 32; c++) t[2 * c] = t[2 * c + 1] = (__fp16)v[c]; return *(HVX_Vector *)t; }
static inline uint16_t *blk(uint16_t *buf, uint32_t width, uint32_t tile, uint32_t cb) { return buf + ((size_t)tile * (width / 32) + cb) * BH; }

static void hvx_copy(uint8_t *restrict dst, const uint8_t *restrict src, uint32_t bytes)
{
    HVX_Vector *d = (HVX_Vector *)dst; const HVX_Vector *s = (const HVX_Vector *)src;
    const uint32_t step = 64, n = bytes / 128;
    for (uint32_t i = 0; i < n; i += step)
    {
        if (i + step < n) Q6_l2fetch_AP((void *)(src + (i + step) * 128), ((uint64_t)8192 << 32) | ((uint64_t)8192 << 16) | 1);
        uint32_t m = n - i < step ? n - i : step;
        for (uint32_t k = 0; k < m; k++) d[i + k] = s[i + k];
    }
}
/* start pulling bytes (a multiple of 8 KB, up to 2 MB) from DDR into L2 */
static inline void prefetch(const void *p, uint32_t bytes)
{ uint32_t h = bytes / 8192; if (h > 255) h = 255; if (h) Q6_l2fetch_AP((void *)p, ((uint64_t)8192 << 32) | ((uint64_t)8192 << 16) | h); }
typedef struct { uint8_t *dst; const uint8_t *src; uint32_t bytes; spintok_t tok; } copy_job_t;
static void copy_worker(void *a) { copy_job_t *j = (copy_job_t *)a; hvx_copy(j->dst, j->src, j->bytes); st_done(&j->tok); }
static void copy_start(copy_job_t *j, uint8_t *dst, const uint8_t *src, uint32_t bytes)
{ j->dst = dst; j->src = src; j->bytes = bytes; st_init(&j->tok, 1); worker_pool_job_t job = { copy_worker, j }; worker_pool_submit(g_pool, job); }
/* a copy in g_nw parts on the workers, waited for */
static void copy_parallel(uint8_t *dst, const uint8_t *src, uint32_t bytes)
{
    const int nw = g_nw; copy_job_t j[MAXW]; uint32_t part = (bytes / nw) & ~127u;
    for (int t = 0; t < nw; t++) copy_start(&j[t], dst + t * part, src + t * part, t == nw - 1 ? bytes - (nw - 1) * part : part);
    for (int t = 0; t < nw; t++) st_wait(&j[t].tok);
}

/* ------------------------------------------------------------------ GELU (tanh form, degree-7 polynomial in qf32) */
static const float GQ[8] = { 1.79372174e+00f, -6.00615498e+00f, 1.74950487e+01f, -3.67833718e+01f, 5.28258996e+01f, -4.83375469e+01f, 2.51038333e+01f, -5.59169678e+00f };
static inline HVX_Vector gelu_sf(HVX_Vector x, const HVX_Vector *c, HVX_Vector inv, HVX_Vector one, HVX_Vector mone, HVX_Vector half)
{
    HVX_Vector s = sfv(Q6_Vqf32_vmpy_VsfVsf(x, inv)); s = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(s, one), mone);
    HVX_Vector s2 = Q6_Vqf32_vmpy_VsfVsf(s, s), acc = Q6_Vqf32_vadd_VsfVsf(c[7], Q6_V_vzero());
    for (int k = 6; k >= 0; k--) acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, s2), c[k]);
    HVX_Vector g = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, Q6_Vqf32_vadd_VsfVsf(s, Q6_V_vzero())), half);
    return Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vzero()), g);
}
static void gelu_poly(__fp16 *buf, uint32_t n)
{
    HVX_Vector *v = (HVX_Vector *)buf, c[8]; for (int k = 0; k < 8; k++) c[k] = splat_sf(GQ[k]);
    const HVX_Vector inv = splat_sf(1.f / 4.5f), one = splat_sf(1.f), mone = splat_sf(-1.f), half = splat_sf(0.5f), one_hf = splat_hf(1.f);
#if __HVX_ARCH__ >= 81
#define GELU_PACK(lo_, hi_) sf2_to_hf(sfv(lo_), sfv(hi_))          /* no qf32 pair on v81 (extended state) */
#else
#define GELU_PACK(lo_, hi_) Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi_, lo_))
#endif
    uint32_t i = 0; const uint32_t nv = n / 64;
    if (g_opt & OPT_ACTX) for (; i + 2 <= nv; i += 2)
    {
        const HVX_VectorPair p0 = Q6_Wqf32_vmpy_VhfVhf(v[i], one_hf), p1 = Q6_Wqf32_vmpy_VhfVhf(v[i + 1], one_hf);
        const HVX_Vector l0 = gelu_sf(sfv(Q6_V_lo_W(p0)), c, inv, one, mone, half), h0 = gelu_sf(sfv(Q6_V_hi_W(p0)), c, inv, one, mone, half);
        const HVX_Vector l1 = gelu_sf(sfv(Q6_V_lo_W(p1)), c, inv, one, mone, half), h1 = gelu_sf(sfv(Q6_V_hi_W(p1)), c, inv, one, mone, half);
        v[i] = GELU_PACK(l0, h0); v[i + 1] = GELU_PACK(l1, h1);
    }
    for (; i < nv; i++)
    {
        HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v[i], one_hf);
        HVX_Vector lo = gelu_sf(sfv(Q6_V_lo_W(p)), c, inv, one, mone, half), hi = gelu_sf(sfv(Q6_V_hi_W(p)), c, inv, one, mone, half);
        v[i] = GELU_PACK(lo, hi);
    }
}

/* ------------------------------------------------------------------ products: out[M x N] = act[M x K] W[K x N] */
enum { OUT_ACT, OUT_HEADQ, OUT_HEADV, OUT_ADD, OUT_SPLIT4, OUT_HEADQ64, OUT_HEADV64,   /* ..64: heads of 64 columns, SDXL's */
       OUT_RESG, OUT_ADDRESG };   /* out (x) += g_resg * the product; ADDRESG: g_resx (x) += g_resg * (out + the product), out left as it was */
static const HVX_Vector *g_resg, *g_resb; static uint16_t *g_resx;   /* g_resb: a bias added before the gate (SDXL: no gate, g_resg 0) */
/* g_splitn: the parts OUT_SPLIT4 makes (Klein's 9216 in three: a part must be a whole number of KC columns). g_hM, g_ht0: the head layouts' whole
 * length and first tile, when a product makes only some of the rows (Klein's double blocks: each stream its own weights, one attention) */
static uint32_t g_splitn = 4, g_hM, g_ht0;
static int g_vkv;
static uint32_t g_outc;   /* OUT_ACT: the first g_outc columns only, closed up (a convolution's 96 channels: N is a multiple of 64) */
static inline uint16_t *out_block(uint16_t *out, int mode, uint32_t M, uint32_t N, uint32_t tile, uint32_t cb)
{
    if (mode == OUT_ACT || mode == OUT_ADD || mode >= OUT_RESG) return blk(out, N, tile, cb);
    if (mode == OUT_SPLIT4) { const uint32_t q = N / g_splitn, nq = q / 32; return blk(out + (size_t)(cb / nq) * M * q, q, tile, cb % nq); }
    if (g_hM) { M = g_hM; tile += g_ht0; }
    if (mode == OUT_HEADQ64) return out + ((size_t)((cb / 2) * (M / 32) + tile) * 2 + cb % 2) * BH;
    if (mode == OUT_HEADV64) return out + (size_t)(cb / 2) * M * 64 + (size_t)(cb % 2) * (M / 32) * BH + (size_t)tile * BH;
    uint32_t head = cb / 4, j = cb % 4, nt = M / 32;
    if (mode == OUT_HEADQ) return out + ((size_t)(head * nt + tile) * 4 + j) * BH;
    return out + (size_t)head * M * DH + (size_t)(j / 2) * M * 64 + (size_t)(j % 2) * nt * BH + (size_t)tile * BH;
}
static uint64_t g_tav, g_tqk, g_tpost;   /* counters (us): waits for weights, for post jobs */
/* after the HMX of one output group: GELU (if asked) and the copy out, on a worker, for tiles j0..j1-1 */
typedef struct { const uint8_t *src; uint16_t *out; int mode, gelu; uint32_t M, N, m0, g, j0, j1; spintok_t *tok; const uint8_t *sg; const HVX_Vector *bias;
                 const HVX_Vector *gain, *gaing;
                 const HVX_Vector *biasg;
                 const uint8_t *srcA; const HVX_Vector *rrw; const uint16_t *rrope; } post_job_t;   /* biasg: the gate's bias (SDXL's GEGLU, gelu 4); srcA..: a head's pair of groups */
static void rr_apply(uint16_t *h0, const HVX_Vector *wv, const HVX_Vector *rs, const HVX_Vector *cs);
static void row_stats(const uint16_t *tile0, uint32_t nblk, float n, float eps, HVX_Vector *mean, HVX_Vector *rstd);
static inline HVX_Vector sigmoid_hf(HVX_Vector x)
{
    const HVX_Vector one = splat_hf(1.f), two = splat_hf(2.f);
    HVX_Vector t = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(x, splat_hf(-1.44269504f))), splat_hf(-16.f)), splat_hf(14.f));   /* 2^15 plus rounding would reach inf */
    HVX_Vector i = Q6_Vh_equals_Vhf(Q6_Vh_vadd_VhVh(t, Q6_Vh_vsplat_R(0x2800)));                   /* t * 1024, fixed point */
    HVX_Vector f = Q6_Vh_vasl_VhR(Q6_V_vand_VV(i, Q6_Vh_vsplat_R(0x3ff)), 5);                       /* t - floor(t), Q15 */
    HVX_Vector q = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(f, Q6_Vh_vsplat_R(2535)), Q6_Vh_vsplat_R(7437));
    q = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(q, f), Q6_Vh_vsplat_R(22788)); q = Q6_Vh_vmpy_VhVh_s1_rnd_sat(q, f);
    HVX_Vector m = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q, Q6_Vh_vsplat_R(16)), 5);
    HVX_Vector e = Q6_Vh_vmax_VhVh(Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_vand_VV(i, Q6_Vh_vsplat_R(0xfc00)), m), Q6_Vh_vsplat_R(0x3c00)), Q6_V_vzero());
    HVX_Vector d = hf(Q6_Vqf16_vadd_VhfVhf(e, one));                                                 /* 1 .. 32769 */
    HVX_Vector y = Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(0x7800), d);                                        /* ~1 / d */
    for (int k = 0; k < 2; k++) y = hf(Q6_Vqf16_vmpy_VhfVhf(y, hf(Q6_Vqf16_vsub_VhfVhf(two, hf(Q6_Vqf16_vmpy_VhfVhf(d, y))))));
    return y;
}
/* e^-x of sf lanes: 2^t by exponent bits and a degree-5 polynomial, t clamped to +-60 */
static inline HVX_Vector expn_sf(HVX_Vector x)
{
    const HVX_Vector one = splat_sf(1.f), magic = splat_sf(12582912.f);
    HVX_Vector t = sfv(Q6_Vqf32_vmpy_VsfVsf(x, splat_sf(-1.44269504f)));
    t = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(t, splat_sf(60.f)), splat_sf(-60.f));
    HVX_Vector rq = sfv(Q6_Vqf32_vadd_VsfVsf(t, magic)), n = Q6_Vw_vsub_VwVw(rq, Q6_V_vsplat_R(0x4B400000));
    HVX_Vector f = sfv(Q6_Vqf32_vsub_VsfVsf(t, sfv(Q6_Vqf32_vsub_VsfVsf(rq, magic))));
    static const float c[5] = { 0.00133336f, 0.00961813f, 0.0555041f, 0.240227f, 0.693147f };
    HVX_Vector p = splat_sf(c[0]);
    for (int i = 1; i < 5; i++) p = sfv(Q6_Vqf32_vadd_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(p, f)), splat_sf(c[i])));
    p = sfv(Q6_Vqf32_vadd_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(p, f)), one));
    return sfv(Q6_Vqf32_vmpy_VsfVsf(p, Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(n, Q6_V_vsplat_R(127)), 23)));
}
/* 1 / d (d >= 1) by Newton steps from the bit-trick guess: 2 steps leave ~2e-4, 3 steps fp32's own error */
static inline HVX_Vector rcp_sf(HVX_Vector d, int steps)
{
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), d); const HVX_Vector two = splat_sf(2.f);
    for (int i = 0; i < steps; i++) y = sfv(Q6_Vqf32_vmpy_VsfVsf(y, sfv(Q6_Vqf32_vsub_VsfVsf(two, sfv(Q6_Vqf32_vmpy_VsfVsf(d, y))))));
    return y;
}
/* MOSS-Image's FFN activation on hf lanes, in fp32: 4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25), tanh(y) = 2 sigmoid(2y) - 1 (fp32 so the
 * difference near 0 keeps its digits).
 * Two exponentials for the three sigmoids: e = e^(-g/2) gives sigmoid(g/2) = 1 / (1 + e) and sigmoid(g) = 1 / (1 + e^2); the two tanh
 * terms keep three Newton steps (they are differences near 0), sigmoid(g) a factor, two (a block took 263 ms with three sigmoids) */
static inline HVX_Vector situ_sf(HVX_Vector g, HVX_Vector h)
{
    const HVX_Vector one = splat_sf(1.f);
    HVX_Vector e1 = Q6_Vsf_vmin_VsfVsf(expn_sf(sfv(Q6_Vqf32_vmpy_VsfVsf(g, splat_sf(0.5f)))), splat_sf(1e18f));
    const HVX_Vector s = rcp_sf(sfv(Q6_Vqf32_vadd_VsfVsf(e1, one)), 3);
    const HVX_Vector sg = rcp_sf(sfv(Q6_Vqf32_vadd_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(e1, e1)), one)), 2);
    const HVX_Vector a = sfv(Q6_Vqf32_vsub_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(s, splat_sf(8.f))), splat_sf(4.f)));
    const HVX_Vector sb = rcp_sf(sfv(Q6_Vqf32_vadd_VsfVsf(expn_sf(sfv(Q6_Vqf32_vmpy_VsfVsf(h, splat_sf(0.08f)))), one)), 3);
    const HVX_Vector b = sfv(Q6_Vqf32_vsub_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(sb, splat_sf(50.f))), splat_sf(25.f)));
    return sfv(Q6_Vqf32_vmpy_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(a, sg)), b));
}
/* s tanh(x / s) in fp16: an odd polynomial while |x / s| < 0.5 (tanh = y (1 - y^2/3 + 2y^4/15 - 17y^6/315), off by < 5e-5 there), else
 * 2s sigmoid(2x / s) - s -- the sigmoid form alone loses everything near 0 to the subtraction */
static inline HVX_Vector stanh_hf(HVX_Vector x, float inv_s, float s)
{
    const HVX_Vector y = hf(Q6_Vqf16_vmpy_VhfVhf(x, splat_hf(inv_s))), y2 = hf(Q6_Vqf16_vmpy_VhfVhf(y, y));
    HVX_Vector p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(y2, splat_hf(-17.f / 315.f))), splat_hf(2.f / 15.f)));
    p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(y2, p)), splat_hf(-1.f / 3.f)));
    p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(y2, p)), splat_hf(1.f)));
    const HVX_Vector small = hf(Q6_Vqf16_vmpy_VhfVhf(x, p));
    const HVX_Vector big = hf(Q6_Vqf16_vsub_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(sigmoid_hf(hf(Q6_Vqf16_vmpy_VhfVhf(y, splat_hf(2.f)))), splat_hf(2.f * s))), splat_hf(s)));
    const HVX_Vector ay = Q6_V_vand_VV(y, Q6_Vh_vsplat_R(0x7FFF));                                   /* |y| */
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(ay, splat_hf(0.5f)), big, small);
}
static int g_situ16;      /* MOSS's activation in fp16 (the default; k2run flag bit 25 gives the fp32 one) */
/* the same in fp16: 4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25) */
static inline HVX_Vector situ_hf16(HVX_Vector g, HVX_Vector h)
{
    return hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(stanh_hf(g, 0.25f, 4.f), sigmoid_hf(g))), stanh_hf(h, 0.04f, 25.f)));
}
static inline HVX_Vector situ_hf32(HVX_Vector g, HVX_Vector h)
{
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
    HVX_VectorPair pg = Q6_Wqf32_vmpy_VhfVhf(g, one), ph = Q6_Wqf32_vmpy_VhfVhf(h, one);
    const HVX_Vector ge = sfv(Q6_V_lo_W(pg)), go = sfv(Q6_V_hi_W(pg)), he = sfv(Q6_V_lo_W(ph)), ho = sfv(Q6_V_hi_W(ph));
    return sf2_to_hf(situ_sf(ge, he), situ_sf(go, ho));
}
static inline HVX_Vector situ_hf(HVX_Vector g, HVX_Vector h) { return g_situ16 ? situ_hf16(g, h) : situ_hf32(g, h); }
static void post_body(post_job_t *p)
{
    for (uint32_t j = p->j0; j < p->j1; j++)
    {
        const uint8_t *src = p->src + j * 2 * BLK;
        if (p->gain) for (int hb = 0; hb < 2; hb++)  /* a DoRA's gain on W + A B: not on the Linear's bias, as ComfyUI leaves that alone */
        { HVX_Vector *v = (HVX_Vector *)(src + hb * BLK); const HVX_Vector g_ = p->gain[2 * p->g + hb]; for (int k = 0; k < 16; k++) v[k] = hf(Q6_Vqf16_vmpy_VhfVhf(v[k], g_)); }
        if (p->gaing) for (int hb = 0; hb < 2; hb++)
        { HVX_Vector *v = (HVX_Vector *)(p->sg + j * 2 * BLK + hb * BLK); const HVX_Vector g_ = p->gaing[2 * p->g + hb]; for (int k = 0; k < 16; k++) v[k] = hf(Q6_Vqf16_vmpy_VhfVhf(v[k], g_)); }
        if (p->bias)                                /* the bias, per column (its column vectors), before the activation */
            for (int hb = 0; hb < 2; hb++)
            {
                HVX_Vector *v = (HVX_Vector *)(src + hb * BLK); const HVX_Vector b = p->bias[2 * p->g + hb];
                for (int k = 0; k < 16; k++) v[k] = hf(Q6_Vqf16_vadd_VhfVhf(v[k], b));
            }
        if (p->gelu >= 2)
        {   /* swiglu: silu(gate) * up, into the up block; 3: MOSS's situ instead */
            HVX_Vector *u = (HVX_Vector *)src; const HVX_Vector *gv = (const HVX_Vector *)(p->sg + j * 2 * BLK);
            if (p->biasg) for (int hb = 0; hb < 2; hb++)
            { HVX_Vector *g2 = (HVX_Vector *)(p->sg + j * 2 * BLK + hb * BLK); const HVX_Vector b = p->biasg[2 * p->g + hb]; for (int k = 0; k < 16; k++) g2[k] = hf(Q6_Vqf16_vadd_VhfVhf(g2[k], b)); }
            if (p->gelu == 4)
            {   /* SDXL's GEGLU: up * gelu(gate), here instead of a pass over both halves in DDR (0.55 s of a 2.98 s UNet pass in a separate pass) */
                HVX_Vector cf[8]; for (int k = 0; k < 8; k++) cf[k] = splat_sf(GQ[k]);
                const HVX_Vector inv = splat_sf(1.f / 4.5f), one = splat_sf(1.f), mone = splat_sf(-1.f), half = splat_sf(0.5f), one_hf = splat_hf(1.f);
                for (int k = 0; k < 32; k++)
                {
                    HVX_VectorPair pp = Q6_Wqf32_vmpy_VhfVhf(gv[k], one_hf);
                    HVX_Vector lo = gelu_sf(sfv(Q6_V_lo_W(pp)), cf, inv, one, mone, half), hi = gelu_sf(sfv(Q6_V_hi_W(pp)), cf, inv, one, mone, half), g1;
#if __HVX_ARCH__ >= 81
                    g1 = sf2_to_hf(sfv(lo), sfv(hi));
#else
                    g1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
#endif
                    u[k] = hfr(Q6_Vqf16_vmpy_VhfVhf(u[k], g1));   /* the OUT_ACT store below clamps */
                }
            }
            else if (p->gelu == 3)
            {
                if (!(g_opt & OPT_ACTX)) for (int k = 0; k < 32; k++) u[k] = situ_hf(gv[k], u[k]);
                else if (g_situ16) for (int k = 0; k < 32; k += 2) { const HVX_Vector a0 = situ_hf16(gv[k], u[k]), a1 = situ_hf16(gv[k + 1], u[k + 1]); u[k] = a0; u[k + 1] = a1; }
                else for (int k = 0; k < 32; k += 2) { const HVX_Vector a0 = situ_hf32(gv[k], u[k]), a1 = situ_hf32(gv[k + 1], u[k + 1]); u[k] = a0; u[k + 1] = a1; }
            }
#define SILU_MUL(g_, u_) hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf((g_), sigmoid_hf(g_))), (u_)))
            else if (g_opt & OPT_ACTX) for (int k = 0; k < 32; k += 2) { const HVX_Vector a0 = SILU_MUL(gv[k], u[k]), a1 = SILU_MUL(gv[k + 1], u[k + 1]); u[k] = a0; u[k + 1] = a1; }
            else for (int k = 0; k < 32; k++) u[k] = SILU_MUL(gv[k], u[k]);
        }
        else if (p->gelu) gelu_poly((__fp16 *)src, 2 * BH);
#ifdef HF_CLAMP
        if (p->mode == OUT_ACT) { HVX_Vector *d = (HVX_Vector *)blk(p->out, g_outc ? g_outc : p->N, p->m0 + j, 2 * p->g); const HVX_Vector *a = (const HVX_Vector *)src;
                                  const int nv = g_outc && g_outc < (2 * p->g + 2) * 32 ? 16 : 32; for (int k = 0; k < nv; k++) d[k] = clamp_hf(a[k]); }
#else
        if (p->mode == OUT_ACT) hvx_copy((uint8_t *)blk(p->out, g_outc ? g_outc : p->N, p->m0 + j, 2 * p->g), src, g_outc && g_outc < (2 * p->g + 2) * 32 ? BLK : 2 * BLK);
#endif
        else if (p->mode == OUT_ADD)
        {
            HVX_Vector *d = (HVX_Vector *)blk(p->out, p->N, p->m0 + j, 2 * p->g); const HVX_Vector *a = (const HVX_Vector *)src;
            if (j + 1 < p->j1) Q6_l2fetch_AP((void *)blk(p->out, p->N, p->m0 + j + 1, 2 * p->g), ((uint64_t)4096 << 32) | ((uint64_t)4096 << 16) | 1);
            for (int k = 0; k < 32; k++) d[k] = hf(Q6_Vqf16_vadd_VhfVhf(d[k], a[k]));
        }
        else if (p->mode >= OUT_RESG)
        {   /* the same values as OUT_ACT / OUT_ADD into y and then residual()'s x += gate * y, without y's trip through DDR */
            const HVX_Vector *a = (const HVX_Vector *)src, *yo = (const HVX_Vector *)blk(p->out, p->N, p->m0 + j, 2 * p->g);
            HVX_Vector *xo = (HVX_Vector *)blk(p->mode == OUT_RESG ? p->out : g_resx, p->N, p->m0 + j, 2 * p->g);
            const HVX_Vector g0 = g_resg ? g_resg[2 * p->g] : Q6_V_vzero(), g1 = g_resg ? g_resg[2 * p->g + 1] : Q6_V_vzero();
            for (int k = 0; k < 32; k++)
            {
#ifdef HF_CLAMP
                const HVX_Vector y = p->mode == OUT_RESG ? a[k] : hfr(Q6_Vqf16_vadd_VhfVhf(yo[k], a[k]));   /* x's sum below clamps */
#else
                const HVX_Vector y = p->mode == OUT_RESG ? a[k] : hf(Q6_Vqf16_vadd_VhfVhf(yo[k], a[k]));
#endif
                const HVX_Vector yb_ = g_resb ? hfr(Q6_Vqf16_vadd_VhfVhf(y, g_resb[2 * p->g + (k >> 4)])) : y;   /* as vk_bias then vk_add */
                xo[k] = FIN(hf(Q6_Vqf16_vadd_VhfVhf(xo[k], g_resg ? hfr(Q6_Vqf16_vmpy_VhfVhf(yb_, k < 16 ? g0 : g1)) : yb_)));
            }
        }
        else for (int hb = 0; hb < 2; hb++) hvx_copy((uint8_t *)out_block(p->out, p->mode, p->M, p->N, p->m0 + j, 2 * p->g + hb), src + hb * BLK, BLK);
    }
}
static void post_worker(void *a)
{
    post_job_t *p = (post_job_t *)a;
    if (p->srcA)
    {   /* a head's first group (waiting in srcA) and its second written out, then the head's RMS norm and RoPE on those 8 KB as
           rms_rope does them (rr_apply), while they are still in the cache */
        post_job_t pa = *p; pa.src = p->srcA; pa.g = p->g - 1; post_body(&pa); post_body(p);
        for (uint32_t j = p->j0; j < p->j1; j++)
        {
            const uint32_t t = p->m0 + j; uint16_t *h0 = out_block(p->out, p->mode, p->M, p->N, t, 2 * (p->g - 1));
            HVX_Vector rs[16]; row_stats(h0, 4, DH, 1e-6f, 0, rs);
            rr_apply(h0, p->rrw, rs, p->rrope ? (const HVX_Vector *)blk((uint16_t *)p->rrope, 128, t, 0) : 0);
        }
    }
    else
    {
        post_body(p);
        if (p->rrw && (p->g & 1))
        {   /* when the next group's posts start only after these are done: the head's first half is
               out already -- norm and RoPE the head here (the same rr_apply) */
            for (uint32_t j = p->j0; j < p->j1; j++)
            {
                const uint32_t t = p->m0 + j; uint16_t *h0 = out_block(p->out, p->mode, p->M, p->N, t, 2 * (p->g - 1));
                HVX_Vector rs[16]; row_stats(h0, 4, DH, 1e-6f, 0, rs);
                rr_apply(h0, p->rrw, rs, p->rrope ? (const HVX_Vector *)blk((uint16_t *)p->rrope, 128, t, 0) : 0);
            }
        }
    }
    st_done(p->tok);
}
/* the weight stream on WCOPY workers (one: two copying threads were slower, DDR bound) */
#define WCOPY 1
#define POSTW (g_nw - (((g_opt & OPT_POSTALL) || ((g_opt & OPT_WDMA) && !g_wfmt)) ? 0 : WCOPY))   /* with the DMA the copier's context is free for the posts; OPT_POSTALL puts the weight copier's context to post work too */
/* ------------------------------------------------------------------ quantized weights, made fp16 on their way into the VTCM
 * g_wfmt (from the run's flags): 0 fp16 / 1 q8 / 2 q4. A matrix [K x N] is its values in the fp16 weight layout's element order,
 * one byte each (q8: 8-bit integers) or half a byte (q4: per 128-byte vector, element i of a 256-element run in the low nibble and
 * element 128 + i in the high one, both + 8), then its scales: per 32 x 32 block, the 32 columns' fp16 scales, blocks in layout
 * order. That is GGUF's Q8_0 / Q4_0 grouping (32 K values of one column share a scale), so those files only need reordering. */
static int g_wfmt = 0;
/* bytes a matrix of n fp16 values takes in the current format */
/* n + n / 16, not n * 17 / 16: size_t is 32 bits on the DSP, and a whole Krea block (434M values) times 17 wrapped -- the LoRA after
 * the weights was then looked for in the wrong place with q8 / q4 */
static inline size_t wbytes(size_t n) { return g_wfmt == 1 ? n + n / 16 : g_wfmt == 2 ? n / 2 + n / 16 : n * 2; }
/* a matrix's start, n values after another (the matrices of a block are stored one after another) */
static inline const uint16_t *WP(const uint16_t *base, size_t n) { return (const uint16_t *)((const uint8_t *)base + wbytes(n)); }
typedef struct { uint8_t *dst; const uint8_t *q[2]; const uint8_t *sc[2]; uint32_t nblk, nh; spintok_t tok; } deq_job_t;   /* nh halves of a group (1 or 2) */
static void deq_worker(void *a)
{
    deq_job_t *j = (deq_job_t *)a; const uint32_t qb = g_wfmt == 1 ? 1024 : 512;
    const HVX_Vector off8 = Q6_Vb_vsplat_R(8), lo4 = Q6_Vb_vsplat_R(15);
    for (uint32_t hb = 0; hb < j->nh * j->nblk; hb++)
    {
        const uint32_t h = hb >= j->nblk, b = hb - h * j->nblk;
        if (b + 4 < j->nblk) Q6_l2fetch_AP((void *)(j->q[h] + (b + 4) * qb), ((uint64_t)qb << 32) | ((uint64_t)qb << 16) | 1);
        /* the block's 32 column scales, each in the two lanes its column has in every vector of the block */
        /* (an aligned load and a rotate: an unaligned one ran 64 bytes past the end of the weights after the last block) */
        const uint8_t *sp = j->sc[h] + b * 64; HVX_Vector s = *(const HVX_Vector *)((uintptr_t)sp & ~(uintptr_t)127);
        if ((uintptr_t)sp & 64) s = Q6_V_vror_VR(s, 64);
        const HVX_Vector sv = Q6_V_lo_W(Q6_W_vshuff_VVR(s, s, -2));
        const HVX_Vector *q = (const HVX_Vector *)(j->q[h] + b * qb); HVX_Vector *d = (HVX_Vector *)(j->dst + hb * BLK);
        if (g_wfmt == 1)
            for (int k = 0; k < 8; k++)
            {
                HVX_VectorPair w = Q6_Wh_vunpack_Vb(q[k]);
                d[2 * k] = hf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(w)), sv));
                d[2 * k + 1] = hf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(w)), sv));
            }
        else
            for (int k = 0; k < 4; k++)
            {
                HVX_Vector lo = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(q[k], lo4), off8), hi = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(q[k], 4), off8);
                HVX_VectorPair wl = Q6_Wh_vunpack_Vb(lo), wh = Q6_Wh_vunpack_Vb(hi);
                d[4 * k] = hf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(wl)), sv));
                d[4 * k + 1] = hf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(wl)), sv));
                d[4 * k + 2] = hf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(wh)), sv));
                d[4 * k + 3] = hf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(wh)), sv));
            }
    }
    st_done(&j->tok);
}
static void deq_start(deq_job_t *j, uint8_t *dst, const uint8_t *q0, const uint8_t *sc0, const uint8_t *q1, const uint8_t *sc1, uint32_t nblk)
{ j->dst = dst; j->q[0] = q0; j->sc[0] = sc0; j->q[1] = q1; j->sc[1] = sc1; j->nblk = nblk; j->nh = q1 ? 2 : 1; st_init(&j->tok, 1); worker_pool_job_t job = { deq_worker, j }; worker_pool_submit(g_pool, job); }

#ifndef DEQJ
#define DEQJ 4                             /* dequantizing jobs a weight group (q8 / q4): 2 kept the HMX waiting on v73 (Krea q8 251 ms a block) */
#endif
typedef struct { copy_job_t h[2]; deq_job_t d[DEQJ]; int n, deq; volatile uint32_t *dd[2]; } wcopy_t;   /* deq 2: the user DMA's descriptors dd */
/* the user DMA (dmlink / dmpoll). Descriptor layouts as llama.cpp's ggml-hexagon htp/dma-queue.h (MIT): v73 a 1D one (size 24
 * bits, the flags, done = bit 31 of word 1), v75 on its 2D one as one row (desc_type 9, 24-bit addresses). In a static ring (the engine
 * follows next pointers, so never on a stack that is gone); each call starts its chain from a fresh zeroed tail (g_dtail), as the
 * llama.cpp queue starts, since a call may arrive on another DSP thread. */
typedef struct { void *next; uint32_t w1, src, dst, w4, w5, w6, w7; } kdesc_t;
static kdesc_t g_dring[16] __attribute__((aligned(64))), g_dzero __attribute__((aligned(64))); static uint32_t g_di; static void *g_dtail;
static volatile uint32_t *kdma_copy(uint8_t *dst, const uint8_t *src, uint32_t bytes)
{
    kdesc_t *d = &g_dring[g_di++ & 15]; memset(d, 0, sizeof(*d)); d->src = (uint32_t)(uintptr_t)src; d->dst = (uint32_t)(uintptr_t)dst;
#if __HVX_ARCH__ < 75
    /* the source read through the caches, only the VTCM side bypassed: with src bypass the DMA read DDR around what the CPU had just written
       there (the app refills the weights every call; its pictures came out different run to run) */
    d->w1 = (bytes & 0xFFFFFF) | (1u << 28);                                        /* size, 1D, dst bypass, not done */
#else
    d->w1 = (bytes & 0xFFFFFF) | (1u << 24) | (1u << 28);                           /* dst_stride = bytes, 2D, dst bypass (src through the caches, as v73) */
    d->w4 = 9;                                                                      /* desc_type 9: 2D, 24-bit addresses */
    d->w5 = (bytes & 0xFFFFFF) | (1u << 24);                                        /* row_size, nrows_lo 1 */
    d->w6 = (bytes & 0xFFFFFF) << 8;                                                /* nrows_hi 0, src_stride */
#endif
    if (!g_dtail) { memset(&g_dzero, 0, sizeof(g_dzero)); g_dtail = &g_dzero; }
    asm volatile("release(%0):at" :: "r"(d) : "memory");   /* the descriptor's stores out to where the DMA reads it (llama.cpp does the same) */
    asm volatile("dmlink(%0, %1)" :: "r"(g_dtail), "r"(d) : "memory"); g_dtail = d;
    return &d->w1;
}
/* the descriptor's done bit read past the DSP's cache (its line invalidated before each look): a cached copy let the HMX start on a group
   the DMA had not finished (with this and the src change, the pictures are the same as without the DMA) */
static void kdma_wait(volatile uint32_t *w1) { for (;;) { asm volatile("dcinva(%0)" :: "r"(w1) : "memory"); if (*w1 >> 31) break; unsigned r; asm volatile("%0 = dmpoll" : "=r"(r) :: "memory"); (void)r; } }
static void wcopy_start(wcopy_t *c, uint8_t *dst, const uint8_t *src, uint32_t bytes)
{
    if (g_opt & OPT_WDMA) { c->n = 1; c->deq = 2; c->dd[0] = kdma_copy(dst, src, bytes); return; }
    c->n = WCOPY; c->deq = 0; uint32_t a = c->n == 1 ? bytes : (bytes / 2) & ~127u;
    copy_start(&c->h[0], dst, src, a); if (c->n == 2) copy_start(&c->h[1], dst + a, src + a, bytes - a);
}
static void wcopy_wait(wcopy_t *c) { for (int i = 0; i < c->n; i++) { if (c->deq == 2) kdma_wait(c->dd[i]); else st_wait(c->deq ? &c->d[i].tok : &c->h[i].tok); } }
/* one weight matrix of a product: W [K x N] in the weight layout, where its output goes and how it is laid out */
typedef struct { const uint16_t *w; uint32_t N; uint16_t *out; int mode; uint32_t wk, k0; const HVX_Vector *bias;
    /* a LoRA applied as it runs, instead of baked into w: la [K x LR] and lb [LR x N] in the weight layout, the strength in
     * g_lora. y = act w + strength (act la) lb -- rank / N + rank / K more work, and w is left alone (so one copy of the
     * weights serves every LoRA and strength, and a strength may change between steps) */
    const uint16_t *la, *lb;
    const HVX_Vector *rrw; const uint16_t *rrope; } wset_t;   /* (OUT_HEADQ): the head norm's 4 column vectors (weights x factor), the RoPE table or 0 */
/* the rank of a LoRA that carries no { lr, gain } header: never in this build (every caller sets g_lhdr, so the header's rank is used);
 * 64, a multiple of 64 and at most KC = 512 (one HMX chunk) */
static uint32_t g_lr = 64;
#define LR g_lr
static float g_lora = 1.f;
/* The LoRAs as they run, after the weights, Linear by Linear (lrun.inc): 64 halfwords { lr, gain }, then with lr > 0: A^T [K x lr], the same 64 halfwords, B^T [lr x N], with gain N / 32 column
 * vectors (a DoRA: W' = gain (W + A B)). So each Linear costs its own rank, one without a LoRA nothing. g_lhdr: this run's are so (else the fixed g_lr) */
static int g_lhdr;
#define WLR(w_) (g_lhdr ? (uint32_t)(w_).lb[-64] : g_lr)
#define WS_GAIN(w_) (g_lhdr && (w_).la && (w_).lb[-63] ? (const HVX_Vector *)((w_).lb + (size_t)(w_).lb[-64] * (w_).N) : 0)
static const uint16_t *lora_next(const uint16_t *p, uint32_t K, uint32_t N, const uint16_t **la, const uint16_t **lb)
{
    const uint32_t r = p[0], g = p[1]; *la = *lb = 0; p += 64; if (!r) return p;
    *la = p; p += (size_t)K * r + 64; *lb = p; p += (size_t)r * N; if (g) p += (size_t)N * 2; return p;
}
/* starts copying group g of a weight set: one run of 2 x 32 columns x K rows, or with wk, the K-row slice of each half */
static void wcopy_group(wcopy_t *c, uint8_t *dst, const wset_t *w, uint32_t g, uint32_t K)
{
    const uint32_t wk = w->wk ? w->wk : K;
    if (g_wfmt)
    {   /* quantized: each half's blocks made fp16 by a worker, the K slice (if any) picked out of the stored K */
        const uint32_t qb = g_wfmt == 1 ? 1024 : 512, nbs = wk / 32, kb0 = w->k0 / 32, nb = K / 32;
        const uint8_t *q = (const uint8_t *)w->w, *sc = q + (size_t)wk * w->N / (g_wfmt == 1 ? 1 : 2);   /* not * qb / 1024: size_t is 32 bits here, 2048 x 2048 x 1024 wrapped to 0 */
        c->n = DEQJ; c->deq = 1;                                              /* DEQJ / 2 jobs per half */
        const uint32_t per = DEQJ / 2, pb = (nb + per - 1) / per;
        for (int h = 0; h < 2; h++)
            for (uint32_t i = 0; i < per; i++)
            {
                const size_t bh = (size_t)g * 2 * nbs + kb0 + h * nbs + i * pb;   /* this job's first block in layout order */
                const uint32_t n_ = i * pb >= nb ? 0 : (nb - i * pb < pb ? nb - i * pb : pb);
                deq_start(&c->d[h * per + i], dst + ((size_t)h * nb + i * pb) * BLK, q + bh * qb, sc + bh * 64, 0, 0, n_);
            }
        return;
    }
    const uint8_t *src = (const uint8_t *)w->w + (size_t)g * wk * 128;
    if (wk == K) { wcopy_start(c, dst, src, K * 128); return; }
    const uint32_t hb = (K / 32) * BLK, off = (w->k0 / 32) * BLK, sh = (wk / 32) * BLK;
    if (g_opt & OPT_WDMA) { c->n = 2; c->deq = 2; c->dd[0] = kdma_copy(dst, src + off, hb); c->dd[1] = kdma_copy(dst + hb, src + sh + off, hb); return; }
    c->n = 2; c->deq = 0; copy_start(&c->h[0], dst, src + off, hb); copy_start(&c->h[1], dst + hb, src + sh + off, hb);
}
/* fills the VTCM activation tiles m0..m0+tiles-1 (instead of copying them from act) */
typedef void (*act_fill_t)(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles);
/* OPT_PREFILL: the next batch's activation tiles made by a worker while the HMX runs this batch -- into a second VTCM buffer, or when VTCM has
 * no room for it into a DDR staging buffer copied in afterwards; the LayerNorm fill (Krea 46 ms a block on v73) no longer holds the HMX up */
typedef struct { act_fill_t fill; void *farg; uint8_t *dst; uint32_t m0, tiles; spintok_t tok; } prefill_job_t;
static void prefill_worker(void *a) { prefill_job_t *j = (prefill_job_t *)a; j->fill(j->farg, j->dst, j->m0, j->tiles); st_done(&j->tok); }
static uint8_t *g_stg; static size_t g_stg_bytes;     /* the staging buffer, kept between calls (allocating it each time cost Anima 10%) */
typedef void (*tile_fn)(void *arg, uint32_t t0, uint32_t t1);
static void tiles_parallel_range(tile_fn fn, void *arg, uint32_t a, uint32_t b);
static void ln_fill(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles);
/* out_i = act W_i for each weight set, the activation tiles loaded (or made by fill) once for all of them */
static uint32_t g_kc;
static void product_multi(const uint16_t *act, act_fill_t fill, void *farg, uint32_t M, uint32_t K, const wset_t *ws, int nw, uint32_t mt, int gelu, uint64_t *t_hmx)
{
    uint8_t *vtcm = g_vtcm;
    int nlora = 0; for (int i = 0; i < nw; i++) if (ws[i].la) nlora++;
    const uint32_t tile_bytes = 32 * K * 2, wt_bytes = K * 64 * 2, half = (K / 32) * BLK;
    /* with a LoRA: one [32 x LR] tile per set for act la (LR / 64 groups) and the lb group being used -- its share is accumulated in the
       product itself (hmx_mpy_chunks_lora), so no output of its own any more: that room went to more tiles a batch */
    uint32_t lrm = 0; for (int i = 0; i < nw; i++) if (ws[i].la && WLR(ws[i]) > lrm) lrm = WLR(ws[i]);   /* room for the set with the most */
    const uint32_t lt_bytes = lrm / 64 * 2 * BLK;
    const uint32_t lora_bytes = nlora ? (uint32_t)(nw * lt_bytes) : 0;     /* act la per set */
    /* output groups in VTCM: two (one in the HMX, one in its post jobs); a SwiGLU's gate and up pairs four, two pairs, so the next pair's
       HMX runs while the last pair's post does -- with two the gate's HMX waited for that post (MOSS's up stage 24 ms over its HMX) */
    int anyrr = 0; for (int i = 0; i < nw; i++) if (ws[i].rrw) anyrr = 1;
    const int nout = gelu >= 2 || anyrr ? 4 : 2;       /* a head's pairs: four buffers, two pairs, as the SwiGLU's */
    /* with a LoRA each weight buffer has lb's group after it, copied ahead with the weights (the main thread copied it before each
       group's HMX, which cost MOSS's up stage 23 ms more with the DMD LoRA) */
    const uint32_t lbx = nlora ? lrm * 128 : 0, wstride = wt_bytes + lbx;
    /* OPT_PREFILL: two activation buffers in VTCM (the next batch made into the other while the HMX reads this one) */
    const int pv = (g_opt & OPT_PREFILL) && fill && !nlora && M / 32 > 1;
    while (mt > 1 && mt * (tile_bytes * (1 + pv) + lora_bytes) + 2 * wstride + 8192 + nout * mt * 2 * BLK > VTCM_BYTES - 65536) mt--;
    uint8_t *v_act0 = vtcm, *v_act1 = vtcm + (size_t)mt * tile_bytes, *v_act = v_act0;
    uint8_t *v_w[2] = { vtcm + (size_t)(1 + pv) * mt * tile_bytes, vtcm + (size_t)(1 + pv) * mt * tile_bytes + wstride }, *v_outs[4];
    for (int o_ = 0; o_ < 4; o_++) v_outs[o_] = v_w[1] + wstride + (size_t)(o_ < nout ? o_ : 0) * mt * 2 * BLK;
    uint8_t *v_lt = v_w[1] + wstride + (size_t)nout * mt * 2 * BLK;   /* act la per set */
    copy_job_t lbj[2]; int lbon[2] = { 0, 0 };
    #define LB_START(b_, set_, grp_) do { if (ws[set_].la) { copy_start(&lbj[b_], v_w[b_] + wt_bytes, (const uint8_t *)ws[set_].lb + (size_t)(grp_) * WLR(ws[set_]) * 128, WLR(ws[set_]) * 128); lbon[b_] = 1; } } while (0)
    post_job_t pj[MAXW]; spintok_t ptok; int posting = 0;
    post_job_t pj2[2][MAXW]; spintok_t tok2[2]; int posting2[2] = { 0, 0 };     /* SwiGLU: a pair slot's post jobs */
    const uint32_t kc_ = g_kc ? g_kc : KC;   /* g_kc: a K that is no multiple of KC (the convolutions') */
    const uint32_t act_ctrl = ((kc_ >> 5) << 11) | 0x7C, wgt_ctrl = ((kc_ >> 5) << 11) - 1, nchunks = K / kc_, chunk = (kc_ / 32) * BLK;
    uint8_t *stg = 0;                                     /* only with three batches or more: with one or two there is little to hide */
    if (fill && !pv && (g_opt & OPT_PREFILL) && M / 32 > mt)
    {
        if (g_stg_bytes < (size_t)mt * tile_bytes) { free(g_stg); g_stg_bytes = (size_t)mt * tile_bytes; g_stg = (uint8_t *)memalign(128, g_stg_bytes); if (!g_stg) g_stg_bytes = 0; }
        stg = g_stg;
    }
    prefill_job_t pf; int pf_on = 0;
    for (uint32_t m0 = 0; m0 < M / 32; m0 += mt)
    {
        uint32_t tiles = (M / 32 - m0) < mt ? (M / 32 - m0) : mt;
        wcopy_t cj;
        if (!nlora) wcopy_group(&cj, v_w[0], &ws[0], 0, K);      /* the first group's weights while the activations are copied */
        uint64_t c0_ = HAP_perf_get_time_us();
        if (fill && pf_on && pf.m0 == m0) { st_wait(&pf.tok); pf_on = 0; if (pv) v_act = pf.dst; else copy_parallel(v_act, stg, tiles * tile_bytes); }
        else if (fill) fill(farg, v_act, m0, tiles); else copy_parallel(v_act, (const uint8_t *)act + (size_t)m0 * tile_bytes, tiles * tile_bytes);
        g_tqk += HAP_perf_get_time_us() - c0_;
        if ((stg || pv) && m0 + mt < M / 32)
        {   /* the next batch's tiles, made while this batch's HMX runs (pv: into the VTCM buffer this batch is not in) */
            pf = (prefill_job_t){ fill, farg, pv ? (v_act == v_act0 ? v_act1 : v_act0) : stg, m0 + mt, (M / 32 - m0 - mt) < mt ? (M / 32 - m0 - mt) : mt };
            st_init(&pf.tok, 1); pf_on = 1; worker_pool_job_t job = { prefill_worker, &pf }; worker_pool_submit(g_pool, job);
        }
        if (nlora)
        {   /* act la [tiles x LR] per set, once for all this batch's groups, with the strength folded in */
            const HVX_Vector sv = splat_hf(g_lora);
            for (int wi2 = 0; wi2 < nw; wi2++)
            {
                if (!ws[wi2].la) continue;
                uint8_t *t_ = v_lt + (size_t)wi2 * mt * lt_bytes;
                const uint32_t wk2 = ws[wi2].wk ? ws[wi2].wk : K;                       /* la is lr / 64 groups of 64 columns, stored with wk rows */
                for (uint32_t gg = 0; gg < WLR(ws[wi2]) / 64; gg++)
                {
                    const uint8_t *la_g = (const uint8_t *)ws[wi2].la + (size_t)gg * wk2 * 128;
                    if (wk2 == K) hvx_copy(v_w[0], la_g, K * 128);
                    else { const uint32_t hb2 = (K / 32) * BLK, off2 = (ws[wi2].k0 / 32) * BLK, sh2 = (wk2 / 32) * BLK;
                           hvx_copy(v_w[0], la_g + off2, hb2);
                           hvx_copy(v_w[0] + hb2, la_g + sh2 + off2, hb2); }
                    for (uint32_t j = 0; j < tiles; j++)
                    {
                        /* a tile's act la is [32 x LR] in the activation layout: group gg's two halves are its blocks 2gg, 2gg + 1 */
                        uint8_t *ap = v_act + j * tile_bytes, *o = t_ + j * lt_bytes + gg * 2 * BLK;
                        hmx_mpy_chunks(ap, act_ctrl, v_w[0], wgt_ctrl, o, 0, g_bias, chunk, nchunks);
                        hmx_mpy_chunks(ap, act_ctrl, v_w[0] + half, wgt_ctrl, o + BLK, 0, g_bias, chunk, nchunks);
                    }
                }
                if (g_lora != 1.f) { HVX_Vector *v = (HVX_Vector *)t_; for (uint32_t k = 0; k < tiles * lt_bytes / 128; k++) v[k] = mul_hf(v[k], sv); }
            }
            wcopy_group(&cj, v_w[0], &ws[0], 0, K);              /* la borrowed v_w[0], so the weights follow it */
            LB_START(0, 0, 0);
        }
        /* the groups of all weight sets in one run: group gi is group g of set wi */
        const int sw = gelu >= 2;                                         /* swiglu pairs: wi alternates, g advances every two */
        for (uint32_t gi = 0, wi = 0, g = 0; wi < (uint32_t)nw; gi++)
        {
            const uint32_t N = ws[wi].N; uint16_t *out = ws[sw ? 0 : wi].out; const int mode = ws[wi].mode;
            const int slot = (gi >> 1) & 1;                                   /* SwiGLU: this pair's buffers */
            const int pr = !sw && ws[wi].rrw != 0, ps_ = (int)((g >> 1) & 1);  /* groups 2h, 2h + 1 (head h) a pair, slot h & 1 */
            uint8_t *w = v_w[gi & 1], *v_out = sw ? v_outs[2 * slot + wi] : pr ? v_outs[2 * ps_ + (g & 1)] : v_outs[gi & 1];
            uint64_t w0_ = HAP_perf_get_time_us(); wcopy_wait(&cj); g_tav += HAP_perf_get_time_us() - w0_;
            uint32_t nwi = wi, ng = g + 1; if (ng == N / 64) { nwi++; ng = 0; }
            if (sw) { nwi = wi ^ 1; ng = wi ? g + 1 : g; if (ng == N / 64) nwi = (uint32_t)nw; }
            /* swiglu: this gate group's HMX overwrites the buffers of the pair two back, whose post may still be reading them */
            if (sw && wi == 0 && posting2[slot]) { uint64_t p0_ = HAP_perf_get_time_us(); st_wait(&tok2[slot]); posting2[slot] = 0; g_tpost += HAP_perf_get_time_us() - p0_; }
            if (pr && !(g & 1))
            {   /* a head's first group into its slot: the pair two heads back and any plain group's post must have left those buffers */
                uint64_t p0_ = HAP_perf_get_time_us();
                if (posting) { st_wait(&ptok); posting = 0; }
                if (posting2[ps_]) { st_wait(&tok2[ps_]); posting2[ps_] = 0; }
                g_tpost += HAP_perf_get_time_us() - p0_;
            }
            if (!sw && !pr && (posting2[0] || posting2[1]))
            {   /* a plain group after paired ones: its buffer is slot 0's */
                uint64_t p0_ = HAP_perf_get_time_us();
                for (int s2 = 0; s2 < 2; s2++) if (posting2[s2]) { st_wait(&tok2[s2]); posting2[s2] = 0; }
                g_tpost += HAP_perf_get_time_us() - p0_;
            }
            if (nwi < (uint32_t)nw) { wcopy_group(&cj, v_w[(gi + 1) & 1], &ws[nwi], ng, K); LB_START((gi + 1) & 1, nwi, ng); }
            uint64_t t0 = HAP_perf_get_time_us();
            if (ws[wi].la)
            {   /* with a LoRA: (act la) lb for this group of 64 columns accumulated right after W's chunks, before the one conversion
                   (hmx_mpy_chunks_lora: one block a load, as a deep load of LR / 32 blocks came out wrong on v81). It had its
                   own output and conversion per group and tile, then an add in the post (a MOSS block with the DMD LoRA took 270 ms that way) */
                if (lbon[gi & 1]) { st_wait(&lbj[gi & 1].tok); lbon[gi & 1] = 0; }
                uint8_t *t_ = v_lt + (size_t)wi * mt * lt_bytes, *v_lb = w + wt_bytes; const uint32_t lr_ = WLR(ws[wi]), lhalf = (lr_ / 32) * BLK;   /* lr_ / 32 single-block loads */
                for (uint32_t j = 0; j < tiles; j++)
                {
                    uint8_t *ap = v_act + j * tile_bytes, *o = v_out + j * 2 * BLK, *lp = t_ + j * lt_bytes;
#if __HEXAGON_ARCH__ < 81
                    if ((g_opt & OPT_LORAD) && lr_ < 1024)
                    {
                        hmx_mpy_chunks_lorad(ap, act_ctrl, w, wgt_ctrl, o, 0, g_bias, chunk, nchunks, lp, v_lb, lr_ / 32);
                        hmx_mpy_chunks_lorad(ap, act_ctrl, w + half, wgt_ctrl, o + BLK, 0, g_bias, chunk, nchunks, lp, v_lb + lhalf, lr_ / 32);
                        continue;
                    }
#endif
                    hmx_mpy_chunks_lora(ap, act_ctrl, w, wgt_ctrl, o, 0, g_bias, chunk, nchunks, lp, v_lb, lr_ / 32);
                    hmx_mpy_chunks_lora(ap, act_ctrl, w + half, wgt_ctrl, o + BLK, 0, g_bias, chunk, nchunks, lp, v_lb + lhalf, lr_ / 32);
                }
            }
            else if (g_opt & OPT_HMXG) hmx_mpy_group(v_act, act_ctrl, w, wgt_ctrl, v_out, tile_bytes, g_bias, chunk, nchunks, tiles, half);
            else
                for (uint32_t j = 0; j < tiles; j++)
                {
                    uint8_t *ap = v_act + j * tile_bytes, *o = v_out + j * 2 * BLK;
                    hmx_mpy_chunks(ap, act_ctrl, w, wgt_ctrl, o, 0, g_bias, chunk, nchunks);
                    hmx_mpy_chunks(ap, act_ctrl, w + half, wgt_ctrl, o + BLK, 0, g_bias, chunk, nchunks);
                }
            *t_hmx += HAP_perf_get_time_us() - t0;
            if (sw && wi == 0) { wi = nwi; g = ng; continue; }                /* the gate group waits in its buffer for its up group */
            if (sw)
            {   /* the pair's post (silu or situ of the gate times up) on its own slot's token: the next pair's HMX goes on meanwhile */
                st_init(&tok2[slot], POSTW); posting2[slot] = 1;
                for (int p = 0; p < POSTW; p++)
                {
                    pj2[slot][p] = (post_job_t){ v_out, out, mode, gelu, M, N, m0, g, tiles * p / POSTW, tiles * (p + 1) / POSTW, &tok2[slot], v_outs[2 * slot], 0 }; pj2[slot][p].gain = WS_GAIN(ws[1]); pj2[slot][p].gaing = WS_GAIN(ws[0]);
                    if (gelu == 4) { pj2[slot][p].bias = ws[1].bias; pj2[slot][p].biasg = ws[0].bias; }   /* SDXL's FF has biases on both halves */
                    worker_pool_job_t job = { post_worker, &pj2[slot][p] }; worker_pool_submit(g_pool, job);
                }
                wi = nwi; g = ng; continue;
            }
            if (pr)
            {   /* the head's first group waits in its buffer; after the second both go to the post with the head's norm and RoPE */
                if (!(g & 1)) { wi = nwi; g = ng; continue; }
                st_init(&tok2[ps_], POSTW); posting2[ps_] = 1;
                for (int p = 0; p < POSTW; p++)
                {
                    pj2[ps_][p] = (post_job_t){ v_out, out, mode, gelu, M, N, m0, g, tiles * p / POSTW, tiles * (p + 1) / POSTW, &tok2[ps_], 0, ws[wi].bias };
                    pj2[ps_][p].gain = WS_GAIN(ws[wi]); pj2[ps_][p].srcA = v_outs[2 * ps_]; pj2[ps_][p].rrw = ws[wi].rrw; pj2[ps_][p].rrope = ws[wi].rrope;
                    worker_pool_job_t job = { post_worker, &pj2[ps_][p] }; worker_pool_submit(g_pool, job);
                }
                wi = nwi; g = ng; continue;
            }
            /* the group before has left its buffer (the one the next group's HMX fills) once its post jobs are done */
            if (posting) { uint64_t p0_ = HAP_perf_get_time_us(); st_wait(&ptok); g_tpost += HAP_perf_get_time_us() - p0_; }
            st_init(&ptok, POSTW); posting = 1;
            for (int p = 0; p < POSTW; p++)
            {
                pj[p] = (post_job_t){ v_out, out, mode, gelu, M, N, m0, g, tiles * p / POSTW, tiles * (p + 1) / POSTW, &ptok, 0, ws[wi].bias };   /* a LoRA's share is in the output already (hmx_mpy_chunks_lora) */
                pj[p].gain = WS_GAIN(ws[wi]);
                worker_pool_job_t job = { post_worker, &pj[p] }; worker_pool_submit(g_pool, job);
            }
            wi = nwi; g = ng;
        }
        /* the next batch refills v_act and these buffers: its pairs' posts must be done first */
        for (int s2 = 0; s2 < 2; s2++) if (posting2[s2]) { uint64_t p0_ = HAP_perf_get_time_us(); st_wait(&tok2[s2]); posting2[s2] = 0; g_tpost += HAP_perf_get_time_us() - p0_; }
    }
    if (posting) { uint64_t p0_ = HAP_perf_get_time_us(); st_wait(&ptok); g_tpost += HAP_perf_get_time_us() - p0_; }
}

/* ------------------------------------------------------------------ elementwise and per-row work in the layout */
typedef void (*tile_fn)(void *arg, uint32_t t0, uint32_t t1);
typedef struct { tile_fn fn; void *arg; uint32_t t0, t1; spintok_t *tok; } tile_job_t;
static void tile_worker(void *a) { tile_job_t *j = (tile_job_t *)a; j->fn(j->arg, j->t0, j->t1); st_done(j->tok); }
/* fn over row tiles a..b-1, in g_nw parts on the worker threads */
static void tiles_parallel_range(tile_fn fn, void *arg, uint32_t a, uint32_t b)
{
    const int nw = g_nw; tile_job_t jobs[MAXW]; spintok_t tok; st_init(&tok, nw); uint32_t n = b - a;
    for (int t = 0; t < nw; t++) { jobs[t] = (tile_job_t){ fn, arg, a + n * t / nw, a + n * (t + 1) / nw, &tok }; worker_pool_job_t job = { tile_worker, &jobs[t] }; worker_pool_submit(g_pool, job); }
    st_wait(&tok);
}
static void tiles_parallel(tile_fn fn, void *arg, uint32_t n) { tiles_parallel_range(fn, arg, 0, n); }
/* the full sum of a vector's 32 sf lanes, in every lane (rotations by 64 .. 4 bytes) */
static inline HVX_Vector lanes_total(HVX_Vector v)                      /* v: sf */
{
    for (int sh = 64; sh >= 4; sh >>= 1) v = sfv(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, sh)));
    return v;
}
/* 1 / sqrt(x) for sf lanes > 0: the bit estimate and one Newton step (relative error ~5e-6, far below fp16) */
static inline HVX_Vector rsqrt_sf(HVX_Vector x)
{
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5f3759df), Q6_Vw_vasr_VwR(x, 1));
    HVX_Vector hx = sfv(Q6_Vqf32_vmpy_VsfVsf(x, splat_sf(0.5f)));
    /* two Newton steps: one left the bit trick's estimate up to 0.17 % off, a scale error on every row of every norm (whole
       blocks erred four times what fp16 alone gives); two leave ~5e-6. Once per row, so it costs nothing that shows */
    for (int i = 0; i < 2; i++)
    {
        HVX_Vector t = sfv(Q6_Vqf32_vmpy_VsfVsf(sfv(Q6_Vqf32_vmpy_VsfVsf(hx, y)), y));
        y = sfv(Q6_Vqf32_vmpy_VsfVsf(y, sfv(Q6_Vqf32_vsub_VsfVsf(splat_sf(1.5f), t))));
    }
    return y;
}
/* sf lanes, even rows e and odd rows o, back to one hf row vector */
static inline HVX_Vector rows_hf(HVX_Vector e, HVX_Vector o)
{ return sf2_to_hf(e, o); }
/* per row pair k of a tile (nblk blocks from tile0, n columns): the row vectors of the mean (if mean) and of 1 / sqrt(var + eps);
 * without mean, the mean square instead of the variance (RMS norm). All in vectors: no scalar fp16 or sqrt. */
static void row_stats(const uint16_t *tile0, uint32_t nblk, float n, float eps, HVX_Vector *mean, HVX_Vector *rstd)
{
    const HVX_Vector one = splat_hf(1.f), inv = splat_sf(1.f / n), ev = splat_sf(eps);
    for (int k = 0; k < 16; k++)
    {
        HVX_Vector se = Q6_V_vzero(), so = Q6_V_vzero(), qe = Q6_V_vzero(), qo = Q6_V_vzero();
        const HVX_Vector *vb = (const HVX_Vector *)tile0 + k;          /* block b's vector k: vb[b * 16] (a block is 16 vectors) */
#if __HVX_ARCH__ < 79
        /* v73 / v75: the qf32 sums carried from block to block, converted once at the end (bit-identical to converting each time on
           v73, the Anima block -3%) */
        for (uint32_t b = 0; b < nblk; b++)
        {
            const HVX_Vector v = vb[b * 16]; const HVX_VectorPair w2 = Q6_Wqf32_vmpy_VhfVhf(v, v);
            qe = b ? Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(w2), qe) : Q6_V_lo_W(w2); qo = b ? Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(w2), qo) : Q6_V_hi_W(w2);
            if (mean) { const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(v, one); se = b ? Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(w), se) : Q6_V_lo_W(w); so = b ? Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(w), so) : Q6_V_hi_W(w); }
        }
        qe = sfv(qe); qo = sfv(qo); if (mean) { se = sfv(se); so = sfv(so); }
#else
        /* v79 / v81: a qf32 value can't be carried in a register (its extended state), so four blocks' products are
           added in qf32 inside one expression and the sum kept as sf: one conversion per four blocks instead of one per block */
        uint32_t b = 0;
#define RS_Q(x_, h_) (h_ ? Q6_V_hi_W(x_) : Q6_V_lo_W(x_))
#define RS_4(acc_, w0_, w1_, w2_, w3_, h_) acc_ = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(RS_Q(w0_, h_), RS_Q(w1_, h_)), \
                                                  Q6_Vqf32_vadd_Vqf32Vqf32(RS_Q(w2_, h_), RS_Q(w3_, h_))), acc_))
        for (; b + 4 <= nblk; b += 4)
        {
            const HVX_Vector v0 = vb[b * 16], v1 = vb[(b + 1) * 16], v2 = vb[(b + 2) * 16], v3 = vb[(b + 3) * 16];
            const HVX_VectorPair a0 = Q6_Wqf32_vmpy_VhfVhf(v0, v0), a1 = Q6_Wqf32_vmpy_VhfVhf(v1, v1), a2 = Q6_Wqf32_vmpy_VhfVhf(v2, v2), a3 = Q6_Wqf32_vmpy_VhfVhf(v3, v3);
            RS_4(qe, a0, a1, a2, a3, 0); RS_4(qo, a0, a1, a2, a3, 1);
            if (mean)
            {
                const HVX_VectorPair m0 = Q6_Wqf32_vmpy_VhfVhf(v0, one), m1 = Q6_Wqf32_vmpy_VhfVhf(v1, one), m2 = Q6_Wqf32_vmpy_VhfVhf(v2, one), m3 = Q6_Wqf32_vmpy_VhfVhf(v3, one);
                RS_4(se, m0, m1, m2, m3, 0); RS_4(so, m0, m1, m2, m3, 1);
            }
        }
#undef RS_4
#undef RS_Q
        for (; b < nblk; b++)
        {   /* the last one to three blocks (a VAE's 96 or 192 channels) one at a time, as before */
            const HVX_Vector v = vb[b * 16]; const HVX_VectorPair w2 = Q6_Wqf32_vmpy_VhfVhf(v, v);
            qe = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(w2), qe)); qo = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(w2), qo));
            if (mean) { const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(v, one); se = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(w), se)); so = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(w), so)); }
        }
#endif
        HVX_Vector ve = sfv(Q6_Vqf32_vmpy_VsfVsf(lanes_total(qe), inv)), vo = sfv(Q6_Vqf32_vmpy_VsfVsf(lanes_total(qo), inv));
        if (mean)
        {
            HVX_Vector me = sfv(Q6_Vqf32_vmpy_VsfVsf(lanes_total(se), inv)), mo = sfv(Q6_Vqf32_vmpy_VsfVsf(lanes_total(so), inv));
            ve = sfv(Q6_Vqf32_vsub_VsfVsf(ve, sfv(Q6_Vqf32_vmpy_VsfVsf(me, me)))); vo = sfv(Q6_Vqf32_vsub_VsfVsf(vo, sfv(Q6_Vqf32_vmpy_VsfVsf(mo, mo))));
            mean[k] = rows_hf(me, mo);
        }
        rstd[k] = rows_hf(rsqrt_sf(sfv(Q6_Vqf32_vadd_VsfVsf(ve, ev))), rsqrt_sf(sfv(Q6_Vqf32_vadd_VsfVsf(vo, ev))));
    }
}

/* h = LayerNorm(x) * (1 + scale) + shift, all M rows; shf / scl: per column block, the column vectors */
/* shf0 / scl0: the vectors of the first z rows instead (a leading frame with its own modulation); z is even, and a vector of a
 * block holds rows 2k and 2k + 1, so a vector is wholly on one side */
typedef struct { const uint16_t *x; uint16_t *h; const HVX_Vector *shf, *scl1; int rms; const HVX_Vector *shf0, *scl0; uint32_t z; } ln_arg_t;   /* rms: RMSNorm, no mean */
static void ln_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    ln_arg_t *a = (ln_arg_t *)a_; const uint16_t *x = a->x; uint16_t *h = a->h; const HVX_Vector *shf = a->shf, *scl1 = a->scl1;
    const float eps = 1e-6f / (RS * RS);
    prefetch(blk((uint16_t *)x, D, t0, 0), D * 64);
    for (uint32_t t = t0; t < t1; t++)
    {
        if (t + 1 < t1) prefetch(blk((uint16_t *)x, D, t + 1, 0), D * 64);
        HVX_Vector mean[16], rstd[16];
        row_stats(blk((uint16_t *)x, D, t, 0), D / 32, D, a->rms ? 1e-6f : eps, a->rms ? 0 : mean, rstd);
        if (a->rms) for (int k = 0; k < 16; k++) mean[k] = Q6_V_vzero();
        for (uint32_t cb = 0; cb < D / 32; cb++)
        {
            const HVX_Vector *xi = (const HVX_Vector *)blk((uint16_t *)x, D, t, cb); HVX_Vector *ho = (HVX_Vector *)blk(h, D, t, cb);
            /* the first z rows' own shift / scale (a leading frame) only in the tiles that have such rows: the per-row choice, with
             * scl0 / shf0 null, came out wrong on v81 (Anima's last block blew up there and in res_tiles' same pattern) */
            if (t * 32 >= a->z)
                for (int k = 0; k < 16; k++)
                {
                    HVX_Vector n = hfr(Q6_Vqf16_vmpy_VhfVhf(hfr(Q6_Vqf16_vsub_VhfVhf(xi[k], mean[k])), rstd[k]));
                    ho[k] = hf(Q6_Vqf16_vadd_VhfVhf(hfr(Q6_Vqf16_vmpy_VhfVhf(n, scl1[cb])), shf[cb]));
                }
            else
                for (int k = 0; k < 16; k++)
                {
                    const int z0 = t * 32 + 2 * k < a->z;
                    HVX_Vector n = hfr(Q6_Vqf16_vmpy_VhfVhf(hfr(Q6_Vqf16_vsub_VhfVhf(xi[k], mean[k])), rstd[k]));
                    ho[k] = hf(Q6_Vqf16_vadd_VhfVhf(hfr(Q6_Vqf16_vmpy_VhfVhf(n, z0 ? a->scl0[cb] : scl1[cb])), z0 ? a->shf0[cb] : shf[cb]));
                }
        }
    }
}
/* act_fill_t: the modulated LayerNorm of x straight into the VTCM tiles (h based so that tile m0 lands at v_act) */
static void ln_fill(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles)
{
    ln_arg_t a = *(ln_arg_t *)arg; a.h = (uint16_t *)v_act - (size_t)m0 * (D / 32) * BH;
    tiles_parallel_range(ln_tiles, &a, m0, m0 + tiles);
}
static void product_to(const uint16_t *act, uint32_t M, uint32_t K, const uint16_t *wl, uint32_t N, uint16_t *out, int mode, uint32_t mt, int gelu, uint64_t *t_hmx)
{ wset_t w = { wl, N, out, mode }; product_multi(act, 0, 0, M, K, &w, 1, mt, gelu, t_hmx); }
static void product(const uint16_t *act, uint32_t M, uint32_t K, const uint16_t *wl, uint32_t N, uint16_t *out, uint32_t mt, int gelu, uint64_t *t_hmx)
{ product_to(act, M, K, wl, N, out, OUT_ACT, mt, gelu, t_hmx); }
/* the same with a LoRA pair applied as it runs */
static void product_lo(const uint16_t *act, uint32_t M, uint32_t K, const uint16_t *wl, uint32_t N, uint16_t *out, int mode, uint32_t mt,
                       const uint16_t *la_, const uint16_t *lb_, uint64_t *t_hmx)
{ wset_t w = { wl, N, out, mode, 0, 0, 0, la_, lb_ }; product_multi(act, 0, 0, M, K, &w, 1, mt, 0, t_hmx); }

/* x += gate * y over [M x D] */
typedef struct { uint16_t *x; const uint16_t *y; const HVX_Vector *gate; const HVX_Vector *gate0; uint32_t z; } res_arg_t;   /* gate0: the first z rows' */
static void res_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    res_arg_t *a = (res_arg_t *)a_; uint16_t *x = a->x; const uint16_t *y = a->y; const HVX_Vector *gate = a->gate;
    for (uint32_t t = t0; t < t1; t++)
    {
        if (t + 1 < t1) { prefetch(blk(x, D, t + 1, 0), D * 64); prefetch(blk((uint16_t *)y, D, t + 1, 0), D * 64); }
        for (uint32_t cb = 0; cb < D / 32; cb++)
        {
            HVX_Vector *xo = (HVX_Vector *)blk(x, D, t, cb); const HVX_Vector *yi = (const HVX_Vector *)blk((uint16_t *)y, D, t, cb);
            const HVX_Vector g = gate[cb];
            if (t * 32 >= a->z)                     /* the usual case (no first-rows gate): one gate for the tile, no per-row choice */
                for (int k = 0; k < 16; k++) xo[k] = FIN(hf(Q6_Vqf16_vadd_VhfVhf(xo[k], hf(Q6_Vqf16_vmpy_VhfVhf(yi[k], g)))));
            else
                for (int k = 0; k < 16; k++) xo[k] = FIN(hf(Q6_Vqf16_vadd_VhfVhf(xo[k], hf(Q6_Vqf16_vmpy_VhfVhf(yi[k], t * 32 + 2 * k < a->z ? a->gate0[cb] : g)))));
        }
    }
}
static void residual(uint16_t *x, const uint16_t *y, uint32_t M, const HVX_Vector *gate) { res_arg_t a = { x, y, gate }; tiles_parallel(res_tiles, &a, M / 32); }

/* per head: q = RMSNorm(q) * w (over the head's 128 columns, 4 blocks), then optionally RoPE with the tile's cos | sin blocks
 * (split-half pairs: column d with d + 64, i.e. block j with block j + 2) */
typedef struct { uint16_t *q; const float *nw; const uint16_t *rope; uint32_t nt; int headq; float wmul; int heads; int full; uint16_t *dst; } rr_arg_t;
/* one head of one tile, h0's 4 blocks: x * rs (the rows' 1 / rms) * wv (weights x factor), then RoPE if cs (rms_rope's and the products' post) */
static void rr_apply(uint16_t *h0, const HVX_Vector *wv, const HVX_Vector *rs, const HVX_Vector *cs)
{
            HVX_Vector *bv[4]; for (int j = 0; j < 4; j++) bv[j] = (HVX_Vector *)(h0 + (size_t)j * BH);
            /* four at a time, loaded before any is stored: the compiler can then pack them */
            for (int j = 0; j < 4; j++) for (int k = 0; k < 16; k += 4)
            {
                const HVX_Vector w = wv[j], x0 = bv[j][k], x1 = bv[j][k + 1], x2 = bv[j][k + 2], x3 = bv[j][k + 3];
                const HVX_Vector y0 = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(x0, rs[k])), w)), y1 = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(x1, rs[k + 1])), w)),
                                 y2 = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(x2, rs[k + 2])), w)), y3 = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(x3, rs[k + 3])), w));
                bv[j][k] = y0; bv[j][k + 1] = y1; bv[j][k + 2] = y2; bv[j][k + 3] = y3;
            }
            if (cs)
            {   /* cs: the tile's RoPE blocks 0, 1: cos; 2, 3: sin */
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 16; k += 2)
                {
                    const HVX_Vector a = bv[j][k], b = bv[j + 2][k], c = cs[j * 16 + k], s = cs[(j + 2) * 16 + k], a1 = bv[j][k + 1], b1 = bv[j + 2][k + 1], c1 = cs[j * 16 + k + 1], s1 = cs[(j + 2) * 16 + k + 1];
                    const HVX_Vector y0 = hf(Q6_Vqf16_vsub_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(c, a)), hf(Q6_Vqf16_vmpy_VhfVhf(s, b)))), z0 = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(s, a)), hf(Q6_Vqf16_vmpy_VhfVhf(c, b))));
                    const HVX_Vector y1 = hf(Q6_Vqf16_vsub_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(c1, a1)), hf(Q6_Vqf16_vmpy_VhfVhf(s1, b1)))), z1 = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(s1, a1)), hf(Q6_Vqf16_vmpy_VhfVhf(c1, b1))));
                    bv[j][k] = y0; bv[j + 2][k] = z0; bv[j][k + 1] = y1; bv[j + 2][k + 1] = z1;
                }
            }
}
/* the 4 column vectors rr_apply takes: norm weights nw [128] times wmul */
static void rr_vecs(HVX_Vector *wv, const float *nw, float wmul)
{ for (int j = 0; j < 4; j++) { float w32[32]; for (int c = 0; c < 32; c++) w32[c] = nw[32 * j + c] * wmul; wv[j] = colvec(w32); } }
static void rr_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    rr_arg_t *ar = (rr_arg_t *)a_; uint16_t *q = ar->q; const float *nw = ar->nw; const uint16_t *rope = ar->rope;
    HVX_Vector wv[4];
    for (int j = 0; j < 4; j++) { float w32[32]; for (int c = 0; c < 32; c++) w32[c] = nw[32 * j + c] * ar->wmul; wv[j] = colvec(w32); }
    HVX_Vector rsf[16];
    for (uint32_t t = t0; t < t1; t++)
        for (int hd = 0; hd < ar->heads; hd++)
        {
            if (ar->full)
            {   /* full: one norm over the whole row (the activation layout's D columns), the weight per column of D */
                if (hd == 0) row_stats(blk(q, D, t, 0), D / 32, (float)D, 1e-6f, 0, rsf);
                for (int j = 0; j < 4; j++) { float w32[32]; for (int c = 0; c < 32; c++) w32[c] = nw[DH * hd + 32 * j + c] * ar->wmul; wv[j] = colvec(w32); }
            }
            uint16_t *h0 = ar->headq ? q + ((size_t)hd * ar->nt + t) * 4 * BH : blk(q, D, t, 4 * hd);
            if (ar->headq) { if (hd + 1 < ar->heads) prefetch(h0 + (size_t)ar->nt * 4 * BH, 8192); else if (t + 1 < t1) prefetch(q + (t + 1) * 4 * BH, 8192); }
            else if (hd == 0 && t + 1 < t1) prefetch(blk(q, D, t + 1, 0), D * 64);
            if (hd == 0 && rope && t + 1 < t1) prefetch(blk((uint16_t *)rope, 128, t + 1, 0), 8192);
            HVX_Vector rs[16]; if (ar->full) memcpy(rs, rsf, sizeof(rs)); else row_stats(h0, 4, DH, 1e-6f, 0, rs);
            rr_apply(h0, wv, rs, rope ? (const HVX_Vector *)blk((uint16_t *)rope, 128, t, 0) : 0);
            if (ar->dst) hvx_copy((uint8_t *)(ar->dst + ((size_t)hd * ar->nt + t) * 4 * BH), (const uint8_t *)h0, 4 * BLK);   /* into the head layout */
        }
}
/* wmul: a factor folded into the norm weights (for q: the softmax's 1 / sqrt(128) * log2 e, so the scores need no scaling) */
static void rms_rope(uint16_t *q, uint32_t M, const float *nw, const uint16_t *rope, int headq, float wmul, int heads) { rr_arg_t a = { q, nw, rope, M / 32, headq, wmul, heads, 0, 0 }; tiles_parallel(rr_tiles, &a, M / 32); }

/* ------------------------------------------------------------------ attention */
/* 2^t, t = x - mrow, mostly in 16-bit integers: t * 1024 (exponent + 10) to fixed point, n = floor(t), f = t - n in Q15, 2^f - 1 = f (c1 + f (c2 + f c3)),
 * and the fp16 bits ((n + 15) << 10) + mantissa. The clamp at -15 - PSH (tmin) keeps the exponent bits from going negative and zero below
 * 2^-(15 + PSH) of the row max; the polynomial's last product is by the Q10 fraction, so its rounding gives the Q10 mantissa at once.
 * Needs tmin = -15 - PSH (softmax_rows's) */
#ifdef HF_CLAMP
#define EXP2_TOP(t_) Q6_Vhf_vmin_VhfVhf((t_), Q6_V_vzero())   /* scores are not clamped in place on v79 / v81: an inf one's x - m */
#else
#define EXP2_TOP(t_) (t_)
#endif
/* x - mrow <= 0 always (mrow is the row's maximum of the clamped scores) and tmin bounds it below: the plain qf16 -> hf conversion, not
   hf()'s, whose +-65504 clamp on v79 / v81 is two more HVX ops a score for nothing (an x - mrow past -65504 is -inf, which vmax takes to
   tmin as it took -65504: the same exponentials; the v79 / v81 blocks are HVX-bound) */
#define EXP2B(x_) ({ HVX_Vector t_ = Q6_Vhf_vmax_VhfVhf(EXP2_TOP(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf((x_), mrow))), tmin); \
    HVX_Vector i_ = Q6_Vh_equals_Vhf(Q6_Vh_vadd_VhVh(t_, k2800)); HVX_Vector f10_ = Q6_V_vand_VV(i_, m3ff), f_ = Q6_Vh_vasl_VhR(f10_, 5); \
    HVX_Vector p_ = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(f_, c3), c2); p_ = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(p_, f_), c1); \
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_vand_VV(i_, mfc00), Q6_Vh_vmpy_VhVh_s1_rnd_sat(p_, f10_)), k3c00); })
/* 1 / x for sf lanes > 0: the bit estimate and three Newton steps (two left ~2e-4, a scale on every softmax row) */
static inline HVX_Vector recip_sf(HVX_Vector x)
{
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), x), two = splat_sf(2.f);
    for (int i = 0; i < 3; i++) y = sfv(Q6_Vqf32_vmpy_VsfVsf(y, sfv(Q6_Vqf32_vsub_VsfVsf(two, sfv(Q6_Vqf32_vmpy_VsfVsf(x, y))))));
    return y;
}
/* g_hmxsum: the softmax rows' sums made by the HMX with P V (a column of ones beside V, accumulated in fp32), not by the HVX workers:
 * the softmax is the attention's slow side on v73 (its workers' wait was a fifth of an Anima block), and the sum was a third of its work */
static int g_hmxsum = 1;
/* causal attention (YuE2's AR prefill, yrun bit 24): g_causal = the keys before the queries (P, a multiple of 32), or -1. Query row r of
 * the call sees keys 0..P + r: past its tile's diagonal block the keys are masked as a range, inside it by g_tri -- vector k of a
 * 32 x 32 score block holds rows 2k (even lanes) and 2k + 1 (odd lanes), key column l >> 1, masked where the column passes the row */
static int g_causal = -1;
static HVX_Vector g_tri[16];
static void tri_init(void)
{
    for (int k = 0; k < 16; k++)
    {
        uint16_t m[64] __attribute__((aligned(128)));
        for (int l = 0; l < 64; l++) m[l] = (l >> 1) > 2 * k + (l & 1) ? 0xffff : 0;
        g_tri[k] = *(HVX_Vector *)m;
    }
}
/* g_smax: each row pair's maximum made beforehand, over all the keys (attention_big: this call's scores are only some of them) */
static const HVX_Vector *g_smax;
/* a key mask that repeats every row of an image: keys are its pixels row by row, g_cmwt tiles of 32 a row, the first g_cmvw of each
 * row real and the rest padding (SDXL at a width that is not a multiple of 1024: the activations' rows padded, kxl.inc); 0: none */
static uint32_t g_cmwt, g_cmvw;
/* softmax numerator of row pairs k0..k1-1 of a tile's scores (nb blocks): fp16 exp2 scaled by 2^PSH, the exponentials eight blocks at a time;
 * rsum[k]: the row vector of 1 / row sum for row pair k (the scale taken out), when the HVX makes the sums.
 * Up to two key ranges [mk[0], mk[1]) and [mk[2], mk[3]) get probability 0 (padding: Anima's past the picture's tokens, Krea's text padding) */
static void softmax_rows(__fp16 *s, uint32_t nb, const uint32_t *mk, HVX_Vector *rsum, int k0, int k1, int diag)
{
    HVX_Vector *v = (HVX_Vector *)s;
    const HVX_Vector tmin = splat_hf(-15.f - PSH), one = splat_hf(1.f), lowest = Q6_Vh_vsplat_R(0xfbff);
    /* up to two masked ranges [mk[0], mk[1]) and [mk[2], mk[3]): per range its blocks and the lanes of its first and last block */
    uint32_t B0[2], B1[2]; HVX_VectorPred PM[2][2];
    for (int g = 0; g < 2; g++)
    {
        const uint32_t ka = mk[2 * g], kb = mk[2 * g + 1];
        const uint32_t b0 = ka / 32, b1 = kb > ka ? (kb + 31) / 32 : b0;              /* blocks with masked keys */
        HVX_VectorPred pm[2] = { Q6_Q_vsetq_R(0), Q6_Q_vsetq_R(0) };                     /* lanes of the first and the last of them */
        for (int e = 0; e < 2 && b1 > b0; e++)
        {
            const uint32_t b = e ? b1 - 1 : b0; uint16_t lm[64] __attribute__((aligned(128)));
            for (int l = 0; l < 64; l++) { uint32_t key = 32 * b + (l >> 1); lm[l] = key >= ka && key < kb ? 0xffff : 0; }
            pm[e] = Q6_Q_vand_VR(*(HVX_Vector *)lm, 0x01010101);
        }
        B0[g] = b0; B1[g] = b1; PM[g][0] = pm[0]; PM[g][1] = pm[1];
    }
    const HVX_Vector k2800 = Q6_Vh_vsplat_R(0x2800), m3ff = Q6_Vh_vsplat_R(0x3ff), mfc00 = Q6_Vh_vsplat_R(0xfc00), k3c00 = Q6_Vh_vsplat_R(0x3c00 + (PSH << 10)),
                     c1 = Q6_Vh_vsplat_R(22788), c2 = Q6_Vh_vsplat_R(7437), c3 = Q6_Vh_vsplat_R(2535);
    for (int k = k0; k < k1; k++)
    {
        for (int g = 0; g < 2; g++)
            for (uint32_t b = B0[g]; b < B1[g]; b++)
            {
                const int full = 32 * b >= mk[2 * g] && 32 * b + 32 <= mk[2 * g + 1];
                v[b * 16 + k] = full ? lowest : Q6_V_vmux_QVV(b == B0[g] ? PM[g][0] : PM[g][1], lowest, v[b * 16 + k]);
            }
        if (diag >= 0 && (uint32_t)diag < nb) v[diag * 16 + k] = Q6_V_vmux_QVV(Q6_Q_vand_VR(g_tri[k], 0x01010101), lowest, v[diag * 16 + k]);
        if (g_cmwt)
        {   /* the padding columns of every row: the row's tile at g_cmvw / 32 from lane g_cmvw % 32 on, the tiles after it whole */
            const uint32_t last = g_cmvw / 32, r = g_cmvw % 32; HVX_VectorPred pc = Q6_Q_vsetq_R(0);
            if (r) { uint16_t lm[64] __attribute__((aligned(128))); for (int l = 0; l < 64; l++) lm[l] = (uint32_t)(l >> 1) >= r ? 0xffff : 0; pc = Q6_Q_vand_VR(*(HVX_Vector *)lm, 0x01010101); }
            for (uint32_t b0 = 0; b0 < nb; b0 += g_cmwt)
                for (uint32_t t = last; t < g_cmwt && b0 + t < nb; t++)
                    v[(b0 + t) * 16 + k] = t == last && r ? Q6_V_vmux_QVV(pc, lowest, v[(b0 + t) * 16 + k]) : lowest;
        }
#ifdef HF_CLAMP
        /* v79 / v81: a score past fp16's range comes out of the HMX as inf (v73 saturates), and inf - inf made Krea's attention rows NaN
           on v81 (its pre-scaled q.k reaches ~3e4 and past): clamp as v73 does */
        /* the maximum over the raw scores, clamped once: the clamped scores are no longer written back (a store pass over the whole tile,
           the v79 / v81 softmax being bound by VTCM traffic); an inf score is caught in EXP2B by its upper bound 0 */
        HVX_Vector m = v[k];
        for (uint32_t b = 1; b < nb; b++) m = Q6_Vhf_vmax_VhfVhf(m, v[b * 16 + k]);
        m = clamp_hf(m);
#else
        HVX_Vector m = v[k];
        for (uint32_t b = 1; b < nb; b++) m = Q6_Vhf_vmax_VhfVhf(m, v[b * 16 + k]);
#endif
        /* the even and odd lanes' maxima by rotating 4 bytes at least (the pair stays together), then word 0 splat: no scalar fp16 */
        for (int sh = 64; sh >= 4; sh >>= 1) m = Q6_Vhf_vmax_VhfVhf(m, Q6_V_vror_VR(m, sh));
        const HVX_Vector mrow = g_smax ? g_smax[k] : Q6_V_vsplat_R(Q6_R_vextract_VR(m, 0));
        HVX_Vector se = Q6_V_vzero(), so = Q6_V_vzero();      /* sf, per lane (even / odd): the row sums in single precision (a qf16 sum
                                                                 dropped low bits on every add: over 5120 keys the sums came out small) */
        /* four blocks loaded before any is stored: the stores to v cannot then hold back the next loads */
        /* the last blocks one at a time: four at a time past the end wrote into the next buffer (126 blocks: all noise) */
        const uint32_t nb4 = nb & ~3u;
        if (g_hmxsum) for (uint32_t b = nb4; b < nb; b++) v[b * 16 + k] = EXP2B(v[b * 16 + k]);
        else for (uint32_t b = nb4; b < nb; b++)
        {
            HVX_Vector e0 = EXP2B(v[b * 16 + k]); v[b * 16 + k] = e0; HVX_VectorPair w0 = Q6_Wqf32_vmpy_VhfVhf(e0, one);
            se = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(w0), se)); so = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(w0), so));
        }
        if (g_hmxsum)
        {   /* eight blocks at a time: more independent work for the packets */
            const uint32_t nb8 = nb & ~7u;
            for (uint32_t b = 0; b < nb8; b += 8)
            {
                HVX_Vector x0 = v[b * 16 + k], x1 = v[(b + 1) * 16 + k], x2 = v[(b + 2) * 16 + k], x3 = v[(b + 3) * 16 + k];
                HVX_Vector x4 = v[(b + 4) * 16 + k], x5 = v[(b + 5) * 16 + k], x6 = v[(b + 6) * 16 + k], x7 = v[(b + 7) * 16 + k];
                HVX_Vector e0 = EXP2B(x0), e1 = EXP2B(x1), e2 = EXP2B(x2), e3 = EXP2B(x3), e4 = EXP2B(x4), e5 = EXP2B(x5), e6 = EXP2B(x6), e7 = EXP2B(x7);
                v[b * 16 + k] = e0; v[(b + 1) * 16 + k] = e1; v[(b + 2) * 16 + k] = e2; v[(b + 3) * 16 + k] = e3;
                v[(b + 4) * 16 + k] = e4; v[(b + 5) * 16 + k] = e5; v[(b + 6) * 16 + k] = e6; v[(b + 7) * 16 + k] = e7;
            }
            for (uint32_t b = nb8; b < nb4; b++) v[b * 16 + k] = EXP2B(v[b * 16 + k]);
            continue;
        }
        for (uint32_t b = 0; b < nb4; b += 4)
        {
            HVX_Vector x0 = v[b * 16 + k], x1 = v[(b + 1) * 16 + k], x2 = v[(b + 2) * 16 + k], x3 = v[(b + 3) * 16 + k];
            HVX_Vector e0 = EXP2B(x0), e1 = EXP2B(x1), e2 = EXP2B(x2), e3 = EXP2B(x3);
            v[b * 16 + k] = e0; v[(b + 1) * 16 + k] = e1; v[(b + 2) * 16 + k] = e2; v[(b + 3) * 16 + k] = e3;
            HVX_VectorPair w0 = Q6_Wqf32_vmpy_VhfVhf(e0, one), w1 = Q6_Wqf32_vmpy_VhfVhf(e1, one), w2 = Q6_Wqf32_vmpy_VhfVhf(e2, one), w3 = Q6_Wqf32_vmpy_VhfVhf(e3, one);
            se = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(w0), Q6_V_lo_W(w1)), Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(w2), Q6_V_lo_W(w3))), se));
            so = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(w0), Q6_V_hi_W(w1)), Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(w2), Q6_V_hi_W(w3))), so));
        }
        if (g_hmxsum) continue;                                         /* the sums come out of P V's HMX (see attention) */
        for (int sh = 64; sh >= 4; sh >>= 1) { se = sfv(Q6_Vqf32_vadd_VsfVsf(se, Q6_V_vror_VR(se, sh))); so = sfv(Q6_Vqf32_vadd_VsfVsf(so, Q6_V_vror_VR(so, sh))); }
        const HVX_Vector unsh = splat_sf(1.f / (1 << PSH));
        rsum[k] = rows_hf(recip_sf(sfv(Q6_Vqf32_vmpy_VsfVsf(se, unsh))), recip_sf(sfv(Q6_Vqf32_vmpy_VsfVsf(so, unsh))));
    }
}
/* a tile's 16 row pairs handed out one at a time (atomic counter) to the workers and, once its HMX work is done, the main thread */
typedef struct { __fp16 *s; uint32_t nb, mk[4]; HVX_Vector *rsum; int next; spintok_t tok; uint64_t t_sub; volatile uint32_t first, last; int done; int diag; } sm_async_t;
static void sm_take(sm_async_t *a)
{
    for (;;) { int k = __atomic_fetch_add(&a->next, 1, __ATOMIC_RELAXED); if (k >= 16) break; softmax_rows(a->s, a->nb, a->mk, a->rsum, k, k + 1, a->diag); }
}
static void sm_worker(void *p)
{
    sm_async_t *a = (sm_async_t *)p; uint32_t st = (uint32_t)(HAP_perf_get_time_us() - a->t_sub);
    if (!a->first || st < a->first) a->first = st ? st : 1;
    sm_take(a); uint32_t en = (uint32_t)(HAP_perf_get_time_us() - a->t_sub); if (en > a->last) a->last = en;
    __atomic_fetch_add(&a->done, 1, __ATOMIC_RELEASE);
}
static void softmax_start(sm_async_t *a, __fp16 *s, uint32_t nb, const uint32_t *mk, HVX_Vector *rsum, int diag)
{
    a->s = s; a->nb = nb; memcpy(a->mk, mk, sizeof(a->mk)); a->rsum = rsum; a->diag = diag; a->next = 0; a->first = a->last = 0; a->done = 0; a->t_sub = HAP_perf_get_time_us();
    /* a short tile (cross attention's few keys): the main thread's MAIN_TAKE does all 16 row pairs, nobody waited for */
    if ((g_opt & OPT_SMSOLO) && nb <= (uint32_t)g_smsolo) { a->done = SMJ; return; }
    for (int t = 0; t < SMJ; t++) { worker_pool_job_t job = { sm_worker, a }; worker_pool_submit(g_pool, job); }
}

/* the reciprocals of four row pairs' sums (vs[k..k+3], each one pair in all its 32 words) into rr[k..k+3], as rows_hf(recip_sf(..)) of
 * each would make them: the four packed a quarter each into one vector, widened and inverted once, each quarter spread back */
static inline void recip_rows4(const HVX_Vector *vs, HVX_Vector *rr, int k, HVX_Vector one_h)
{
    const HVX_VectorPred q32 = Q6_Q_vsetq_R(32), q64 = Q6_Q_vsetq_R(64), q96 = Q6_Q_vsetq_R(96);
    const HVX_Vector x = Q6_V_vmux_QVV(q32, vs[k], Q6_V_vmux_QVV(q64, vs[k + 1], Q6_V_vmux_QVV(q96, vs[k + 2], vs[k + 3])));
    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(x, one_h);
    const HVX_Vector y = rows_hf(recip_sf(sfv(Q6_V_lo_W(w))), recip_sf(sfv(Q6_V_hi_W(w))));
    for (int j = 0; j < 4; j++)
    {
        const HVX_Vector z = Q6_V_vror_VR(y, 32 * j);                    /* quarter j to words 0..7 */
        const HVX_Vector s1 = Q6_V_vmux_QVV(q32, z, Q6_V_vror_VR(z, 96)); /* words 8..15 = 0..7 */
        rr[k + j] = Q6_V_vmux_QVV(q64, s1, Q6_V_vror_VR(s1, 64));         /* words 16..31 = 0..15 */
    }
}
static void av_finish_part(uint8_t *vo, const uint8_t *vsum, HVX_Vector *rr, uint8_t *dst, int hmxsum, int k0, int k1)
{   /* row pairs k0..k1-1 of the four blocks: the same arithmetic as av_finish, the copy out by vector */
    const HVX_Vector one_h = splat_hf(1.f);
    if (hmxsum) for (int kk = k0; kk < k1; kk++) { HVX_Vector s_ = ((const HVX_Vector *)vsum)[kk];
        HVX_VectorPair w_ = Q6_Wqf32_vmpy_VhfVhf(s_, one_h); rr[kk] = rows_hf(recip_sf(sfv(Q6_V_lo_W(w_))), recip_sf(sfv(Q6_V_hi_W(w_)))); }
    const HVX_Vector *ov_ = (const HVX_Vector *)vo; HVX_Vector *d_ = (HVX_Vector *)dst;
    for (int kk = k0; kk < k1; kk++) { const HVX_Vector r_ = rr[kk];
        d_[kk] = hf(Q6_Vqf16_vmpy_VhfVhf(ov_[kk], r_)); d_[16 + kk] = hf(Q6_Vqf16_vmpy_VhfVhf(ov_[16 + kk], r_));
        d_[32 + kk] = hf(Q6_Vqf16_vmpy_VhfVhf(ov_[32 + kk], r_)); d_[48 + kk] = hf(Q6_Vqf16_vmpy_VhfVhf(ov_[48 + kk], r_)); }
}
/* a P V tile's finish (the AV macro's second half): rows' 1 / sum from the ones product (hmxsum) or as the softmax left them (rr), the
 * four output blocks divided, then copied out to dst */
static void av_finish(uint8_t *vo, const uint8_t *vsum, HVX_Vector *rr, uint8_t *dst, int hmxsum)
{
    const HVX_Vector one_h = splat_hf(1.f);
    if (hmxsum) for (int kk = 0; kk < 16; kk += 4) recip_rows4((const HVX_Vector *)vsum, rr, kk, one_h);
    HVX_Vector *ov_ = (HVX_Vector *)vo;
    for (int kk = 0; kk < 16; kk++) { HVX_Vector r_ = rr[kk];
        const HVX_Vector o0_ = ov_[kk], o1_ = ov_[16 + kk], o2_ = ov_[32 + kk], o3_ = ov_[48 + kk];
        const HVX_Vector p0_ = hf(Q6_Vqf16_vmpy_VhfVhf(o0_, r_)), p1_ = hf(Q6_Vqf16_vmpy_VhfVhf(o1_, r_)), p2_ = hf(Q6_Vqf16_vmpy_VhfVhf(o2_, r_)), p3_ = hf(Q6_Vqf16_vmpy_VhfVhf(o3_, r_));
        ov_[kk] = p0_; ov_[16 + kk] = p1_; ov_[32 + kk] = p2_; ov_[48 + kk] = p3_; }
    hvx_copy(dst, vo, 4 * BLK);
}
/* the DiT attention's crew -- SMJ - 1 workers spinning on gen for the next tile's rows, the main thread taking rows too once
 * its HMX work is done (as kxl.inc's xsm_t); started after a head's preparation (which needs the pool) and let go at the head's end */
typedef struct { int gen, quit, next, done; __fp16 *s; uint32_t nb, mk[4]; HVX_Vector *rsum; int diag; spintok_t tok;
                 uint8_t *fo, *fsum, *fdst; HVX_Vector *frr; int fhs; } dsm_t;   /* fo..: a P V finish taken as extra work (OPT_AVCREW) */
static void dsm_rows(dsm_t *g)
{
    const int n = g->fo ? 20 : 16;   /* jobs 16..19: the finish's row pairs 4 at a time */
    for (;;) { const int k = __atomic_fetch_add(&g->next, 1, __ATOMIC_RELAXED); if (k >= n) break;
               if (k >= 16) av_finish_part(g->fo, g->fsum, g->frr, g->fdst, g->fhs, 4 * (k - 16), 4 * (k - 15)); else softmax_rows(g->s, g->nb, g->mk, g->rsum, k, k + 1, g->diag); }
}
static void dsm_worker(void *p)
{
    dsm_t *g = (dsm_t *)p; int seen = 0;
    for (;;)
    {
        int gen; while ((gen = __atomic_load_n(&g->gen, __ATOMIC_ACQUIRE)) == seen) { }
        if (__atomic_load_n(&g->quit, __ATOMIC_ACQUIRE)) break;
        seen = gen; dsm_rows(g); __atomic_fetch_add(&g->done, 1, __ATOMIC_RELEASE);
    }
    st_done(&g->tok);
}
static void dsm_start(dsm_t *g, int nwk)
{ memset(g, 0, sizeof(*g)); st_init(&g->tok, nwk); for (int t = 0; t < nwk; t++) { worker_pool_job_t job = { dsm_worker, g }; worker_pool_submit(g_pool, job); } }
static void dsm_go(dsm_t *g, __fp16 *sc, uint32_t nb, const uint32_t *mk, HVX_Vector *rsum, int diag)
{
    g->s = sc; g->nb = nb; memcpy(g->mk, mk, sizeof(g->mk)); g->rsum = rsum; g->diag = diag;   /* fo..: set by the caller before (the workers read them once gen moves) */
    __atomic_store_n(&g->next, 0, __ATOMIC_RELAXED); __atomic_store_n(&g->done, 0, __ATOMIC_RELAXED); __atomic_fetch_add(&g->gen, 1, __ATOMIC_RELEASE);
}
static void dsm_finish(dsm_t *g, int nwk) { dsm_rows(g); while (__atomic_load_n(&g->done, __ATOMIC_ACQUIRE) < nwk) { } }
static void dsm_stop(dsm_t *g) { __atomic_store_n(&g->quit, 1, __ATOMIC_RELEASE); __atomic_fetch_add(&g->gen, 1, __ATOMIC_RELEASE); st_wait(&g->tok); }

/* The 32 x 32 block transpose for K^T, with the HVX scatter into VTCM: vector k of the source block (rows 2k, 2k + 1; lane l is
 * row 2k + (l & 1), column l >> 1) goes out with one vscatter whose halfword offsets put element (r, c) where the transposed block
 * keeps it, 32 (c & ~1) + 2r + (c & 1). 16 scatters a block instead of 1024 moves. */
static HVX_Vector g_toff[16];
static void transpose_init(void)
{
    for (int k = 0; k < 16; k++)
    {
        uint16_t o[64] __attribute__((aligned(128)));
        for (int l = 0; l < 64; l++) { int r = 2 * k + (l & 1), c = l >> 1; o[l] = (uint16_t)(2 * (32 * (c & ~1) + 2 * r + (c & 1))); }
        g_toff[k] = *(HVX_Vector *)o;
    }
}
static inline void transpose_block(uint8_t *dst_vtcm, const uint16_t *src)
{
    const HVX_Vector *sv = (const HVX_Vector *)src;
    for (int k = 0; k < 16; k++) Q6_vscatter_RMVhV((size_t)dst_vtcm, BLK - 1, g_toff[k], sv[k]);
}
/* the scatters have reached VTCM before anything reads it */
static inline void scatter_fence(uint8_t *vtcm_addr)
{
    asm volatile("vmem(%0+#0):scatter_release\n" :: "r"(vtcm_addr) : "memory");
    volatile HVX_Vector t = *(volatile HVX_Vector *)vtcm_addr; (void)t;
}
/* a head's preparation for attention() on the workers -- key tiles t0..t1 transposed (and that share of v's and q's copies),
 * each worker's scatters fenced by itself before its job counts as done */
typedef struct { uint8_t *v_q, *v_v, *v_kt; const uint8_t *qs, *vs; const uint16_t *kh; uint32_t qhead, khead, nk, kt_group, kt_half; int dok; uint32_t hb; } hp_arg_t;   /* hb: blocks a key tile's head row (0: 4, SDXL 2) */
static void hp_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    const hp_arg_t *a = (const hp_arg_t *)a_; const uint32_t n = a->nk;
    const size_t q0 = ((size_t)a->qhead * t0 / n) & ~(size_t)127, q1 = t1 == n ? a->qhead : ((size_t)a->qhead * t1 / n) & ~(size_t)127;
    if (q1 > q0) hvx_copy(a->v_q + q0, a->qs + q0, (uint32_t)(q1 - q0));
    if (!a->dok || t1 <= t0) return;
    const size_t v0 = ((size_t)a->khead * t0 / n) & ~(size_t)127, v1 = t1 == n ? a->khead : ((size_t)a->khead * t1 / n) & ~(size_t)127;
    if (v1 > v0) hvx_copy(a->v_v + v0, a->vs + v0, (uint32_t)(v1 - v0));
    const uint32_t hb = a->hb ? a->hb : 4;
    for (uint32_t t = t0; t < t1; t++)
        for (uint32_t j = 0; j < hb; j++)
        {
            if (j == 0 && ((t - t0) & 7) == 0 && t + 8 < t1) prefetch(a->kh + (size_t)(t + 8) * hb * BH, hb * 16384);
            transpose_block(a->v_kt + (t / 2) * a->kt_group + (t % 2) * a->kt_half + j * BLK, a->kh + ((size_t)t * hb + j) * BH);
        }
    scatter_fence(a->v_kt + (t0 / 2) * a->kt_group + (t0 % 2) * a->kt_half);
}

/* The attention when its keys do not fit VTCM the usual way (YuE2's long songs; past the end the blocks overwrite each other, silently): one query
 * tile and one score row at a time, the keys and values in slabs. With more than one slab a first pass finds every row's maximum over all the
 * keys, then each slab's exp(s - max) V and row sums are added up, so the result is the one pass's, at a second QK's cost. Not causal. */
static void attention_big(const uint16_t *q, uint32_t M, const uint16_t *k, const uint16_t *v, uint32_t L, const uint32_t *mk, int kvrep, uint16_t *o, uint64_t *t_tr, uint64_t *t_sm)
{
    const uint32_t nq = M / 32, nk = L / 32; uint32_t ns = ((VTCM_BYTES - 65536) / BLK - 16) / 11;
    ns &= ~15u; if (ns >= nk) ns = nk;
    uint8_t *v_q = g_vtcm, *v_kt = v_q + 4 * BLK, *v_v = v_kt + (size_t)ns * 4 * BLK, *v_o = v_v + (size_t)ns * 4 * BLK, *v_s = v_o + 4 * BLK, *v_one = v_s + (size_t)(ns + 2) * BLK, *v_sum = v_one + (size_t)ns * BLK;
    const uint32_t qk_act = ((DH >> 5) << 11) | 0x7C, qk_wgt = ((DH >> 5) << 11) - 1, kt_half = 4 * BLK, kt_group = 2 * kt_half;
    HVX_Vector *mx = (HVX_Vector *)memalign(128, (size_t)nq * 16 * 128); float *sum = (float *)malloc((size_t)nq * 32 * 4); if (!mx || !sum) { free(mx); free(sum); return; }
    for (int hd = 0; hd < NH; hd++)
    {
        const uint16_t *qh = q + (size_t)hd * M * DH; const int hk = hd / kvrep; const uint16_t *kh = k + (size_t)hk * L * DH, *vh = v + (size_t)hk * L * DH;
        #define SLAB_K(b0_, n_) do { for (uint32_t t_ = 0; t_ < (n_); t_++) for (int j_ = 0; j_ < 4; j_++) transpose_block(v_kt + (t_ / 2) * kt_group + (t_ % 2) * kt_half + j_ * BLK, kh + ((size_t)((b0_) + t_) * 4 + j_) * BH); scatter_fence(v_kt); } while (0)
        #define SLAB_MK(b0_, n_) { mk[0] > 32 * (b0_) ? mk[0] - 32 * (b0_) : 0, mk[1] > 32 * (b0_) ? (mk[1] - 32 * (b0_) < 32 * (n_) ? mk[1] - 32 * (b0_) : 32 * (n_)) : 0, mk[2] > 32 * (b0_) ? mk[2] - 32 * (b0_) : 0, mk[3] > 32 * (b0_) ? (mk[3] - 32 * (b0_) < 32 * (n_) ? mk[3] - 32 * (b0_) : 32 * (n_)) : 0 }
        memset(sum, 0, (size_t)nq * 32 * 4);
        if (nk > ns)
        {   /* every row pair's maximum over all the slabs */
            const HVX_Vector lowest = Q6_Vh_vsplat_R(0xfbff); for (size_t i = 0; i < (size_t)nq * 16; i++) mx[i] = lowest;
            for (uint32_t b0 = 0; b0 < nk; b0 += ns)
            {
                const uint32_t n = nk - b0 < ns ? nk - b0 : ns; uint32_t mks[4] = SLAB_MK(b0, n); uint64_t a = HAP_perf_get_time_us(); SLAB_K(b0, n); *t_tr += HAP_perf_get_time_us() - a;
                for (uint32_t qt = 0; qt < nq; qt++)
                {
                    hvx_copy(v_q, (const uint8_t *)(qh + (size_t)qt * 4 * BH), 4 * BLK);
                    hmx_mpy_row(v_q, qk_act, v_kt, qk_wgt, v_s, kt_half, g_bias, n);
                    const HVX_Vector *sv = (const HVX_Vector *)v_s;
                    for (int kk = 0; kk < 16; kk++)
                    {
                        HVX_Vector m = lowest;
                        for (uint32_t b = 0; b < n; b++)
                        {
                            HVX_Vector x = sv[b * 16 + kk];
                            for (int g = 0; g < 2; g++)
                            {
                                const uint32_t ka = mks[2 * g], kb = mks[2 * g + 1];
                                if (kb > ka && 32 * b < kb && 32 * b + 32 > ka)
                                {
                                    uint16_t lm[64] __attribute__((aligned(128)));
                                    for (int l = 0; l < 64; l++) { uint32_t key = 32 * b + (l >> 1); lm[l] = key >= ka && key < kb ? 0xffff : 0; }
                                    x = Q6_V_vmux_QVV(Q6_Q_vand_VR(*(HVX_Vector *)lm, 0x01010101), lowest, x);
                                }
                            }
#ifdef HF_CLAMP
                            x = clamp_hf(x);
#endif
                            m = Q6_Vhf_vmax_VhfVhf(m, x);
                        }
                        for (int sh = 64; sh >= 4; sh >>= 1) m = Q6_Vhf_vmax_VhfVhf(m, Q6_V_vror_VR(m, sh));
                        mx[qt * 16 + kk] = Q6_Vhf_vmax_VhfVhf(mx[qt * 16 + kk], Q6_V_vsplat_R(Q6_R_vextract_VR(m, 0)));
                    }
                }
            }
        }
        for (uint32_t b0 = 0; b0 < nk; b0 += ns)
        {
            const uint32_t n = nk - b0 < ns ? nk - b0 : ns; uint32_t mks[4] = SLAB_MK(b0, n); uint64_t a = HAP_perf_get_time_us();
            SLAB_K(b0, n);
            for (uint32_t g = 0; g < 2; g++) for (uint32_t ha = 0; ha < 2; ha++)
                hvx_copy(v_v + (size_t)g * n * 4 * BLK / 2 + (size_t)ha * n * BLK, (const uint8_t *)(vh + (size_t)g * L * 64 + (size_t)ha * nk * BH + (size_t)b0 * BH), n * BLK);
            memset(v_one, 0, (size_t)n * BLK); { __fp16 *o1 = (__fp16 *)v_one; for (uint32_t kk = 0; kk < 32 * n; kk++) o1[(kk >> 5) * 1024 + 32 * ((kk & 31) & ~1) + (kk & 1)] = (__fp16)1.f; }
            *t_tr += HAP_perf_get_time_us() - a;
            uint32_t kcb = n < KC / 32 ? n : KC / 32; while (n % kcb) kcb--;
            const uint32_t kc = kcb * 32, av_act = ((kc >> 5) << 11) | 0x7C, av_wgt = ((kc >> 5) << 11) - 1, av_chunk = (kc / 32) * BLK;
            for (uint32_t qt = 0; qt < nq; qt++)
            {
                hvx_copy(v_q, (const uint8_t *)(qh + (size_t)qt * 4 * BH), 4 * BLK);
                hmx_mpy_row(v_q, qk_act, v_kt, qk_wgt, v_s, kt_half, g_bias, n);
                HVX_Vector rs[16]; sm_async_t sa; g_smax = nk > ns ? mx + (size_t)qt * 16 : 0;
                softmax_start(&sa, (__fp16 *)v_s, n, mks, rs, -1);
                uint64_t c = HAP_perf_get_time_us(); MAIN_TAKE; while (__atomic_load_n(&sa.done, __ATOMIC_ACQUIRE) < SMJ) { } *t_sm += HAP_perf_get_time_us() - c;
                g_smax = 0;
                for (uint32_t g = 0; g < 2; g++)
                {
                    hmx_mpy_chunks(v_s, av_act, v_v + (size_t)g * n * 4 * BLK / 2, av_wgt, v_o + (2 * g) * BLK, 0, g_bias_av, av_chunk, n / kcb);
                    hmx_mpy_chunks(v_s, av_act, v_v + (size_t)g * n * 4 * BLK / 2 + (size_t)n * BLK, av_wgt, v_o + (2 * g + 1) * BLK, 0, g_bias_av, av_chunk, n / kcb);
                }
                hmx_mpy_chunks(v_s, av_act, v_one, av_wgt, v_sum, 0, g_bias_av, av_chunk, n / kcb);
                for (int kk = 0; kk < 16; kk++) { const __fp16 *sf_ = (const __fp16 *)&((HVX_Vector *)v_sum)[kk]; sum[qt * 32 + 2 * kk] += (float)sf_[0]; sum[qt * 32 + 2 * kk + 1] += (float)sf_[1]; }
                uint16_t *od = blk(o, D, qt, 4 * hd);
                if (b0 == 0) hvx_copy((uint8_t *)od, v_o, 4 * BLK);
                else { HVX_Vector *d_ = (HVX_Vector *)od; const HVX_Vector *a_ = (const HVX_Vector *)v_o; for (int kk = 0; kk < 64; kk++) d_[kk] = add_hf(d_[kk], a_[kk]); }
            }
        }
        for (uint32_t qt = 0; qt < nq; qt++)
        {   /* divided by the rows' sums, all the keys' */
            HVX_Vector *d_ = (HVX_Vector *)blk(o, D, qt, 4 * hd);
            for (int kk = 0; kk < 16; kk++)
            {
                const HVX_Vector r_ = rows_hf(splat_sf(1.f / sum[qt * 32 + 2 * kk]), splat_sf(1.f / sum[qt * 32 + 2 * kk + 1]));
                for (int bb = 0; bb < 4; bb++) d_[bb * 16 + kk] = hf(Q6_Vqf16_vmpy_VhfVhf(d_[bb * 16 + kk], r_));
            }
        }
        #undef SLAB_K
        #undef SLAB_MK
    }
    free(mx); free(sum);
}
/* o[:, head] = softmax(q k^T / sqrt(128)) v for every head; q [M x D] queries, k, v [L x D] keys and values, all in the layout.
 * Per head: q's head blocks gathered into VTCM, k transposed block by block into the weight layout of K^T [128 x L], v's head
 * blocks reordered into the weight layout of V [L x 128]; then per 32-query tile: scores (HMX), softmax (4 workers), e v (HMX). */
/* kvrep query heads share a key / value head (GQA; 1 = plain); keys in [mk[0], mk[1]) and [mk[2], mk[3]) masked */
static void attention(const uint16_t *q, uint32_t M, const uint16_t *k, const uint16_t *v, uint32_t L, const uint32_t *mk, int kvrep, uint16_t *o, uint64_t *t_tr, uint64_t *t_sm)
{
    /* what the usual way holds in VTCM: q, K^T, V, the output, three score rows, a spare, the ones, the sums */
    if (g_causal < 0 && ((size_t)4 * (M / 32) + (size_t)14 * (L / 32) + 32) * BLK > VTCM_BYTES - 65536) { attention_big(q, M, k, v, L, mk, kvrep, o, t_tr, t_sm); return; }
    const uint32_t nq = M / 32, nk = L / 32, qhead = nq * 4 * BLK, khead = nk * 4 * BLK, vhead = nk * 4 * BLK;
    uint8_t *v_q = g_vtcm, *v_kt = v_q + qhead, *v_v = v_kt + khead, *v_o = v_v + vhead, *v_s[3], *v_c;
    /* a gap between the score rows: the HMX's output reached a little past its end, into the row the softmax workers were on (about one Anima block in 400 came out a little off; 0 in 2000 with it) */
    v_s[0] = v_o + 4 * BLK; v_s[1] = v_s[0] + (nk + 2) * BLK; v_s[2] = v_s[1] + (nk + 2) * BLK; v_c = v_s[2] + (nk + 2) * BLK;
    /* the ones (key k's weight 1 in all 32 columns of a 32-column half in the weight layout) and the row sums' output block: every
       column of it is the row's sum, so vector kk is already row pair kk's sums lane by lane (no vextract + vsplat; v73
       Anima block -8 ms of P V: the vector-to-scalar move held the pipeline) */
    uint8_t *v_one = v_c + nk * BLK, *v_sum = v_one + nk * BLK; const HVX_Vector one_h = splat_hf(1.f);
    if (g_hmxsum) for (uint32_t b = 0; b < nk * BLK / 128; b++) ((HVX_Vector *)v_one)[b] = one_h;   /* all ones: every key, every column */
    const uint32_t qk_act = ((DH >> 5) << 11) | 0x7C, qk_wgt = ((DH >> 5) << 11) - 1;
    /* the AV chunk: the most blocks (at most KC) that divide L evenly -- L = 4224 (Krea) is not a multiple of 512, and chunks of
     * 512 would leave its last keys out */
    uint32_t kcb = (L < KC ? L : KC) / 32; while ((L / 32) % kcb) kcb--;
    const uint32_t kc = kcb * 32, av_act = ((kc >> 5) << 11) | 0x7C, av_wgt = ((kc >> 5) << 11) - 1, av_chunk = (kc / 32) * BLK;
    const uint32_t kt_half = 4 * BLK, kt_group = 2 * kt_half, v_group = L * 64 * 2, v_half = (L / 32) * BLK;
    for (int hd = 0; hd < NH; hd++)
    {
        uint64_t a = HAP_perf_get_time_us();
        if (g_opt & OPT_HEADPAR)
        {   /* the same copies and transposes on the workers */
            const int hk = hd / kvrep;
            hp_arg_t ha = { v_q, v_v, v_kt, (const uint8_t *)(q + (size_t)hd * M * DH), (const uint8_t *)(v + (size_t)hk * L * DH), k + (size_t)hk * L * DH,
                            qhead, khead, nk, kt_group, kt_half, hd % kvrep == 0 };
            tiles_parallel_range(hp_tiles, &ha, 0, nk);
        }
        else {
        hvx_copy(v_q, (const uint8_t *)(q + (size_t)hd * M * DH), qhead);  /* q is head-major */
        const int hk = hd / kvrep;                                                      /* the key / value head */
        if (hd % kvrep == 0) {
        hvx_copy(v_v, (const uint8_t *)(v + (size_t)hk * L * DH), khead);                 /* v is in the per-head weight layout */
        const uint16_t *kh = k + (size_t)hk * L * DH;                                    /* k is head-major too */
        for (uint32_t t = 0; t < nk; t++)
            for (int j = 0; j < 4; j++)
            {
                if (j == 0 && (t & 7) == 0 && t + 8 < nk) prefetch(kh + (size_t)(t + 8) * 4 * BH, 65536);
                /* K^T: group t / 2 of 64 keys, half t % 2, row block j (head dims 32j..32j+31): the block transposed */
                transpose_block(v_kt + (t / 2) * kt_group + (t % 2) * kt_half + j * BLK, kh + ((size_t)t * 4 + j) * BH);

            }
        scatter_fence(v_kt);
        }
        }
        uint64_t b = HAP_perf_get_time_us(); *t_tr += b - a;
        HVX_Vector rs[3][16]; sm_async_t sa;
        /* output block n's weights at v_kt + n * kt_half (kt_group = 2 kt_half): one call for the tile's nk score blocks */
        #define QK(qt_, dst_) hmx_mpy_row(v_q + (qt_) * 4 * BLK, qk_act, v_kt, qk_wgt, (dst_), kt_half, g_bias, nk)
        #define AV(qt_) do { uint8_t *src_ = v_s[(qt_) % 3]; HVX_Vector *rr_ = rs[(qt_) % 3]; for (uint32_t g = 0; g < 2; g++) { \
            hmx_mpy_chunks(src_, av_act, v_v + g * v_group, av_wgt, v_o + (2 * g) * BLK, 0, g_bias_av, av_chunk, L / kc); \
            hmx_mpy_chunks(src_, av_act, v_v + g * v_group + v_half, av_wgt, v_o + (2 * g + 1) * BLK, 0, g_bias_av, av_chunk, L / kc); } \
            if (g_hmxsum) { hmx_mpy_chunks(src_, av_act, v_one, av_wgt, v_sum, 0, g_bias_av, av_chunk, L / kc); \
                for (int kk = 0; kk < 16; kk += 4) recip_rows4((const HVX_Vector *)v_sum, rr_, kk, one_h); } \
            HVX_Vector *ov_ = (HVX_Vector *)v_o; for (int kk = 0; kk < 16; kk++) { HVX_Vector r_ = rr_[kk]; \
                const HVX_Vector o0_ = ov_[kk], o1_ = ov_[16 + kk], o2_ = ov_[32 + kk], o3_ = ov_[48 + kk]; \
                const HVX_Vector p0_ = hf(Q6_Vqf16_vmpy_VhfVhf(o0_, r_)), p1_ = hf(Q6_Vqf16_vmpy_VhfVhf(o1_, r_)), p2_ = hf(Q6_Vqf16_vmpy_VhfVhf(o2_, r_)), p3_ = hf(Q6_Vqf16_vmpy_VhfVhf(o3_, r_)); \
                ov_[kk] = p0_; ov_[16 + kk] = p1_; ov_[32 + kk] = p2_; ov_[48 + kk] = p3_; } \
            hvx_copy((uint8_t *)blk(o, D, (qt_), 4 * hd), v_o, 4 * BLK); } while (0)
        #define AVH(qt_, vo_, vs_) do { uint8_t *src_ = v_s[(qt_) % 3]; for (uint32_t g = 0; g < 2; g++) { \
            hmx_mpy_chunks(src_, av_act, v_v + g * v_group, av_wgt, (vo_) + (2 * g) * BLK, 0, g_bias_av, av_chunk, L / kc); \
            hmx_mpy_chunks(src_, av_act, v_v + g * v_group + v_half, av_wgt, (vo_) + (2 * g + 1) * BLK, 0, g_bias_av, av_chunk, L / kc); } \
            if (g_hmxsum) hmx_mpy_chunks(src_, av_act, v_one, av_wgt, (vs_), 0, g_bias_av, av_chunk, L / kc); } while (0)
        QK(0, v_s[0]);
        const int crew = (g_opt & OPT_CREW) && SMJ >= 2, nwk = SMJ - 1; static dsm_t dg;
        /* OPT_AVCREW: tile t's P V into buffer t & 1 (the second: v_c), its finish by the crew in tile t + 2's round */
        const int avc = crew && (g_opt & OPT_AVCREW) && nk >= 5;
        uint8_t *vob[2] = { v_o, v_c }, *vsb[2] = { v_sum, v_c + 4 * BLK };
        if (crew) dsm_start(&dg, nwk);
        for (uint32_t qt = 0; qt < nq; qt++)
        {
            /* causal: the keys past this tile's diagonal block masked as the second range (yrun's caller leaves that one empty) */
            const uint32_t mt[4] = { mk[0], mk[1], g_causal >= 0 ? (uint32_t)g_causal + 32 * (qt + 1) : mk[2], g_causal >= 0 ? L : mk[3] };
            const int dg_ = g_causal >= 0 ? g_causal / 32 + (int)qt : -1;
            if (crew) { dg.fo = 0; if (avc && qt >= 2) { dg.fo = vob[(qt - 2) & 1]; dg.fsum = vsb[(qt - 2) & 1]; dg.frr = rs[(qt - 2) % 3]; dg.fdst = (uint8_t *)blk(o, D, qt - 2, 4 * hd); dg.fhs = g_hmxsum; } }
            if (crew) dsm_go(&dg, (__fp16 *)v_s[qt % 3], nk, mt, rs[qt % 3], dg_); else softmax_start(&sa, (__fp16 *)v_s[qt % 3], nk, mt, rs[qt % 3], dg_);
            { const uint64_t a0_ = HAP_perf_get_time_us(); if (qt > 0) { if (avc) AVH(qt - 1, vob[(qt - 1) & 1], vsb[(qt - 1) & 1]); else AV(qt - 1); } const uint64_t a1_ = HAP_perf_get_time_us();
              if (qt + 1 < nq) QK(qt + 1, v_s[(qt + 1) % 3]); g_tatt[0] += a1_ - a0_; g_tatt[1] += HAP_perf_get_time_us() - a1_; }
            uint64_t c = HAP_perf_get_time_us();
            if (crew) dsm_finish(&dg, nwk);
            else { MAIN_TAKE; while (__atomic_load_n(&sa.done, __ATOMIC_ACQUIRE) < SMJ) { } }   /* spin: a synctoken wait costs ~40 us a tile on v81 */
            *t_sm += HAP_perf_get_time_us() - c;
        }
        if (crew) dsm_stop(&dg);
        if (avc)
        {   /* the last two tiles' finishes here */
            AVH(nq - 1, vob[(nq - 1) & 1], vsb[(nq - 1) & 1]);
            if (nq >= 2) av_finish(vob[(nq - 2) & 1], vsb[(nq - 2) & 1], rs[(nq - 2) % 3], (uint8_t *)blk(o, D, nq - 2, 4 * hd), g_hmxsum);
            av_finish(vob[(nq - 1) & 1], vsb[(nq - 1) & 1], rs[(nq - 1) % 3], (uint8_t *)blk(o, D, nq - 1, 4 * hd), g_hmxsum);
        }
        else AV(nq - 1);
        #undef AVH
        #undef QK
        #undef AV
    }
}

/* ------------------------------------------------------------------ the DSP for one call: clocks, VTCM, HMX, the worker pool */
static unsigned int g_vtot, g_vav;   /* the VTCM the chip reports: total, available (for the record) */
typedef struct { uint32_t ctx; unsigned int *hmx_busy; } eng_t;
static uint32_t g_tbs[6];   /* eng_begin's parts (us): DCVS vote, VTCM query + acquire, HMX power (+ clock), HVX power, HMX lock, pool + tables */
static int eng_begin(eng_t *e)
{
    uint64_t tb0 = HAP_perf_get_time_us(), tb1;
#define TB(i_) do { tb1 = HAP_perf_get_time_us(); g_tbs[i_] = (uint32_t)(tb1 - tb0); tb0 = tb1; } while (0)
    HAP_power_request_t pr; int ps = 0; g_splitn = 4; g_hM = g_ht0 = 0; g_lhdr = 0;
    const int vote = !(g_opt & OPT_KEEPVOTE) || !g_voted;
    memset(&pr, 0, sizeof(pr)); pr.type = HAP_power_set_DCVS_v3;
    pr.dcvs_v3.set_dcvs_enable = TRUE; pr.dcvs_v3.dcvs_enable = TRUE; pr.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    pr.dcvs_v3.set_bus_params = TRUE; pr.dcvs_v3.bus_params.min_corner = pr.dcvs_v3.bus_params.max_corner = pr.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    pr.dcvs_v3.set_core_params = TRUE; pr.dcvs_v3.core_params.min_corner = pr.dcvs_v3.core_params.max_corner = pr.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    pr.dcvs_v3.set_sleep_disable = TRUE; pr.dcvs_v3.sleep_disable = TRUE;
    if (vote) HAP_power_set((void *)&ps, &pr);
    TB(0);
    compute_res_attr_t res; HAP_compute_res_attr_init(&res);
    { unsigned int tot = 0, av = 0; compute_res_vtcm_page_t tl, al;
      const unsigned int need = g_need8 ? 8u * 1024 * 1024 : 4u * 1024 * 1024; g_need8 = 0;
      if (HAP_compute_res_query_VTCM(0, &tot, &tl, &av, &al) == 0 && av >= need) VTCM_BYTES = av > 32u * 1024 * 1024 ? 32u * 1024 * 1024 : av;
      else if (VTCM_BYTES < need) VTCM_BYTES = need;
      g_vtot = tot; g_vav = av;
      /* a chip with less than the engine's 8 MB at all (some cut-down v73 parts): say so instead of failing the
         acquire -- 0x4000 | its MB, which the app shows as the reason (Engine.why) */
      if (tot && tot < need) return 0x4000 | (int)(tot >> 20); }
    HAP_compute_res_attr_set_vtcm_param(&res, VTCM_BYTES, 1);
    unsigned int cap = 0; int lock4 = HAP_compute_res_query_capability(HAP_COMPUTE_RES_PREEMPTION_CAPABILITY, &cap) == 0 && (cap & HAP_COMPUTE_RES_HMX_LOCK4);
    if (!lock4) HAP_compute_res_attr_set_hmx_param(&res, 1);              /* with lock4 the HMX is locked separately */
    uint32_t ctx = HAP_compute_res_acquire(&res, 100000);
    if (!ctx) return AEE_ERESOURCENOTFOUND;
    e->ctx = ctx;
    g_vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&res); TB(1);
    if (vote) { memset(&pr, 0, sizeof(pr)); pr.type = HAP_power_set_HMX; pr.hmx.power_up = TRUE; HAP_power_set((void *)&ps, &pr); }
#if __HEXAGON_ARCH__ >= 75
    if (vote)
    {   /* the HMX's own clock (v75 on): without a vote it runs at its default, its lowest; at the top corner as llama.cpp's
           burst mode. A chip without a separate HMX clock answers AEE_EBADPARM: then the power_up above stands */
        memset(&pr, 0, sizeof(pr)); pr.type = HAP_power_set_HMX_v2; pr.hmx_v2.set_power = TRUE; pr.hmx_v2.power_up = TRUE; pr.hmx_v2.set_clock = TRUE;
        pr.hmx_v2.target_corner = pr.hmx_v2.min_corner = pr.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX; pr.hmx_v2.perf_mode = 0;
        g_hmxclk = HAP_power_set((void *)&ps, &pr);
    }
#endif
    TB(2); if (vote) { memset(&pr, 0, sizeof(pr)); pr.type = HAP_power_set_HVX; pr.hvx.power_up = TRUE; HAP_power_set((void *)&ps, &pr); }
    g_voted = 1; TB(3);
    /* v81 and later: lock4, and the thread marks itself busy before HMX instructions (the legacy lock gives garbage there) */
    unsigned int *hmx_busy = 0; int err;
    e->hmx_busy = 0;
    if (lock4)
    { err = HAP_compute_res_hmx_lock4(ctx, &hmx_busy, 100000); if (!err && hmx_busy) *hmx_busy = 1; else if (!err) err = AEE_EBADPARM; }
    else err = HAP_compute_res_hmx_lock(ctx);
    if (err) { HAP_compute_res_release(ctx); return err; }
    e->hmx_busy = hmx_busy; TB(4);
    if (!g_pool || !(g_opt & OPT_KEEPPOOL)) { if (g_pool) worker_pool_deinit(&g_pool); worker_pool_init(&g_pool); } transpose_init(); tri_init();
    g_nw = (int)num_hvx128_contexts; if (g_nw > (int)num_workers) g_nw = (int)num_workers; if (g_nw > MAXW) g_nw = MAXW; if (g_nw >= 8) g_nw -= (g_opt & OPT_WDMA) ? (__HVX_ARCH__ == 79 ? 0 : 1) : 2;   /* with the DMA copying: v81 7 of 8 contexts (61.9 -> 59.6 ms against 6 with an HVX weight copier), v79 all 8 (77.1~77.6 ms against 78.7~79.8 with 7) */
    if (g_nw < 2) g_nw = 2;
    /* the HMX bias / scale block (identity) lives at the top of VTCM, out of every stage's way */
    g_bias = g_vtcm + VTCM_BYTES - 4096; memset(g_bias, 0, 4096);
    for (int ch = 0; ch < 32; ch++) { ((uint32_t *)g_bias)[ch] = 0x00003C00u; ((uint32_t *)(g_bias + 128))[ch] = 0; }
    g_bias_av = g_bias + 2048;
    for (int ch = 0; ch < 32; ch++) { ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - (PSH << 10)); ((uint32_t *)(g_bias_av + 128))[ch] = 0; }
    g_dtail = 0;   /* this call's DMA chain starts fresh */
    TB(5);
#undef TB
    return 0;
}
static void pool_close(void) { if (g_pool) worker_pool_deinit(&g_pool); }
static void eng_end(eng_t *e)
{
    if (!(g_opt & OPT_KEEPPOOL)) worker_pool_deinit(&g_pool);    /* else kept for the session: making it was most of a small call's overhead */
    if (e->hmx_busy) { *e->hmx_busy = 0; HAP_compute_res_hmx_unlock4(e->ctx, e->hmx_busy); } else HAP_compute_res_hmx_unlock(e->ctx);
    HAP_compute_res_release(e->ctx);
}

/* the input layer: x = patches W^T / 16 (patches [T x 512]: 68 values a token, zero padded; W^T [512 x 2048], zero rows past 68) */
int kblock_embed(remote_handle64 h, const uint16_t *pat, int patLen, const uint16_t *w, int wLen, uint16_t *xo, int xoLen, int tokens)
{
    g_D = 2048; g_NH = 16; g_wfmt = 0;                      /* Anima; fp16 weights here */
    const uint32_t T = tokens;
    if ((size_t)patLen < (size_t)T * 512 || wLen < 512 * D || (size_t)xoLen < (size_t)T * D) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    uint64_t th = 0; product(pat, T, 512, w, D, xo, 128, 0, &th);
    eng_end(&eg); return 0;
}

/* the output layer: out = (LayerNorm(x) (1 + scale) + shift) W^T, W^T [2048 x 64]; fmod: shift | scale [2 x 2048] */
int kblock_final(remote_handle64 h, const uint16_t *x, int xLen, const uint16_t *w, int wLen, const float *fmod, int fmodLen, uint16_t *res, int resLen, int tokens)
{
    g_D = 2048; g_NH = 16; g_wfmt = 0;                      /* Anima; fp16 weights here */
    const uint32_t T = tokens;
    if ((size_t)xLen < (size_t)T * D || wLen < D * 64 || fmodLen < 2 * D || (size_t)resLen < (size_t)T * 64) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    static HVX_Vector shf[MAXD / 32], scl1[MAXD / 32];                          /* 16 KB: off the small DSP stack */
    for (int cb = 0; cb < D / 32; cb++) { float a[32], b2[32]; for (int c = 0; c < 32; c++) { a[c] = fmod[32 * cb + c]; b2[c] = 1.f + fmod[D + 32 * cb + c]; } shf[cb] = colvec(a); scl1[cb] = colvec(b2); }
    ln_arg_t lf = { x, 0, shf, scl1 };
    /* N = 64 is a single output group, which came out wrong on v81 (6.95% for the whole step against v73's 0.31%; the blocks and the
     * input layer were right): the product runs at N = 128 with a zero second group, and the first 64 columns are kept */
    const size_t wg = (size_t)2 * (D / 32) * BH;                                    /* halfwords of one 64-column group of W^T (its two 32-column halves) */
    uint16_t *w2 = (uint16_t *)memalign(128, 2 * wg * 2), *r2 = (uint16_t *)memalign(128, (size_t)T * 128 * 2);
    if (!w2 || !r2) { free(w2); free(r2); eng_end(&eg); return AEE_ENOMEMORY; }
    memcpy(w2, w, wg * 2); memset(w2 + wg, 0, wg * 2);
    uint64_t th = 0; wset_t w1 = { w2, 128, r2, OUT_ACT }; product_multi(0, ln_fill, &lf, T, D, &w1, 1, 128, 0, &th);
    for (uint32_t t = 0; t < T / 32; t++) memcpy(res + (size_t)t * 2 * BH, r2 + (size_t)t * 4 * BH, 2 * BLK);
    free(w2); free(r2);
    eng_end(&eg); return 0;
}

int kblock_run(remote_handle64 h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen,
               uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int ctxlen, int ffn, int kvalid, int inplace, uint32_t *times, int timesLen)
{
    g_smj = 4;
    g_D = 2048; g_NH = 16; g_wfmt = 0;                      /* Anima; fp16 weights here */
    const uint32_t T = tokens, C = ctxlen, CD = 1024, F = ffn;
    const size_t nW = (size_t)D * D * 6 + (size_t)CD * D * 2 + (size_t)D * F * 2, nI = (size_t)T * D + (size_t)C * CD + (size_t)T * 128;
    const size_t nS = (size_t)T * D * 6 + (size_t)T * F + (size_t)C * D * 2;
    if (scrLen == 16)
    {   /* the scratch here on the DSP (as k2run's), not 172 MB through FastRPC every call -- the host passes 16 for it */
        if (g_dscrn < nS) { free(g_dscr); g_dscr = (uint16_t *)memalign(128, nS * 2); g_dscrn = g_dscr ? nS : 0; if (!g_dscr) return AEE_ENOMEMORY; }
        scr = g_dscr; scrLen = (int)nS;
    }
    g_wfmt = (inplace >> 8) & 3;                                     /* flags 0x100 / 0x200: q8 / q4 weights */
    if (g_wfmt > 2) return AEE_EBADPARM;   /* 0: fp16, 1: q8, 2: q4; 3 is no format */
    if ((size_t)wtsLen * 2 < wbytes(nW) || (size_t)inpLen < nI || prmLen < 9 * D + 4 * DH || (size_t)scrLen < nS || xoLen < (int)(T * D) || timesLen < 12) return AEE_EBADPARM;
    uint64_t t_start = HAP_perf_get_time_us();
    eng_t eg; int err = eng_begin(&eg); if (err) return err; const uint32_t t_beg = (uint32_t)(HAP_perf_get_time_us() - t_start);   /* what a call's setup costs */

    /* weights, in the order the host packs them */
    const uint16_t *wq = wts, *wk = WP(wq, D * D), *wv = WP(wk, D * D), *wo = WP(wv, D * D), *wcq = WP(wo, D * D), *wck = WP(wcq, D * D), *wcv = WP(wck, CD * D),
                   *wco = WP(wcv, CD * D), *wm1 = WP(wco, D * D), *wm2 = WP(wm1, (size_t)D * F);
    /* a LoRA applied as it runs (flag 0x1000): its pairs follow the weights in the same buffer, A [K x LR] then B [LR x N] in the
     * weight layout, the ten Linears in the order above; the strength is inplace's top 16 bits over 1000. No baking, so one copy
     * of the weights serves every LoRA, and the strength may differ from step to step. */
    const uint16_t *la[10] = { 0 }, *lb[10] = { 0 };
    if (inplace & 0x1000)
    {
        const uint32_t KK[10] = { D, D, D, D, D, CD, CD, D, D, F }, NN[10] = { D, D, D, D, D, D, D, D, F, D };
        g_lhdr = 1;
        const uint16_t *p = (const uint16_t *)((const uint8_t *)wts + wbytes(nW)), *pe = wts + wtsLen;
        for (int m = 0; m < 10; m++) { if (p + 64 > pe) { eng_end(&eg); return AEE_EBADPARM; } p = lora_next(p, KK[m], NN[m], &la[m], &lb[m]); if (p > pe) { eng_end(&eg); return AEE_EBADPARM; } }
        g_lora = (float)((inplace >> 16) & 0xFFFF) / 1000.f;
    }
    else g_lora = 1.f;
    const uint16_t *ctxin = inp + (size_t)T * D, *rope = ctxin + (size_t)C * CD;
    uint16_t *x = xo, *hb = scr, *qb = hb + (size_t)T * D, *kb = qb + (size_t)T * D, *vb = kb + (size_t)T * D, *ob = vb + (size_t)T * D,
             *yb = ob + (size_t)T * D, *hid = yb + (size_t)T * D, *ck = hid + (size_t)T * F, *cv = ck + (size_t)C * D;
    if (!(inplace & 1)) hvx_copy((uint8_t *)x, (const uint8_t *)inp, T * D * 2);   /* inplace: the stream is already in xo (the block before's output) */
    /* column vectors of the modulations: [modulation][shift, 1 + scale, gate / 16][column block] */
    static HVX_Vector mv[3][3][MAXD / 32];
    for (int m = 0; m < 3; m++)
        for (uint32_t cb = 0; cb < D / 32; cb++)
        {
            float sh[32], sc1[32], gt[32];
            for (int c = 0; c < 32; c++) { sh[c] = prm[(m * 3 + 0) * D + cb * 32 + c]; sc1[c] = 1.f + prm[(m * 3 + 1) * D + cb * 32 + c]; gt[c] = prm[(m * 3 + 2) * D + cb * 32 + c] / RS; }
            mv[m][0][cb] = colvec(sh); mv[m][1][cb] = colvec(sc1); mv[m][2][cb] = colvec(gt);
        }
    const float *nrm = prm + 9 * D;
    uint64_t tm[12] = {0}, th = 0, ttr = 0, tsm = 0;
    #define STAGE(i, ...) do { uint64_t s_ = HAP_perf_get_time_us(); __VA_ARGS__; tm[i] += HAP_perf_get_time_us() - s_; } while (0)
    /* self-attention */
    ln_arg_t lf[3]; for (int i = 0; i < 3; i++) lf[i] = (ln_arg_t){ x, 0, mv[i][0], mv[i][1] };
    /* stage 2 now includes the LayerNorm (made into VTCM as the activation tiles) */
    const float QS = 1.44269504f / sqrtf((float)DH);
    const int fr = (g_opt & OPT_RRPOST) != 0;   /* q's and k's norm and RoPE in the products' post */
    static HVX_Vector rrv[4][4];
    if (fr) { rr_vecs(rrv[0], nrm, QS); rr_vecs(rrv[1], nrm + DH, 1.f); rr_vecs(rrv[2], nrm + 2 * DH, QS); rr_vecs(rrv[3], nrm + 3 * DH, 1.f); }
    STAGE(2, { wset_t w3[3] = { { wq, D, qb, OUT_HEADQ, 0, 0, 0, la[0], lb[0] }, { wk, D, kb, OUT_HEADQ, 0, 0, 0, la[1], lb[1] }, { wv, D, vb, OUT_HEADV, 0, 0, 0, la[2], lb[2] } };
               if (fr) { w3[0].rrw = rrv[0]; w3[0].rrope = rope; w3[1].rrw = rrv[1]; w3[1].rrope = rope; }
               product_multi(0, ln_fill, &lf[0], T, D, w3, 3, 128, 0, &th); });
    STAGE(3, { if (!fr) { rms_rope(qb, T, nrm, rope, 1, QS, NH); rms_rope(kb, T, nrm + DH, rope, 1, 1.f, NH); } });
    g_hmxsum = 1;
    const uint32_t mks[4] = { (uint32_t)kvalid, T, 0, 0 }, mkc[4] = { 0, 0, 0, 0 };
    /* P V's output scaled down 2^-4 more, as for Krea and Klein: head 15 of block 27, its scores nearly flat over 4096 keys, passed 65504 before the division (inf on v81, which showed up as noise) */
    if (g_hmxsum) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - ((PSH + 4) << 10));
    STAGE(4, attention(qb, T, kb, vb, T, mks, 1, ob, &ttr, &tsm));
    if (g_hmxsum) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - (PSH << 10));
    const int rp = (g_opt & OPT_RESPOST) != 0;
    STAGE(5, { if (rp) { g_resg = mv[0][2]; product_lo(ob, T, D, wo, D, x, OUT_RESG, 128, la[3], lb[3], &th); }
               else { product_lo(ob, T, D, wo, D, yb, OUT_ACT, 128, la[3], lb[3], &th); residual(x, yb, T, mv[0][2]); } });
    /* cross-attention: keys and values from the text */
    STAGE(2, { wset_t w1 = { wcq, D, qb, OUT_HEADQ, 0, 0, 0, la[4], lb[4] }; if (fr) w1.rrw = rrv[2]; product_multi(0, ln_fill, &lf[1], T, D, &w1, 1, 128, 0, &th);
               wset_t wk_ = { wck, D, ck, OUT_HEADQ, 0, 0, 0, la[5], lb[5] }; if (fr) wk_.rrw = rrv[3]; product_multi(ctxin, 0, 0, C, CD, &wk_, 1, 16, 0, &th);
               product_lo(ctxin, C, CD, wcv, D, cv, OUT_HEADV, 16, la[6], lb[6], &th); });
    STAGE(3, { if (!fr) { rms_rope(qb, T, nrm + 2 * DH, 0, 1, QS, NH); rms_rope(ck, C, nrm + 3 * DH, 0, 1, 1.f, NH); } });
    STAGE(4, attention(qb, T, ck, cv, C, mkc, 1, ob, &ttr, &tsm));
    STAGE(5, { if (rp) { g_resg = mv[1][2]; product_lo(ob, T, D, wco, D, x, OUT_RESG, 128, la[7], lb[7], &th); }
               else { product_lo(ob, T, D, wco, D, yb, OUT_ACT, 128, la[7], lb[7], &th); residual(x, yb, T, mv[1][2]); } });
    /* MLP */
    /* the down product in four K quarters (the hidden written as four [T x F/4] parts): K = F/4 lets ~4x the row tiles into the VTCM,
     * so the 32 MB of weights are read ~4x fewer times (the weights stay as stored: each quarter copies its slice of the rows) */
    STAGE(6, { wset_t w1 = { wm1, F, hid, OUT_SPLIT4, 0, 0, 0, la[8], lb[8] }; product_multi(0, ln_fill, &lf[2], T, D, &w1, 1, 128, 1, &th);
               g_resg = mv[2][2]; g_resx = x;
               for (uint32_t q = 0; q < 4; q++) { wset_t w2 = { wm2, D, yb, q ? (rp && q == 3 ? OUT_ADDRESG : OUT_ADD) : OUT_ACT, F, q * (F / 4), 0, la[9], lb[9] };
                   product_multi(hid + (size_t)q * T * (F / 4), 0, 0, T, F / 4, &w2, 1, 128, 0, &th); }
               if (!rp) residual(x, yb, T, mv[2][2]); });
    #undef STAGE

    eng_end(&eg);
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t_start);
    for (int i = 1; i <= 6; i++) times[i] = (uint32_t)tm[i];
    times[7] = (uint32_t)th; times[8] = (uint32_t)ttr; times[9] = (uint32_t)tsm; times[10] = (uint32_t)g_tav; times[11] = (uint32_t)g_tqk;
    if (timesLen > 12) times[12] = (uint32_t)g_tpost;   /* the waits for post jobs */
    if (timesLen > 13) times[13] = t_beg;
    if (timesLen > 19) for (int i = 0; i < 6; i++) times[14 + i] = g_tbs[i];
    if (timesLen > 23) times[22] = times[23] = 0;   /* the core and HMX clocks are not read in this build: these two slots stay 0 */
    if (timesLen > 25) { times[24] = (uint32_t)g_nw; times[25] = VTCM_BYTES >> 10; }
    if (timesLen > 21) { times[20] = (uint32_t)g_tatt[0]; times[21] = (uint32_t)g_tatt[1]; }
    g_tatt[0] = g_tatt[1] = 0;
    g_tav = g_tqk = g_tpost = 0;
    return 0;
}

/* ------------------------------------------------------------------ Krea 2 (single stream: text + image tokens, GQA, gated attention, SwiGLU) */
/* a = a * sigmoid(b) (gate: a the attention output, b the gate) or a = silu(a) * b (swiglu), n fp16 values in the same layout */
typedef struct { __fp16 *a; const __fp16 *b; uint32_t v0, v1; int swiglu; spintok_t *tok; } ew_job_t;
static void ew_worker(void *p)
{
    ew_job_t *j = (ew_job_t *)p; HVX_Vector *A = (HVX_Vector *)j->a; const HVX_Vector *B = (const HVX_Vector *)j->b;
    for (uint32_t i = j->v0; i < j->v1; i++)
    {
        if (((i - j->v0) & 63) == 0 && i + 128 <= j->v1) { prefetch(A + i + 64, 8192); prefetch(B + i + 64, 8192); }   /* never past the end: an l2fetch there faults */
        A[i] = j->swiglu ? hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(A[i], sigmoid_hf(A[i]))), B[i])) : hf(Q6_Vqf16_vmpy_VhfVhf(A[i], sigmoid_hf(B[i])));
    }
    st_done(j->tok);
}
static void elementwise(__fp16 *a, const __fp16 *b, size_t n, int swiglu)
{
    const int nw = g_nw; ew_job_t jobs[MAXW]; spintok_t tok; st_init(&tok, nw); const uint32_t nv = (uint32_t)(n / 64);
    for (int t = 0; t < nw; t++) { jobs[t] = (ew_job_t){ a, b, nv * t / nw, nv * (t + 1) / nw, swiglu, &tok }; worker_pool_job_t job = { ew_worker, &jobs[t] }; worker_pool_submit(g_pool, job); }
    st_wait(&tok);
}

/* FLUX.2 Klein's double block (k2run flag 0x800): the rows 0..T-1 the text, T.. the image, each stream its own products, modulations and q / k
 * norms, one attention over all. D 3072, 24 heads, no grouped kv, SiLU gated MLP 9216, LayerNorm without weights, no bias.
 *   wts: for the text, then the image: q, k (head-permuted), v, out, gate, up, down as three K = 3072 parts
 *   prm: [text, image][pre, post][scale, shift, gate] D each | text q, k norms | image q, k norms (permuted, 128 each)
 *   keys [ka, kb) masked: the text's padding, so kb is T, a multiple of 32; [ka2, kb2) the spare rows after the image */
static int klein_double(const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen,
                        int tokens, int ka, int kb, int ka2, int kb2, int inplace, uint32_t *times, int timesLen)
{
    const int T = kb;      /* the text's rows end where its padding does */
    g_smj = 6; g_D = (prmLen - 4 * 128) / 12; g_NH = g_D / 128; g_situ16 = 0; g_wfmt = (inplace >> 8) & 3; g_lora = 1.f;   /* the width from the parameters: 4B 3072, 9B 4096 */
    if (g_wfmt > 2) return AEE_EBADPARM;   /* 0: fp16, 1: q8, 2: q4; 3 is no format */
    const int lo = (inplace & 0x1000) != 0; if (lo) { g_lhdr = 1; g_lora = (float)((inplace >> 16) & 0xFFFF) / 1000.f; }
    const uint32_t L = tokens, F = 3 * D, FQ = D; const uint32_t M[2] = { (uint32_t)T, L - T }, t0s[2] = { 0, (uint32_t)T / 32 };
    const size_t nW1 = (size_t)D * D * 4 + (size_t)D * F * 3, nS = (size_t)L * (D * 6 + F * 2);
    if ((size_t)wtsLen * 2 < 2 * wbytes(nW1) || (size_t)inpLen < (size_t)L * (D + 128) || prmLen < (int)(12 * D + 4 * DH) || (size_t)scrLen < nS || (size_t)xoLen < (size_t)L * D || timesLen < 12 ||
        L % 32 || T % 32 || !T || (uint32_t)T >= L) return AEE_EBADPARM;
    uint64_t t_start = HAP_perf_get_time_us();
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_splitn = 3;
    const uint16_t *rope = inp + (size_t)L * D;
    uint16_t *x = xo, *qb = scr, *kbuf = qb + (size_t)L * D, *vb = kbuf + (size_t)L * D, *ob = vb + (size_t)L * D, *yb = ob + (size_t)L * D, *ha = yb + (size_t)L * D;
    static HVX_Vector mv[2][2][3][MAXD / 32];
    for (int s = 0; s < 2; s++) for (int m = 0; m < 2; m++)
        for (uint32_t cb = 0; cb < D / 32; cb++)
        {
            float sh[32], sc[32], gt[32]; const float *mod = prm + (size_t)(s * 6 + m * 3) * D;
            for (int c = 0; c < 32; c++) { const uint32_t i = 32 * cb + c; sc[c] = 1.f + mod[i]; sh[c] = mod[D + i]; gt[c] = mod[2 * D + i]; }
            mv[s][m][0][cb] = colvec(sh); mv[s][m][1][cb] = colvec(sc); mv[s][m][2][cb] = colvec(gt);
        }
    const float *nrm = prm + 12 * D;
    uint64_t tm[8] = { 0 }, th = 0, ttr = 0, tsm = 0;
    #define STAGE(i, ...) do { uint64_t s_ = HAP_perf_get_time_us(); __VA_ARGS__; tm[i] += HAP_perf_get_time_us() - s_; } while (0)
    const float QS = 1.44269504f / sqrtf((float)DH);
    const uint16_t *w[2][7];
    { const uint16_t *p = wts;
      for (int s = 0; s < 2; s++) { for (int m = 0; m < 4; m++) { w[s][m] = p; p = WP(p, (size_t)D * D); } for (int m = 4; m < 6; m++) { w[s][m] = p; p = WP(p, (size_t)D * F); } w[s][6] = p; p = WP(p, (size_t)D * F); } }
    const uint16_t *la[2][7] = { { 0 } }, *lb[2][7] = { { 0 } };
    if (lo)
    {   /* after the weights, per stream per Linear A^T [K x LR] then B^T [LR x N]; unlike kblock_run's, this walk is not checked against wtsLen */
        const uint32_t KK[7] = { D, D, D, D, D, D, F }, NN[7] = { D, D, D, D, F, F, D };
        const uint16_t *p = (const uint16_t *)((const uint8_t *)wts + 2 * wbytes(nW1));
        for (int s = 0; s < 2; s++) for (int m = 0; m < 7; m++) p = lora_next(p, KK[m], NN[m], &la[s][m], &lb[s][m]);
    }
    for (int s = 0; s < 2; s++)
    {
        ln_arg_t lf = { blk(x, D, t0s[s], 0), 0, mv[s][0][0], mv[s][0][1], 0 };
        wset_t w3[3] = { { w[s][0], D, qb, OUT_HEADQ, 0, 0, 0, la[s][0], lb[s][0] }, { w[s][1], D, kbuf, OUT_HEADQ, 0, 0, 0, la[s][1], lb[s][1] }, { w[s][2], D, vb, OUT_HEADV, 0, 0, 0, la[s][2], lb[s][2] } };
        const int frd = (g_opt & OPT_RRPOST) != 0;   /* this stream's q, k norm and RoPE in the post (the RoPE rows from its first) */
        static HVX_Vector rrd[2][4];
        if (frd) { rr_vecs(rrd[0], nrm + (size_t)s * 2 * DH, QS); rr_vecs(rrd[1], nrm + (size_t)s * 2 * DH + DH, 1.f);
                   const uint16_t *rp0 = blk((uint16_t *)rope, 128, t0s[s], 0); w3[0].rrw = rrd[0]; w3[0].rrope = rp0; w3[1].rrw = rrd[1]; w3[1].rrope = rp0; }
        g_hM = L; g_ht0 = t0s[s];
        STAGE(1, product_multi(0, ln_fill, &lf, M[s], D, w3, 3, 128, 0, &th));
        g_hM = 0; g_ht0 = 0;
        rr_arg_t aq = { qb, nrm + (size_t)s * 2 * DH, rope, L / 32, 1, QS, NH, 0, 0 }, ak = { kbuf, nrm + (size_t)s * 2 * DH + DH, rope, L / 32, 1, 1.f, NH, 0, 0 };
        if (!frd) STAGE(2, { tiles_parallel_range(rr_tiles, &aq, t0s[s], t0s[s] + M[s] / 32); tiles_parallel_range(rr_tiles, &ak, t0s[s], t0s[s] + M[s] / 32); });
    }
    const uint32_t mk[4] = { (uint32_t)ka, (uint32_t)kb, (uint32_t)ka2, (uint32_t)kb2 };
    if (g_hmxsum) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - ((PSH + 4) << 10));
    STAGE(3, attention(qb, L, kbuf, vb, L, mk, 1, ob, &ttr, &tsm));
    if (g_hmxsum) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - (PSH << 10));
    for (int s = 0; s < 2; s++)
    {
        uint16_t *xs = blk(x, D, t0s[s], 0);
        const int rpd = (g_opt & OPT_RESPOST) != 0;   /* the residuals in the products' post */
        STAGE(4, { if (rpd) { g_resg = mv[s][0][2]; product_lo(blk(ob, D, t0s[s], 0), M[s], D, w[s][3], D, xs, OUT_RESG, 128, la[s][3], lb[s][3], &th); }
                   else { product_lo(blk(ob, D, t0s[s], 0), M[s], D, w[s][3], D, yb, OUT_ACT, 128, la[s][3], lb[s][3], &th); residual(xs, yb, M[s], mv[s][0][2]); } });
        ln_arg_t lf = { xs, 0, mv[s][1][0], mv[s][1][1], 0 };
        wset_t w2[2] = { { w[s][4], F, ha, OUT_SPLIT4, 0, 0, 0, la[s][4], lb[s][4] }, { w[s][5], F, ha, OUT_SPLIT4, 0, 0, 0, la[s][5], lb[s][5] } };
        STAGE(5, product_multi(0, ln_fill, &lf, M[s], D, w2, 2, 128, 2, &th));
        g_resg = mv[s][1][2]; g_resx = xs;
        STAGE(6, { for (uint32_t q = 0; q < 3; q++) product_lo(ha + (size_t)q * M[s] * FQ, M[s], FQ, WP(w[s][6], (size_t)q * FQ * D), D, yb, q ? (rpd && q == 2 ? OUT_ADDRESG : OUT_ADD) : OUT_ACT, 128, la[s][6] ? la[s][6] + (size_t)q * FQ * lb[s][6][-64] : 0, lb[s][6], &th);
                   if (!rpd) residual(xs, yb, M[s], mv[s][1][2]); });
    }
    #undef STAGE
    eng_end(&eg);
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t_start);
    for (int i = 1; i <= 6; i++) times[i] = (uint32_t)tm[i];
    times[7] = (uint32_t)th; times[8] = (uint32_t)ttr; times[9] = (uint32_t)tsm; times[10] = (uint32_t)g_tav; times[11] = (uint32_t)g_tqk;
    if (timesLen > 12) times[12] = (uint32_t)g_tpost;   /* the waits for post jobs */
    if (timesLen > 23) times[22] = times[23] = 0;   /* the core and HMX clocks are not read in this build: these two slots stay 0 */
    g_tav = g_tqk = g_tpost = 0;
    return 0;
}

/* one Krea 2 block, x [L x 6144] in place in xo (from inp's first L x 6144 unless inplace), rope [L x 128] after it in inp.
 * wts: wq, wk, wv, gate, wo, mlp gate, up, down (W^T weight layouts, q / k rows head-permuted to even | odd dims).
 * prm: tvec + mod [6 x D] (prescale, preshift, pregate, postscale, postshift, postgate) | pre_w | post_w | qn_w | kn_w (scales + 1).
 * Keys ka..kb-1 (the text padding) masked. */
int kblock_k2run(remote_handle64 h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen,
                 uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int ka, int kb, int ka2, int kb2, int inplace, uint32_t *times, int timesLen)
{
    if ((inplace & 0xC00) && scrLen == 16)
    {
        const size_t need = (size_t)tokens * 12 * ((prmLen - 512) / 12);
        if (g_dscrn < need) { free(g_dscr); g_dscr = (uint16_t *)memalign(128, need * 2); g_dscrn = g_dscr ? need : 0; if (!g_dscr) return AEE_ENOMEMORY; }
        scr = g_dscr; scrLen = (int)need;
    }
    if (inplace & 0x800) return klein_double(wts, wtsLen, inp, inpLen, prm, prmLen, scr, scrLen, xo, xoLen, tokens, ka, kb, ka2, kb2, inplace, times, timesLen);
    /* flag 0x400 (with 0x4): Klein's single block: as MOSS's, with SiLU gating, MLP 9216, a LayerNorm, one residual from the attention's out product and the MLP's down product */
    const int klein = (inplace & 0x400) != 0;
    g_smj = 6;            /* 6 softmax jobs: on v81 its attention 67 -> 60 ms a block; v73 has 4 contexts anyway */
    /* flag 0x4: a MOSS-Image block instead (D 3072, 24 heads, no grouped kv, ffn 8192 with its situ activation, no attention gate:
     * weights wq, wk, wv, wo, mlp gate, up, down; the norms' weights as they are, the modulation shared by all blocks) */
    const int moss = (inplace & 0x4) != 0;
    /* fp16 situ: block 252.7 -> 234.5 ms and 0.082 -> 0.066 %; bit 25 = the fp32 one, read only without a LoRA (with one, bits
       16-31 are its strength: 1000 has bit 25 set, and every DMD run had gone through the fp32 activation at a cost of 19 ms a block) */
    g_situ16 = moss && ((inplace & 0x1000) || !(inplace & (1 << 25)));
    g_D = klein ? (prmLen - 4 * 128) / 12 : moss ? 3072 : 6144; g_NH = D / 128;
    if (klein) g_situ16 = 0;
    const uint32_t L = tokens, KV = moss ? (uint32_t)D : 1536, F2 = klein ? 3 * D : moss ? 8192 : 16384, KREP = moss ? 1 : 4;
    const size_t nW = (size_t)D * D * (moss ? 2 : 3) + (size_t)D * KV * 2 + (size_t)D * F2 * 3, nS = (size_t)L * (D * 4 + KV * 2 + F2 * 2);   /* F2 * 2: the hidden, and room for a second one that nothing uses */
    g_wfmt = (inplace >> 8) & 3;                                     /* flags 0x100 / 0x200: q8 / q4 weights */
    if (g_wfmt > 2) return AEE_EBADPARM;   /* 0: fp16, 1: q8, 2: q4; 3 is no format */
    if ((size_t)wtsLen * 2 < wbytes(nW) || (size_t)inpLen < (size_t)L * (D + 128) || prmLen < (int)(8 * D + 2 * DH) || (size_t)scrLen < nS || (size_t)xoLen < (size_t)L * D || timesLen < 12 || L % 32) return AEE_EBADPARM;
    /* (MOSS: 4192 rows, 131 blocks, so attention's AV chunk of 1 block, came out 80 % off; an odd chunk is wrong, YuE2 found 13 too --
       MossGen.rows() picks a length with an even one) */
    uint64_t t_start = HAP_perf_get_time_us();
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    if (klein) g_splitn = 3;
    const uint16_t *wq = wts, *wk = WP(wq, (size_t)D * D), *wv = WP(wk, (size_t)D * KV), *wg = moss ? 0 : WP(wv, (size_t)D * KV), *wo = moss ? WP(wv, (size_t)D * KV) : WP(wg, (size_t)D * D),
                   *wmg = WP(wo, (size_t)D * D), *wmu = WP(wmg, (size_t)D * F2), *wmd = WP(wmu, (size_t)D * F2);
    /* a LoRA applied as it runs (flag 0x1000, as kblock_run): after the weights, per Linear A^T [K x LR] then B^T
     * [LR x N] in the weight layout (wq, wk, wv, attn gate, wo, mlp gate, up, down; wq / wk's columns in the engine's head order),
     * down's A^T as four K = 4096 quarters like its weights; the strengths are in B */
    const uint16_t *la[8] = { 0 }, *lb[8] = { 0 };
    if (inplace & 0x1000)
    {
        const uint32_t KK[8] = { D, D, D, D, D, D, D, F2 }, NN[8] = { D, KV, KV, D, D, F2, F2, D };
        g_lhdr = 1;
        const uint16_t *p = (const uint16_t *)((const uint8_t *)wts + wbytes(nW)), *pe = wts + wtsLen;
        for (int m = 0; m < 8; m++) { if (moss && m == 3) continue; if (p + 64 > pe) { eng_end(&eg); return AEE_EBADPARM; } p = lora_next(p, KK[m], NN[m], &la[m], &lb[m]); if (p > pe) { eng_end(&eg); return AEE_EBADPARM; } }
        g_lora = (float)((inplace >> 16) & 0xFFFF) / 1000.f;
    }
    else g_lora = 1.f;
    const uint16_t *rope = inp + (size_t)L * D;
    uint16_t *x = xo, *qb = scr, *kbuf = qb + (size_t)L * D, *vb = kbuf + (size_t)L * KV, *gb = vb + (size_t)L * KV, *ob = gb + (size_t)L * D,
             *yb = ob + (size_t)L * D, *ha = yb + (size_t)L * D;
    if (!(inplace & 1)) hvx_copy((uint8_t *)x, (const uint8_t *)inp, L * D * 2);
    /* column vectors: [pre | post][shift, (1 + scale) * norm weight, gate] */
    static HVX_Vector mv[2][3][MAXD / 32];
    for (int m = 0; m < 2; m++)
        for (uint32_t cb = 0; cb < D / 32; cb++)
        {
            float sh[32], sc[32], gt[32]; const float *mod = prm + (size_t)m * 3 * D, *nw = prm + 6 * D + (size_t)m * D;
            for (int c = 0; c < 32; c++) { uint32_t i = 32 * cb + c; sc[c] = (1.f + mod[i]) * nw[i]; sh[c] = mod[D + i]; gt[c] = mod[2 * D + i]; }
            mv[m][0][cb] = colvec(sh); mv[m][1][cb] = colvec(sc); mv[m][2][cb] = colvec(gt);
        }
    const float *qn = prm + 8 * D, *kn = qn + DH;
    uint64_t tm[8] = { 0 }, th = 0, ttr = 0, tsm = 0;
    #define STAGE(i, ...) do { uint64_t s_ = HAP_perf_get_time_us(); __VA_ARGS__; tm[i] += HAP_perf_get_time_us() - s_; } while (0)
    ln_arg_t lf[2] = { { x, 0, mv[0][0], mv[0][1], !klein }, { x, 0, mv[1][0], mv[1][1], !klein } };
    const float QS = 1.44269504f / sqrtf((float)DH);
    const int fr = (g_opt & OPT_RRPOST) != 0;   /* q's and k's norm and RoPE in the post */
    static HVX_Vector rrk[2][4]; if (fr) { rr_vecs(rrk[0], qn, QS); rr_vecs(rrk[1], kn, 1.f); }
    STAGE(1, { wset_t w4[4] = { { wq, D, qb, OUT_HEADQ, 0, 0, 0, la[0], lb[0] }, { wk, KV, kbuf, OUT_HEADQ, 0, 0, 0, la[1], lb[1] },
                                { wv, KV, vb, OUT_HEADV, 0, 0, 0, la[2], lb[2] }, { wg, D, gb, OUT_ACT, 0, 0, 0, la[3], lb[3] } };
               if (fr) { w4[0].rrw = rrk[0]; w4[0].rrope = rope; w4[1].rrw = rrk[1]; w4[1].rrope = rope; }
               product_multi(0, ln_fill, &lf[0], L, D, w4, moss ? 3 : 4, 128, 0, &th); });
    /* debug (no LoRA; bits 16-18 = n): stop after a stage with an intermediate [L x D] in x -- 1 the attention gate's output, 2 the attention's
       output (before the gate), 3 the out projection's (before the residual), 4 q after rms_rope, 5 k, 6 v (Klein: the MLP hidden), 7 the VTCM figures -- for
       comparing a chip against another */
    const int dstage = !(inplace & 0x1000) ? (inplace >> 16) & 7 : 0;
    if (dstage == 7) { tm[1] = VTCM_BYTES; tm[2] = (uint64_t)g_nw; tm[3] = g_vtot; tm[4] = g_vav; goto k2done; }   /* 7: VTCM used, HVX workers, VTCM total, available in times[1..4] */
    if (dstage == 1) { hvx_copy((uint8_t *)x, (const uint8_t *)gb, L * D * 2); goto k2done; }
    STAGE(2, { if (!fr) { rms_rope(qb, L, qn, rope, 1, QS, NH); rms_rope(kbuf, L, kn, rope, 1, 1.f, KV / DH); } });
    if (dstage == 4) { hvx_copy((uint8_t *)x, (const uint8_t *)qb, L * D * 2); goto k2done; }      /* 4: q after its norm and RoPE (head layout) */
    if (dstage == 5) { hvx_copy((uint8_t *)x, (const uint8_t *)kbuf, L * KV * 2); goto k2done; }  /* 5: k after its norm and RoPE, 6: v (the rest of x as it came) */
    if (dstage == 6 && !klein) { hvx_copy((uint8_t *)x, (const uint8_t *)vb, L * KV * 2); goto k2done; }   /* Klein's 6: the MLP hidden, below */
    const uint32_t mk[4] = { (uint32_t)ka, (uint32_t)kb, (uint32_t)ka2, (uint32_t)kb2 };
    /* P V's output scaled down 2^-4 more (with the HMX row sums the same scale cancels in their reciprocal): over Krea's 4224 keys the
       unnormalised sum p v passed 65504 in some heads -- inf on v81 (the block 336 % off), clamped (still wrong) on v73 */
    const int avx = g_hmxsum;
    if (avx) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - ((PSH + 4) << 10));
    STAGE(3, attention(qb, L, kbuf, vb, L, mk, KREP, ob, &ttr, &tsm));
    if (avx) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - (PSH << 10));
    if (dstage == 2) { hvx_copy((uint8_t *)x, (const uint8_t *)ob, L * D * 2); goto k2done; }
    if (klein)
    {   /* attention and MLP from the same norm of x, one residual */
        STAGE(4, product_lo(ob, L, D, wo, D, yb, OUT_ACT, 128, la[4], lb[4], &th));
        if (dstage == 3) { hvx_copy((uint8_t *)x, (const uint8_t *)yb, L * D * 2); goto k2done; }   /* 3: the attention's out projection */
        STAGE(5, { wset_t w2[2] = { { wmg, F2, ha, OUT_SPLIT4, 0, 0, 0, la[5], lb[5] }, { wmu, F2, ha, OUT_SPLIT4, 0, 0, 0, la[6], lb[6] } }; product_multi(0, ln_fill, &lf[0], L, D, w2, 2, 128, 2, &th); });
        if (dstage == 6) { hvx_copy((uint8_t *)x, (const uint8_t *)ha, L * D * 2); goto k2done; }   /* 6: the MLP hidden after silu(gate) * up, its first L x D */
        const int rpk = (g_opt & OPT_RESPOST) != 0; g_resg = mv[0][2]; g_resx = x;   /* the residuals in the post */
        STAGE(6, { for (uint32_t q = 0; q < 3; q++) product_lo(ha + (size_t)q * L * (F2 / 3), L, F2 / 3, WP(wmd, (size_t)q * (F2 / 3) * D), D, yb, rpk && q == 2 ? OUT_ADDRESG : OUT_ADD, 128, la[7] ? la[7] + (size_t)q * (F2 / 3) * lb[7][-64] : 0, lb[7], &th);
                   if (!rpk) residual(x, yb, L, mv[0][2]); });
        goto k2done;
    }
    const int rp = (g_opt & OPT_RESPOST) && dstage != 3; g_resg = mv[0][2]; g_resx = x;   /* the residuals in the post */
    STAGE(4, { if (!moss) elementwise((__fp16 *)ob, (const __fp16 *)gb, (size_t)L * D, 0);
               product_lo(ob, L, D, wo, D, rp ? x : yb, rp ? OUT_RESG : OUT_ACT, 128, la[4], lb[4], &th);
               if (dstage == 3) { hvx_copy((uint8_t *)x, (const uint8_t *)yb, L * D * 2); goto k2done; }
               if (!rp) residual(x, yb, L, mv[0][2]); });
    /* gate and up groups interleaved, silu(gate) * up made in VTCM: only the hidden row goes to DDR */
    STAGE(5, { wset_t w2[2] = { { wmg, F2, ha, OUT_SPLIT4, 0, 0, 0, la[5], lb[5] }, { wmu, F2, ha, OUT_SPLIT4, 0, 0, 0, la[6], lb[6] } }; product_multi(0, ln_fill, &lf[1], L, D, w2, 2, 128,
                                     moss ? 3 : 2, &th); });
    /* down in four K = 4096 quarters (the hidden's and the weight's): with K = 16384 only three row tiles fit VTCM and the 200MB of
     * weights were read 44 times a block; summed in fp16 into yb;
     * the hidden in four quarters (OUT_SPLIT4), the weight as four K quarters to match (Krea 4096 each, MOSS 2048) */
    g_resg = mv[1][2];
    STAGE(6, { for (int q = 0; q < 4; q++) product_lo(ha + (size_t)q * L * (F2 / 4), L, F2 / 4, WP(wmd, (size_t)q * (F2 / 4) * D), D, yb, q ? (rp && q == 3 ? OUT_ADDRESG : OUT_ADD) : OUT_ACT, 128,
                                                     la[7] ? la[7] + (size_t)q * (F2 / 4) * lb[7][-64] : 0, lb[7], &th);
               if (!rp) residual(x, yb, L, mv[1][2]); });
k2done:
    #undef STAGE
    eng_end(&eg);
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t_start);
    for (int i = 1; i <= 6; i++) times[i] = (uint32_t)tm[i];
    times[7] = (uint32_t)th; times[8] = (uint32_t)ttr; times[9] = (uint32_t)tsm; times[10] = (uint32_t)g_tav; times[11] = (uint32_t)g_tqk;
    if (timesLen > 12) times[12] = (uint32_t)g_tpost;   /* the waits for post jobs */
    if (timesLen > 23) times[22] = times[23] = 0;   /* the core and HMX clocks are not read in this build: these two slots stay 0 */
    g_tav = g_tqk = g_tpost = 0;
    return 0;
}

/* ------------------------------------------------------------------ YuE2's NAR (flow matching) layer: a Qwen3 layer whose queries are
 * the N latent rows and whose keys are the AR prefix (P rows, from the AR pass's KV cache, already normed and roped) followed by
 * the N rows' own. D 2048, 16 q heads / 8 kv heads of 128, SwiGLU 6144, RMSNorm with plain weights, RoPE in the split-half
 * (NEOX) form, no mask but the padding. x [N x D] in place in xo.
 *   inp: x [N x D] act | rope [N x 128] (cos | sin) | prefix K, 8 heads x [P x 128] (head layout) | prefix V, 8 heads (V head layout, M = P)
 *   prm: pre_w [D] | post_w [D] | q_norm [128] | k_norm [128]
 *   wts: wq, wk, wv, wo, mlp gate, up, down as 4 K quarters (kconv-style layouts; q8 / q4 by flags as the others)
 *   masked keys: [ka, kb) and [ka2, kb2) (the paddings of the prefix and of the rows) */
static int g_ykeep; static const uint16_t *g_ypk;   /* kblock_yrunl: the engine held over the layers, the prefix apart from the RoPE */
int kblock_yrun(remote_handle64 h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen,
                uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int prefix, int ka, int kb, int ka2, int kb2, int inplace,
                uint32_t *times, int timesLen)
{
    g_smj = 4;
    g_D = 2048; g_NH = 16; g_wfmt = (inplace >> 8) & 3;
    if (g_wfmt > 2) return AEE_EBADPARM;   /* 0: fp16, 1: q8, 2: q4; 3 is no format */
    const uint32_t N = tokens, P = prefix, L = N + P, KV = 1024, F = 6144, KH = KV / DH;
    const size_t nW = (size_t)D * D * 2 + (size_t)D * KV * 2 + (size_t)D * F * 3;
    const int nox = (inplace >> 28) & 1;                 /* bit 28: inp has no x, the stream is in xo already (with 28 layers its room is GBs of the DSP's 32-bit space) */
    const size_t nI = (nox ? 0 : (size_t)N * D) + (size_t)N * 128 + (size_t)P * KV * 2;
    const size_t nS = (size_t)N * D * 3 + (size_t)N * KV * 2 + (size_t)L * KV * 2 + (size_t)N * F;
    if ((size_t)wtsLen * 2 < wbytes(nW) || (size_t)inpLen < nI || prmLen < (int)(2 * D + 2 * DH) || (size_t)scrLen < nS || (size_t)xoLen < (size_t)N * D ||
        timesLen < 12 || N % 32 || P % 32 || !N) return AEE_EBADPARM;
    uint64_t t_start = HAP_perf_get_time_us();
    eng_t eg; int err = g_ykeep ? 0 : eng_begin(&eg); if (err) return err;
    const uint16_t *wq = wts, *wk = WP(wq, (size_t)D * D), *wv = WP(wk, (size_t)D * KV), *wo = WP(wv, (size_t)D * KV),
                   *wmg = WP(wo, (size_t)D * D), *wmu = WP(wmg, (size_t)D * F), *wmd = WP(wmu, (size_t)D * F);
    const uint16_t *rope = nox ? inp : inp + (size_t)N * D, *pk = g_ypk ? g_ypk : rope + (size_t)N * 128, *pv = pk + (size_t)P * KV;
    uint16_t *x = xo, *qb = scr, *ob = qb + (size_t)N * D, *yb = ob + (size_t)N * D, *kn = yb + (size_t)N * D, *vn = kn + (size_t)N * KV,
             *kf = vn + (size_t)N * KV, *vf = kf + (size_t)L * KV, *ha = vf + (size_t)L * KV;
    if (!(inplace & 1)) hvx_copy((uint8_t *)x, (const uint8_t *)inp, N * D * 2);
    /* the norms' column vectors: no shift, the weight as the scale; the residuals ungated */
    static HVX_Vector zero_v[MAXD / 32], w_v[2][MAXD / 32], one_v[MAXD / 32];
    for (int m = 0; m < 2; m++)
        for (uint32_t cb = 0; cb < D / 32; cb++)
        {
            float z[32], w[32], o[32];
            for (int c = 0; c < 32; c++) { z[c] = 0.f; w[c] = prm[(size_t)m * D + 32 * cb + c]; o[c] = 1.f; }
            w_v[m][cb] = colvec(w); if (!m) { zero_v[cb] = colvec(z); one_v[cb] = colvec(o); }
        }
    const float *qn = prm + 2 * D, *knw = qn + DH;
    uint64_t tm[8] = { 0 }, th = 0, ttr = 0, tsm = 0;
    #define STAGE(i, ...) do { uint64_t s_ = HAP_perf_get_time_us(); __VA_ARGS__; tm[i] += HAP_perf_get_time_us() - s_; } while (0)
    ln_arg_t lf[2] = { { x, 0, zero_v, w_v[0], 1 }, { x, 0, zero_v, w_v[1], 1 } };
    const float QS = 1.44269504f / sqrtf((float)DH);
    STAGE(1, { wset_t w3[3] = { { wq, D, qb, OUT_HEADQ }, { wk, KV, kn, OUT_HEADQ }, { wv, KV, vn, OUT_HEADV } };
               product_multi(0, ln_fill, &lf[0], N, D, w3, 3, 128, 0, &th); });
    STAGE(2, { rms_rope(qb, N, qn, rope, 1, QS, 16); rms_rope(kn, N, knw, rope, 1, 1.f, KH); });
    /* the keys and values: per kv head the prefix's rows, then the N new ones (V: in each of its four column sections) */
    STAGE(7, {
        for (uint32_t hd = 0; hd < KH; hd++)
        {
            if (P) hvx_copy((uint8_t *)(kf + (size_t)hd * L * DH), (const uint8_t *)(pk + (size_t)hd * P * DH), P * DH * 2);
            hvx_copy((uint8_t *)(kf + (size_t)hd * L * DH + (size_t)P * DH), (const uint8_t *)(kn + (size_t)hd * N * DH), N * DH * 2);
            for (int sct = 0; sct < 4; sct++)
            {
                const uint32_t a = sct / 2, b = sct % 2;
                uint16_t *d = vf + (size_t)hd * L * DH + (size_t)a * L * 64 + (size_t)b * (L / 32) * BH;
                if (P) hvx_copy((uint8_t *)d, (const uint8_t *)(pv + (size_t)hd * P * DH + (size_t)a * P * 64 + (size_t)b * (P / 32) * BH), (P / 32) * BH * 2);
                hvx_copy((uint8_t *)(d + (size_t)(P / 32) * BH), (const uint8_t *)(vn + (size_t)hd * N * DH + (size_t)a * N * 64 + (size_t)b * (N / 32) * BH), (N / 32) * BH * 2);
            }
        }
    });
    uint32_t mk[4] = { (uint32_t)ka, (uint32_t)kb, (uint32_t)ka2, (uint32_t)kb2 };
    /* bit 24: causal (YuE2's AR prefill): query row r sees keys 0..P + r; the second masked range is then the attention's own */
    g_causal = (inplace >> 24) & 1 ? (int)P : -1;
    STAGE(3, attention(qb, N, kf, vf, L, mk, 2, ob, &ttr, &tsm));
    g_causal = -1;
    STAGE(4, { product(ob, N, D, wo, D, yb, 128, 0, &th); residual(x, yb, N, one_v); });
    STAGE(5, { wset_t w2[2] = { { wmg, F, ha, OUT_SPLIT4 }, { wmu, F, ha, OUT_SPLIT4 } }; product_multi(0, ln_fill, &lf[1], N, D, w2, 2, 128, 2, &th); });
    STAGE(6, { for (int q = 0; q < 4; q++) product_to(ha + (size_t)q * N * (F / 4), N, F / 4, WP(wmd, (size_t)q * (F / 4) * D), D, yb, q ? OUT_ADD : OUT_ACT, 128, 0, &th);
               residual(x, yb, N, one_v); });
    #undef STAGE
    if (!g_ykeep) eng_end(&eg);
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t_start);
    for (int i = 1; i <= 7; i++) times[i] = (uint32_t)tm[i];
    times[8] = (uint32_t)th; times[9] = (uint32_t)ttr; times[10] = (uint32_t)tsm; times[11] = 0;
    return 0;
}
/* all the layers of one AR token in one call. wts, prm, pre: the layers' side by side; kvo: each layer's new K and V; pre2 (when long enough): each layer's keys so far, the next group's prefix */
int kblock_yrunl(remote_handle64 h, const uint16_t *wts, int wtsLen, const uint16_t *rope, int ropeLen, const uint16_t *pre, int preLen, const float *prm, int prmLen, uint16_t *scr, int scrLen,
                 uint16_t *xo, int xoLen, uint16_t *kvo, int kvoLen, uint16_t *pre2, int pre2Len, int tokens, int prefix, int nl, int inplace, uint32_t *times, int timesLen)
{
    const uint32_t N = tokens, P = prefix, L = N + P; const size_t KV2 = (size_t)1024 * 2, PL = (size_t)P * KV2, nW = (size_t)wtsLen / nl, nPr = (size_t)prmLen / nl, nKV = (size_t)N * KV2, nL = (size_t)L * KV2;
    if (nl < 1 || (size_t)kvoLen < nKV * nl || (size_t)preLen < PL * nl || (size_t)ropeLen < (size_t)N * 128 || timesLen < 12) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_ykeep = 1; const size_t nD = (size_t)N * 2048; uint32_t sum[12] = { 0 }; const uint64_t t0 = HAP_perf_get_time_us();
    for (int l = 0; l < nl && !err; l++)
    {
        g_ypk = pre + (size_t)l * PL; uint32_t tm[12];
        err = kblock_yrun(h, wts + (size_t)l * nW, (int)nW, rope, ropeLen + (int)PL, prm + (size_t)l * nPr, (int)nPr, scr, scrLen, xo, xoLen, tokens, prefix, 0, 0, 0, 0, inplace, tm, 12);
        if (err) break;
        const uint16_t *kn = scr + nD * 3, *kf = kn + nKV;
        hvx_copy((uint8_t *)(kvo + (size_t)l * nKV), (const uint8_t *)kn, nKV * 2);
        if ((size_t)pre2Len >= nL * nl) hvx_copy((uint8_t *)(pre2 + (size_t)l * nL), (const uint8_t *)kf, nL * 2);
        for (int i = 1; i < 12; i++) sum[i] += tm[i];
    }
    g_ykeep = 0; g_ypk = 0; eng_end(&eg);
    for (int i = 1; i < 12; i++) times[i] = sum[i];
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t0);
    return err;
}

/* Products alone, for an LLM run piece by piece with its norms, recurrences and attention on the host (MOSS-Image's Qwen3.5 text
 * encoder). x [m x k] and y in the activation layout, weights W^T in kconv's layout and format (flags bits 8-9).
 *   flags & 15 = 0: y [m x n] = x W, W^T [k x n]
 *   flags & 15 = 1: a SwiGLU MLP, w = gate^T [k x n] | up^T [k x n] | down^T as four [n/4 x k] quarters (Krea's order):
 *                   y [m x k] = (silu(x gate) * (x up)) down, the hidden [m x n] in scr (four quarters, as Krea's) */
int kblock_lin(remote_handle64 h, const uint16_t *w, int wLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen,
               int m, int k, int n, int flags)
{
    const int mode = flags & 15; const uint32_t M = (uint32_t)m, K = (uint32_t)k, N = (uint32_t)n;
    g_wfmt = (flags >> 8) & 3; g_D = K <= MAXD ? K : MAXD; g_NH = 16;
    if (g_wfmt > 2) return AEE_EBADPARM;   /* 0: fp16, 1: q8, 2: q4; 3 is no format */
    /* the products take K in chunks of KC: K % KC columns would be left out without a word (Krea's text encoder, K = 2432: 31 % off) */
    if (M % 32 || K % 32 || (mode == 1 && K % KC) || N % 64 || (size_t)xLen < (size_t)M * K || mode > 1) return AEE_EBADPARM;
    if (mode == 0 && ((size_t)wLen * 2 < wbytes((size_t)K * N) || (size_t)yLen < (size_t)M * N)) return AEE_EBADPARM;
    if (mode == 1 && ((size_t)wLen * 2 < wbytes((size_t)K * N * 3) || (size_t)yLen < (size_t)M * K || (size_t)scrLen < (size_t)M * N || N % 128)) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    uint64_t th = 0;
    if (mode == 0) { uint32_t kcb = K / 32 < KC / 32 ? K / 32 : KC / 32; while ((K / 32) % kcb) kcb--; g_kc = kcb * 32; product_lo(x, M, K, w, N, y, OUT_ACT, 128, 0, 0, &th); g_kc = 0; }
    else
    {
        const uint16_t *wg = w, *wu = WP(wg, (size_t)K * N), *wd = WP(wu, (size_t)K * N);
        /* a hidden size whose quarters are not whole chunks (Krea's text encoder, 9728): down in one piece, as the host lays it out */
        const int split = (N / 4) % KC == 0;
        wset_t w2[2] = { { wg, N, scr, split ? OUT_SPLIT4 : OUT_ACT, 0, 0, 0, 0, 0 }, { wu, N, scr, split ? OUT_SPLIT4 : OUT_ACT, 0, 0, 0, 0, 0 } };
        product_multi(x, 0, 0, M, K, w2, 2, 128, 2, &th);
        if (!split) product_lo(scr, M, N, wd, K, y, OUT_ACT, 128, 0, 0, &th);
        else for (int q = 0; q < 4; q++) product_lo(scr + (size_t)q * M * (N / 4), M, N / 4, WP(wd, (size_t)q * (N / 4) * K), K, y, q ? OUT_ADD : OUT_ACT, 128, 0, 0, &th);
    }
    eng_end(&eg); return 0;
}

int kblock_probe(remote_handle64 h, int mode, uint16_t *res, int resLen)
{
    if (resLen < 1024) return AEE_EBADPARM;
    if (mode & 0x40000000) { g_opt = mode & 0xFFFFFF; return 0; }   /* the optimisation switches (OPT_*) for this session */
    if (mode == 211)
    {   /* the chip's account of itself for a bug report (kxl_host.inc nHwInfo), before and without needing the engine's resources:
           int [0] VTCM total, [1] available (bytes), [2] the VTCM the engine asks for, [3] HVX threads the engine uses, [4] HMX lock4,
           [5] the HMX clock vote, [6] eng_begin's answer (0, or why the engine could not start here: 0x4000 | MB for too little VTCM),
           [7] the fp16 HMX one-tile self-test's relative error x 1e6 (-1 not run), [8] __HEXAGON_ARCH__ */
        int32_t *o = (int32_t *)res; memset(res, 0, 64); o[7] = -1; o[8] = __HEXAGON_ARCH__;
        unsigned int tot = 0, av = 0, cap = 0; compute_res_vtcm_page_t tl, al;
        if (HAP_compute_res_query_VTCM(0, &tot, &tl, &av, &al) == 0) { o[0] = (int32_t)tot; o[1] = (int32_t)av; }
        o[4] = HAP_compute_res_query_capability(HAP_COMPUTE_RES_PREEMPTION_CAPABILITY, &cap) == 0 && (cap & HAP_COMPUTE_RES_HMX_LOCK4);
        eng_t eg; const int err = eng_begin(&eg); o[2] = (int32_t)VTCM_BYTES; o[3] = g_nw; o[5] = g_hmxclk; o[6] = err;
        if (!err)
        {   /* a [32 x 32] x [32 x 32] fp16 tile through the engine's HMX product (hmx_mpy_chunks), against the exact answer */
            uint8_t *act = g_vtcm, *wgt = act + 8192, *out = wgt + 8192; __fp16 *a = (__fp16 *)act, *w = (__fp16 *)wgt, *y = (__fp16 *)out;
            for (int r = 0; r < 32; r++) for (int k = 0; k < 32; k++) a[64 * (r / 2) + 2 * k + (r & 1)] = (__fp16)(((3 * r + 5 * k) % 9 - 4) / 2.f);
            for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) w[32 * (k & ~1) + 2 * c + (k & 1)] = (__fp16)((7 * k + 3 * c) % 5 - 2);
            memset(out, 0, 4096); hmx_mpy_chunks(act, 0x87C, wgt, 0x7FF, out, 0, g_bias, 2048, 1);
            double se = 0, sr = 0;
            for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++)
            {
                float want = 0; for (int k = 0; k < 32; k++) want += (((3 * r + 5 * k) % 9 - 4) / 2.f) * (float)((7 * k + 3 * c) % 5 - 2);
                const float got = (float)y[64 * (r / 2) + 2 * c + (r & 1)]; se += (double)(got - want) * (got - want); sr += (double)want * want;
            }
            o[7] = (int32_t)(sqrt(se / (sr + 1e-30)) * 1e6 > 2e9 ? 2e9 : sqrt(se / (sr + 1e-30)) * 1e6);
            eng_end(&eg);
        }
        return 0;
    }
    return AEE_EBADPARM;
}

#include "kvae.inc"
#include "kxl.inc"
#include "kaud.inc"
