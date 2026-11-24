/* kconv.c -- on the phone: a checkpoint (safetensors or GGUF) of Anima (Cosmos DiT), Krea 2, MOSS-Image, FLUX.2 Klein or SDXL to the
 * engine's files, so a user can swap in another checkpoint of the same architecture without a PC. The model is told from the tensor
 * names. SDXL is written in f16 only (the others: f16, q8 or q4).
 * Anima:
 *   per block i:  weights_b<i>.f16  the 10 matrices W^T in the HMX weight layout, in the order kblock_imp.c reads them
 *                 norms_b<i>.f32    q/k RMS-norm weights of self- and cross-attention, [4][128]
 * HMX weight layout of W^T [K x N]: per 64-column group its two 32-column halves, each the K/32 row blocks of 32 x 32;
 * inside a block element (r, c) at 32 (r & ~1) + 2c + (r & 1).
 *   kconv <model.safetensors | model.gguf> <out dir> [threads] [f16 | q8 | q4] [bundle dir]
 * q8 / q4: the block matrices in the engine's quantized formats (kblock_imp.c g_wfmt): per matrix its values in the layout's
 * element order (8-bit integers, or two 4-bit values a byte: element i and 128 + i of each 256), then per 32 x 32 block its 32 column
 * scales (fp16). The grouping and the rounding are ggml's Q8_0 / Q4_0, so a GGUF file of those types comes through unchanged.
 * bundle dir: for SDXL, the folder with the text encoders' ONNX files and clip_map.txt (KX_TE in the environment overrides it).
 * format.txt in the out dir says which (none: f16). A GGUF input (F32, F16, BF16, Q8_0, Q4_0 tensors) works like a safetensors one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

static int g_fd;
static char *g_hdr;
static uint64_t g_data0;
static const char *g_pre = "";
static int g_q = 0;                                   /* output format: 0 f16, 1 q8, 2 q4 */
static const char *g_bundle = "";                        /* SDXL: the folder with the text encoders (clip_map.txt, ONNX files), "" none */

typedef struct { int dtype; int64_t shape[4]; int nd; uint64_t off, end; } tinfo_t;   /* dtype: 0 bf16, 1 f16, 2 f32, 3 q8_0, 4 q4_0, 5 f8_e4m3, 6 f8_e5m2 */

/* ---------------------------------------------------------------- GGUF: the tensor table read once, names looked up in it */
typedef struct { char *name; tinfo_t t; } gt_t;
static gt_t *g_gt; static int g_ngt, g_gguf;
static int rd(void *p, size_t n, uint64_t *pos) { if (pread(g_fd, p, n, (off_t)*pos) != (ssize_t)n) return -1; *pos += n; return 0; }
static char *rdstr(uint64_t *pos) { uint64_t n; if (rd(&n, 8, pos) || n > (1u << 20)) return 0; char *s = malloc(n + 1); if (rd(s, n, pos)) { free(s); return 0; } s[n] = 0; return s; }
static int skipval(uint32_t ty, uint64_t *pos)
{
    static const int sz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    if (ty == 8) { char *s = rdstr(pos); if (!s) return -1; free(s); return 0; }
    if (ty == 9) { uint32_t et; uint64_t n; if (rd(&et, 4, pos) || rd(&n, 8, pos)) return -1;
                   if (et != 8 && et != 9 && et < 13) { *pos += n * sz[et]; return 0; } for (uint64_t i = 0; i < n; i++) if (skipval(et, pos)) return -1; return 0; }
    if (ty < 13) { *pos += sz[ty]; return 0; }
    return -1;
}
static int gguf_open(void)
{
    uint64_t pos = 4, nt, nkv, align = 32; uint32_t ver;
    if (rd(&ver, 4, &pos) || rd(&nt, 8, &pos) || rd(&nkv, 8, &pos) || nt > 100000) return -1;
    for (uint64_t i = 0; i < nkv; i++)
    {
        char *k = rdstr(&pos); uint32_t ty; if (!k || rd(&ty, 4, &pos)) return -1;
        if (!strcmp(k, "general.alignment") && ty == 4) { uint32_t a; if (rd(&a, 4, &pos)) return -1; align = a; } else if (skipval(ty, &pos)) return -1;
        free(k);
    }
    g_gt = calloc(nt, sizeof(gt_t)); g_ngt = (int)nt;
    for (uint64_t i = 0; i < nt; i++)
    {
        gt_t *e = &g_gt[i]; uint32_t nd, ty; uint64_t dims[4] = { 1, 1, 1, 1 }, off;
        if (!(e->name = rdstr(&pos)) || rd(&nd, 4, &pos) || nd > 4 || rd(dims, 8 * nd, &pos) || rd(&ty, 4, &pos) || rd(&off, 8, &pos)) return -1;
        e->t.dtype = ty == 30 ? 0 : ty == 1 ? 1 : ty == 0 ? 2 : ty == 8 ? 3 : ty == 2 ? 4 : -1 - (int)ty;
        e->t.nd = (int)nd; uint64_t n = 1;
        for (uint32_t d = 0; d < nd; d++) { e->t.shape[d] = (int64_t)dims[nd - 1 - d]; n *= dims[d]; }       /* torch order */
        e->t.off = off; e->t.end = off + (ty == 8 ? n / 32 * 34 : ty == 2 ? n / 32 * 18 : ty == 0 ? n * 4 : n * 2);
    }
    g_data0 = (pos + align - 1) / align * align; g_gguf = 1;
    return 0;
}

/* the header entry of a tensor: "<name>":{"dtype":"BF16","shape":[a,b],"data_offsets":[s,e]} */
static int find_tensor(const char *name, tinfo_t *t)
{
    if (g_gguf)
    {
        char full[512]; snprintf(full, sizeof(full), "%s%s", g_pre, name);
        for (int i = 0; i < g_ngt; i++) if (!strcmp(g_gt[i].name, full))
        {
            *t = g_gt[i].t;
            if (t->dtype < 0) { fprintf(stderr, "%s: GGUF type %d not read here (F32, F16, BF16, Q8_0, Q4_0 are)\n", full, -1 - t->dtype); exit(1); }
            return 0;
        }
        return -1;
    }
    char key[512]; snprintf(key, sizeof(key), "\"%s%s\"", g_pre, name);
    const char *p = strstr(g_hdr, key); if (!p) return -1;
    p += strlen(key);
    const char *d = strstr(p, "\"dtype\""), *s = strstr(p, "\"shape\""), *o = strstr(p, "\"data_offsets\"");
    if (!d || !s || !o) return -1;
    d = strchr(d + 7, '"') + 1;
    t->dtype = !strncmp(d, "BF16", 4) ? 0 : !strncmp(d, "F16", 3) ? 1 : !strncmp(d, "F32", 3) ? 2 : !strncmp(d, "F8_E4M3", 7) ? 5 : !strncmp(d, "F8_E5M2", 7) ? 6 : -1;
    if (t->dtype < 0) { fprintf(stderr, "%s: unsupported dtype %.6s\n", name, d); return -1; }
    s = strchr(s, '[') + 1; t->nd = 0;
    while (*s != ']' && t->nd < 4) { t->shape[t->nd++] = strtoll(s, (char **)&s, 10); while (*s == ',' || *s == ' ') s++; }
    o = strchr(o, '[') + 1; t->off = strtoull(o, (char **)&o, 10); while (*o == ',' || *o == ' ') o++; t->end = strtoull(o, 0, 10);
    return 0;
}

/* fp8 (an e4m3fn or e5m2 file, as ComfyUI's "fp8" and "fp8_scaled" checkpoints are) to fp32, exactly; a scaled file's per-tensor or
 * per-row multiplier taken from <name minus weight>scale_weight or weight_scale beside it */
static float *read_f32(const char *name, int64_t *rows, int64_t *cols);
static void fp8_f32(const uint8_t *r, float *f, size_t n, int e4m3, const char *name)
{
    float lut[256];
    for (int b = 0; b < 256; b++)
    {
        const int s = b >> 7, e = e4m3 ? (b >> 3) & 15 : (b >> 2) & 31, m = e4m3 ? b & 7 : b & 3, mb = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
        float v = e ? ldexpf(1.f + (float)m / (1 << mb), e - bias) : ldexpf((float)m / (1 << mb), 1 - bias);
        if (e4m3 ? (b & 127) == 127 : e == 31) v = e4m3 || m ? NAN : INFINITY;
        lut[b] = s ? -v : v;
    }
    for (size_t i = 0; i < n; i++) f[i] = lut[r[i]];
    const char *w = strstr(name, "weight"); if (!w || strcmp(w, "weight")) return;
    static const char *sn[2] = { "scale_weight", "weight_scale" };
    for (int k = 0; k < 2; k++)
    {
        char sname[512]; snprintf(sname, sizeof(sname), "%.*s%s", (int)(w - name), name, sn[k]); tinfo_t st;
        if (find_tensor(sname, &st)) continue;
        int64_t sr, sc; float *sv = read_f32(sname, &sr, &sc); const size_t ns = (size_t)(sr * sc);
        if (ns == 1) for (size_t i = 0; i < n; i++) f[i] *= sv[0];
        else { const size_t row = n / ns; for (size_t i = 0; i < n; i++) f[i] *= sv[i / row]; }
        free(sv); return;
    }
}

/* the tensor as fp32 (n values) */
static float *read_f32(const char *name, int64_t *rows, int64_t *cols)
{
    tinfo_t t; if (find_tensor(name, &t)) { fprintf(stderr, "missing %s%s\n", g_pre, name); exit(1); }
    size_t bytes = t.end - t.off, n = t.dtype == 3 ? bytes / 34 * 32 : t.dtype == 4 ? bytes / 18 * 32 : t.dtype >= 5 ? bytes : bytes / (t.dtype == 2 ? 4 : 2);
    uint8_t *raw = malloc(bytes); float *f = malloc(n * 4);
    if (pread(g_fd, raw, bytes, (off_t)(g_data0 + t.off)) != (ssize_t)bytes) { fprintf(stderr, "read %s failed\n", name); exit(1); }
    if (t.dtype == 2) memcpy(f, raw, bytes);
    else if (t.dtype == 3 || t.dtype == 4)
        for (size_t b = 0; b < n / 32; b++)
        {   /* ggml blocks: fp16 scale, then 32 8-bit integers (Q8_0) or 16 bytes of nibbles, low ones values 0-15, high ones 16-31, + 8 (Q4_0) */
            const uint8_t *r = raw + b * (t.dtype == 3 ? 34 : 18); __fp16 d; memcpy(&d, r, 2); float *o = f + b * 32;
            if (t.dtype == 3) for (int k = 0; k < 32; k++) o[k] = (float)d * (float)(int8_t)r[2 + k];
            else for (int k = 0; k < 16; k++) { o[k] = (float)d * (float)((r[2 + k] & 15) - 8); o[k + 16] = (float)d * (float)((r[2 + k] >> 4) - 8); }
        }
    else if (t.dtype == 0) { const uint16_t *h = (const uint16_t *)raw; for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)h[i] << 16; memcpy(&f[i], &u, 4); } }
    else if (t.dtype >= 5) fp8_f32(raw, f, n, t.dtype == 5, name);
    else { const __fp16 *h = (const __fp16 *)raw; for (size_t i = 0; i < n; i++) f[i] = (float)h[i]; }
    free(raw);
    *rows = t.nd > 1 ? t.shape[0] : 1; *cols = t.nd > 1 ? t.shape[1] : t.shape[0];
    return f;
}

/* torch Linear weight W [N out x K in] -> W^T [K x N] in the HMX weight layout, fp16 */
typedef struct { const float *w; __fp16 *o; int64_t K, N, g0, g1; } lay_t;
static void *lay_part(void *a_)
{
    lay_t *a = (lay_t *)a_; const int64_t K = a->K, kb = K / 32;
    for (int64_t g = a->g0; g < a->g1; g++)
        for (int h = 0; h < 2; h++)
            for (int c = 0; c < 32; c++)
            {
                const float *src = a->w + (g * 64 + h * 32 + c) * K;                     /* row n of W: column n of W^T */
                __fp16 *dst = a->o + (g * 2 + h) * kb * 1024 + 2 * c;
                for (int64_t k = 0; k < K; k++) dst[(k >> 5) * 1024 + 32 * ((k & 31) & ~1) + ((k & 31) & 1)] = (__fp16)src[k];
            }
    return 0;
}
static void layout(const float *w, __fp16 *o, int64_t N, int64_t K, int threads)
{
    pthread_t th[16]; lay_t a[16]; int64_t ng = N / 64;
    for (int t = 0; t < threads; t++) { a[t] = (lay_t){ w, o, K, N, ng * t / threads, ng * (t + 1) / threads }; pthread_create(&th[t], 0, lay_part, &a[t]); }
    for (int t = 0; t < threads; t++) pthread_join(th[t], 0);
}

/* q8 / q4 blocks b0..b1-1 of o (put's loop, one thread's share) */
typedef struct { const __fp16 *o; uint8_t *data; __fp16 *sc; size_t b0, b1; } qb_t;
static void *qb_part(void *a_)
{
    qb_t *A = (qb_t *)a_; const __fp16 *o = A->o; uint8_t *data = A->data; __fp16 *sc = A->sc; int8_t q[1024];
    for (size_t b = A->b0; b < A->b1; b++)
    {
        const __fp16 *blk = o + b * 1024;
        for (int c = 0; c < 32; c++)
        {
            /* column c of the block: rows r at 32 (r & ~1) + 2c + (r & 1) -- ggml's Q8_0 (amax / 127, rounded) or Q4_0 (the signed max / -8) */
            float amax = 0, mx = 0;
            for (int r = 0; r < 32; r++) { float v = (float)blk[32 * (r & ~1) + 2 * c + (r & 1)]; if (fabsf(v) > amax) { amax = fabsf(v); mx = v; } }
            float d = g_q == 1 ? amax / 127.f : mx / -8.f; __fp16 dh = (__fp16)d; float id = (float)dh != 0.f ? 1.f / (float)dh : 0.f;
            sc[b * 32 + c] = dh;
            for (int r = 0; r < 32; r++)
            {
                const int e = 32 * (r & ~1) + 2 * c + (r & 1); float v = (float)blk[e] * id;
                if (g_q == 1) q[e] = (int8_t)(v > 127.f ? 127 : v < -127.f ? -127 : lrintf(v));
                else { int u = (int)(v + 8.5f); q[e] = (int8_t)(u > 15 ? 15 : u < 0 ? 0 : u); }
            }
        }
        if (g_q == 1) memcpy(data + b * 1024, q, 1024);
        else for (int k = 0; k < 4; k++) for (int i = 0; i < 128; i++) data[b * 512 + k * 128 + i] = (uint8_t)(q[k * 256 + i] | q[k * 256 + 128 + i] << 4);
    }
    return 0;
}
/* one block matrix W [N x K] out to f: W^T in the layout, as fp16 or in the quantized format g_q */
static void put(FILE *f, const float *w, int64_t N, int64_t K, int threads)
{
    __fp16 *o = malloc((size_t)N * K * 2); layout(w, o, N, K, threads);
    if (!g_q) { fwrite(o, 2, (size_t)N * K, f); free(o); return; }
    const size_t nb = (size_t)N * K / 1024; uint8_t *data = malloc(g_q == 1 ? nb * 1024 : nb * 512); __fp16 *sc = malloc(nb * 32 * 2);
    /* the blocks shared by the threads (each independent: the same bytes as one thread) */
    pthread_t th[64]; qb_t qa[64]; if (threads > 64) threads = 64; if (threads < 1) threads = 1;
    for (int t = 0; t < threads; t++) { qa[t] = (qb_t){ o, data, sc, nb * t / threads, nb * (t + 1) / threads }; pthread_create(&th[t], 0, qb_part, &qa[t]); }
    for (int t = 0; t < threads; t++) pthread_join(th[t], 0);
    fwrite(data, 1, g_q == 1 ? nb * 1024 : nb * 512, f); fwrite(sc, 2, nb * 32, f);
    free(data); free(sc); free(o);
}

/* Krea 2: q / k rows of each head permuted to [even dims | odd dims] (the engine's split-half RoPE; q.k unchanged) */
static void headperm(float *w, int64_t heads, int64_t K)
{
    float *t = malloc((size_t)128 * K * 4);
    for (int64_t h = 0; h < heads; h++)
    {
        float *hw = w + (size_t)h * 128 * K;
        for (int d = 0; d < 128; d++) memcpy(t + (size_t)d * K, hw + (size_t)(d < 64 ? 2 * d : 2 * (d - 64) + 1) * K, K * 4);
        memcpy(hw, t, (size_t)128 * K * 4);
    }
    free(t);
}
static int krea(const char *out, int threads)
{
    int nb = 0; for (;; nb++) { char n[96]; tinfo_t t; snprintf(n, sizeof(n), "blocks.%d.attn.wq.weight", nb); if (find_tensor(n, &t)) break; }
    printf("Krea 2: %d blocks\n", nb);
    static const char *mats[8] = { "attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down" };
    struct timeval a, b; gettimeofday(&a, 0);
    for (int i = 0; i < nb; i++)
    {
        char p[1024]; snprintf(p, sizeof(p), "%s/weights_b%d.f16.part", out, i); FILE *f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
        for (int m = 0; m < 8; m++)
        {
            char n[128]; snprintf(n, sizeof(n), "blocks.%d.%s.weight", i, mats[m]); int64_t N, K; float *w = read_f32(n, &N, &K);
            if (m < 2) headperm(w, N / 128, K);
            /* mlp.down: its K = 16384 as four quarters, each its own weight matrix (the engine sums four K = 4096 products) */
            const int parts = m == 7 ? 4 : 1; const int64_t Kq = K / parts; float *wq = parts > 1 ? malloc((size_t)N * Kq * 4) : w;
            for (int q = 0; q < parts; q++)
            {
                if (parts > 1) for (int64_t n = 0; n < N; n++) memcpy(wq + n * Kq, w + n * K + q * Kq, Kq * 4);
                put(f, wq, N, Kq, threads);
            }
            if (parts > 1) free(wq);
            free(w);
        }
        fclose(f); char q[1024]; snprintf(q, sizeof(q), "%s/weights_b%d.f16", out, i); rename(p, q);
        /* the per-block constants: mod [6D] | prenorm + 1 | postnorm + 1 | qnorm + 1 (permuted) | knorm + 1 (permuted); tvec is added per step */
        snprintf(p, sizeof(p), "%s/params_b%d.f32", out, i); f = fopen(p, "wb");
        static const char *pn[5] = { "mod.lin", "prenorm.scale", "postnorm.scale", "attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale" };
        for (int m = 0; m < 5; m++)
        {
            char n[128]; snprintf(n, sizeof(n), "blocks.%d.%s", i, pn[m]); int64_t r, c; float *w = read_f32(n, &r, &c); size_t cnt = (size_t)(r * c);
            if (m) for (size_t k = 0; k < cnt; k++) w[k] += 1.f;
            if (m >= 3) { float t2[128]; for (int d = 0; d < 128; d++) t2[d] = w[d < 64 ? 2 * d : 2 * (d - 64) + 1]; memcpy(w, t2, sizeof(t2)); }
            fwrite(w, 4, cnt, f); free(w);
        }
        fclose(f);
        gettimeofday(&b, 0); printf("block %d done, %.1f s\n", i, (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1e6); fflush(stdout);
    }
    return 0;
}

/* MOSS-Image (MOSSImageTransformer): its 42 blocks for the engine's Krea path in MOSS mode (moss.txt marks the folder). Per block
 * weights wq, wk, wv (proj.qkv's three thirds), wo, mlp gate, up (ffn_up_gate's second and first halves: hidden, gate = chunk(2)), down
 * as four K = 2048 quarters; no head permutation (its RoPE is already split-half). params_b<i>.f32: zeros [6D] (the modulation is shared,
 * the app sends it each step) | norm1 | norm2 | q_norm | k_norm, weights as they are. Everything outside the blocks (embedders, the text
 * refiner, time embedding, modulations, the head) to moss_aux.f16 (fp16 row-major) with moss_aux.json (name -> shape, byte offset). */
static int moss(const char *out, int threads)
{
    int nb = 0; for (;; nb++) { char n[96]; tinfo_t t; snprintf(n, sizeof(n), "layers.%d.proj.qkv.weight", nb); if (find_tensor(n, &t)) break; }
    printf("MOSS-Image: prefix '%s', %d blocks\n", g_pre, nb);
    { char p[1024]; snprintf(p, sizeof(p), "%s/moss.txt", out); FILE *f = fopen(p, "w"); if (f) { fprintf(f, "moss %d\n", nb); fclose(f); } }
    struct timeval a, b; gettimeofday(&a, 0);
    {
        static const char *aux[] = { "video_embedder.", "text_embedder.", "time_embedding.", "shared_block_modulation.", "video_head_modulation.",
                                     "final_norm_video.", "final_linear_video." };
        char p[1024]; snprintf(p, sizeof(p), "%s/moss_aux.f16", out); FILE *f = fopen(p, "wb");
        snprintf(p, sizeof(p), "%s/moss_aux.json", out); FILE *j = fopen(p, "w"); if (!f || !j) { perror(p); return 1; }
        fprintf(j, "{\"tensors\": {"); uint64_t off = 0; int first = 1;
        /* every tensor of the header whose name (after the prefix) starts with one of those */
        const size_t pl = strlen(g_pre);
        for (const char *q = g_hdr; (q = strchr(q, '"')); )
        {
            const char *e = strchr(q + 1, '"'); if (!e) break;
            char name[256]; size_t nl = (size_t)(e - q - 1); if (nl >= sizeof(name)) { q = e + 1; continue; }
            memcpy(name, q + 1, nl); name[nl] = 0; q = e + 1;
            if (*q != ':' || strncmp(name, g_pre, pl)) continue;
            const char *nm = name + pl; int want = 0; for (size_t k = 0; k < sizeof(aux) / sizeof(aux[0]); k++) want |= !strncmp(nm, aux[k], strlen(aux[k]));
            if (!want) continue;
            tinfo_t t; if (find_tensor(nm, &t)) continue;
            int64_t r, c; float *w = read_f32(nm, &r, &c); size_t n = (size_t)r * c;
            __fp16 *h = malloc(n * 2); for (size_t k = 0; k < n; k++) h[k] = (__fp16)w[k]; fwrite(h, 2, n, f); free(h); free(w);
            if (t.nd > 1) fprintf(j, "%s\"%s\": {\"shape\": [%lld, %lld], \"offset\": %llu}", first ? "" : ", ", nm, (long long)r, (long long)c, (unsigned long long)off);
            else fprintf(j, "%s\"%s\": {\"shape\": [%lld], \"offset\": %llu}", first ? "" : ", ", nm, (long long)c, (unsigned long long)off);
            first = 0; off += n * 2;
        }
        fprintf(j, "}, \"blocks\": %d}", nb); fclose(j); fclose(f);
    }
    for (int i = 0; i < nb; i++)
    {
        char p[1024]; snprintf(p, sizeof(p), "%s/weights_b%d.f16.part", out, i); FILE *f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
        char n[128]; int64_t N, K;
        snprintf(n, sizeof(n), "layers.%d.proj.qkv.weight", i); float *w = read_f32(n, &N, &K);
        for (int m = 0; m < 3; m++) put(f, w + (size_t)m * (N / 3) * K, N / 3, K, threads);
        free(w);
        snprintf(n, sizeof(n), "layers.%d.proj.o.weight", i); w = read_f32(n, &N, &K); put(f, w, N, K, threads); free(w);
        snprintf(n, sizeof(n), "layers.%d.proj.ffn_up_gate.weight", i); w = read_f32(n, &N, &K);
        put(f, w + (size_t)(N / 2) * K, N / 2, K, threads); put(f, w, N / 2, K, threads); free(w);          /* gate, then up (the hidden) */
        snprintf(n, sizeof(n), "layers.%d.proj.ffn_down.weight", i); w = read_f32(n, &N, &K);
        { const int64_t Kq = K / 4; float *wq = malloc((size_t)N * Kq * 4);
          for (int q = 0; q < 4; q++) { for (int64_t r = 0; r < N; r++) memcpy(wq + r * Kq, w + r * K + q * Kq, Kq * 4); put(f, wq, N, Kq, threads); }
          free(wq); }
        free(w); fclose(f);
        char q[1024]; snprintf(q, sizeof(q), "%s/weights_b%d.f16", out, i); rename(p, q);
        snprintf(p, sizeof(p), "%s/params_b%d.f32", out, i); f = fopen(p, "wb");
        int64_t D = 3072; float *z = calloc((size_t)6 * D, 4); fwrite(z, 4, (size_t)6 * D, f); free(z);
        static const char *pn[4] = { "norm1", "norm2", "q_norm", "k_norm" };
        for (int m = 0; m < 4; m++) { snprintf(n, sizeof(n), "layers.%d.%s.weight", i, pn[m]); int64_t r, c; w = read_f32(n, &r, &c); fwrite(w, 4, (size_t)(r * c), f); free(w); }
        fclose(f);
        gettimeofday(&b, 0); printf("block %d done, %.1f s\n", i, (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1e6); fflush(stdout);
    }
    return 0;
}

/* FLUX.2 Klein (Flux2Transformer2DModel, diffusers' names): its double blocks, then its single ones, for the engine's MOSS mode (klein.txt marks the
 * folder). weights_b<i>: per stream (double: the text, then the image) q, k (head-permuted: its RoPE is on adjacent pairs), v, out, gate, up, down
 * as three K = 3072 parts; a single block's out is the first 3072 columns of to_out, its down the rest. params_b<i>.f32: q, k norms (permuted),
 * per stream. Everything else to moss_aux.f16 / .json as MOSS's. */
static int g_kD = 3072;
static float *cols(const float *w, int64_t N, int64_t K, int64_t c0, int64_t nc)
{ float *o = malloc((size_t)N * nc * 4); for (int64_t r = 0; r < N; r++) memcpy(o + r * nc, w + r * K + c0, (size_t)nc * 4); return o; }
static void put_down(FILE *f, const float *w, int64_t N, int64_t K, int64_t c0, int threads)
{ for (int q = 0; q < 3; q++) { float *p = cols(w, N, K, c0 + q * g_kD, g_kD); put(f, p, N, g_kD, threads); free(p); } }
static void put_norm(FILE *f, const char *name)
{ int64_t r, c; float *w = read_f32(name, &r, &c); float t2[128]; for (int d = 0; d < 128; d++) t2[d] = w[d < 64 ? 2 * d : 2 * (d - 64) + 1]; fwrite(t2, 4, 128, f); free(w); }
static int klein(const char *out, int threads)
{
    int nd = 0, ns = 0; tinfo_t t; char n[160]; const int bfl = !find_tensor("single_blocks.0.linear1.weight", &t);      /* the original's names */
    for (;; nd++) { snprintf(n, sizeof(n), bfl ? "double_blocks.%d.img_attn.proj.weight" : "transformer_blocks.%d.attn.to_q.weight", nd); if (find_tensor(n, &t)) break; }
    for (;; ns++) { snprintf(n, sizeof(n), bfl ? "single_blocks.%d.linear2.weight" : "single_transformer_blocks.%d.attn.to_out.weight", ns); if (find_tensor(n, &t)) break; }
    { int64_t r0, c0; float *w0 = read_f32(bfl ? "img_in.weight" : "x_embedder.weight", &r0, &c0); g_kD = (int)r0; free(w0); }
    const int nb = nd + ns; printf("FLUX.2 Klein: prefix '%s', %d blocks\n", g_pre, nb);
    char p[1024]; snprintf(p, sizeof(p), "%s/klein.txt", out); FILE *f = fopen(p, "w"); if (f) { fprintf(f, "klein %d %d %d\n", nd, ns, g_kD); fclose(f); }
    {
        snprintf(p, sizeof(p), "%s/moss_aux.f16", out); f = fopen(p, "wb"); snprintf(p, sizeof(p), "%s/moss_aux.json", out); FILE *j = fopen(p, "w"); if (!f || !j) { perror(p); return 1; }
        static const char *aux[] = { "x_embedder.weight", "context_embedder.weight", "time_guidance_embed.timestep_embedder.linear_1.weight", "time_guidance_embed.timestep_embedder.linear_2.weight",
                                     "double_stream_modulation_img.linear.weight", "double_stream_modulation_txt.linear.weight", "single_stream_modulation.linear.weight", "norm_out.linear.weight", "proj_out.weight" };
        static const char *auxb[] = { "img_in.weight", "txt_in.weight", "time_in.in_layer.weight", "time_in.out_layer.weight", "double_stream_modulation_img.lin.weight", "double_stream_modulation_txt.lin.weight",
                                      "single_stream_modulation.lin.weight", "final_layer.adaLN_modulation.1.weight", "final_layer.linear.weight" };
        fprintf(j, "{\"tensors\": {"); unsigned long long off = 0;
        for (size_t k = 0; k < sizeof(aux) / sizeof(aux[0]); k++)
        {
            int64_t r, c; float *w = read_f32(bfl ? auxb[k] : aux[k], &r, &c); size_t cnt = (size_t)r * c;
            if (bfl && k == 7) { float *sw = malloc(cnt * 4); memcpy(sw, w + (size_t)(r / 2) * c, (size_t)(r / 2) * c * 4); memcpy(sw + (size_t)(r / 2) * c, w, (size_t)(r / 2) * c * 4); free(w); w = sw; }   /* its shift, scale; the release's scale, shift */
            __fp16 *h = malloc(cnt * 2); for (size_t q = 0; q < cnt; q++) h[q] = (__fp16)w[q]; fwrite(h, 2, cnt, f); free(h); free(w);
            fprintf(j, "%s\"%s\": {\"shape\": [%lld, %lld], \"offset\": %llu}", k ? ", " : "", aux[k], (long long)r, (long long)c, off); off += cnt * 2;
        }
        fprintf(j, "}, \"blocks\": %d}", nb); fclose(j); fclose(f);
    }
    struct timeval a, b; gettimeofday(&a, 0);
    for (int i = 0; i < nb; i++)
    {
        snprintf(p, sizeof(p), "%s/weights_b%d.f16.part", out, i); f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
        int64_t N, K; float *w;
        if (i < nd)
        {
            static const char *lin[2][7] = { { "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "ff_context.linear_in", 0, "ff_context.linear_out" },
                                             { "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.linear_in", 0, "ff.linear_out" } };
            static const char *st[2] = { "txt", "img" };
            for (int s = 0; bfl && s < 2; s++)
            {
                snprintf(n, sizeof(n), "double_blocks.%d.%s_attn.qkv.weight", i, st[s]); w = read_f32(n, &N, &K);
                for (int m = 0; m < 3; m++) { float *c = w + (size_t)m * g_kD * K; if (m < 2) headperm(c, g_kD / 128, K); put(f, c, g_kD, K, threads); } free(w);
                snprintf(n, sizeof(n), "double_blocks.%d.%s_attn.proj.weight", i, st[s]); w = read_f32(n, &N, &K); put(f, w, N, K, threads); free(w);
                snprintf(n, sizeof(n), "double_blocks.%d.%s_mlp.0.weight", i, st[s]); w = read_f32(n, &N, &K); put(f, w, N / 2, K, threads); put(f, w + (size_t)(N / 2) * K, N / 2, K, threads); free(w);
                snprintf(n, sizeof(n), "double_blocks.%d.%s_mlp.2.weight", i, st[s]); w = read_f32(n, &N, &K); put_down(f, w, N, K, 0, threads); free(w);
            }
            for (int s = 0; !bfl && s < 2; s++)
            {
                for (int m = 0; m < 4; m++) { snprintf(n, sizeof(n), "transformer_blocks.%d.%s.weight", i, lin[s][m]); w = read_f32(n, &N, &K); if (m < 2) headperm(w, N / 128, K); put(f, w, N, K, threads); free(w); }
                snprintf(n, sizeof(n), "transformer_blocks.%d.%s.weight", i, lin[s][4]); w = read_f32(n, &N, &K); put(f, w, N / 2, K, threads); put(f, w + (size_t)(N / 2) * K, N / 2, K, threads); free(w);
                snprintf(n, sizeof(n), "transformer_blocks.%d.%s.weight", i, lin[s][6]); w = read_f32(n, &N, &K); put_down(f, w, N, K, 0, threads); free(w);
            }
        }
        else
        {
            const int si = i - nd; int64_t No, Ko;
            snprintf(n, sizeof(n), bfl ? "single_blocks.%d.linear1.weight" : "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight", si); w = read_f32(n, &N, &K);
            for (int m = 0; m < 3; m++) { float *c = w + (size_t)m * g_kD * K; if (m < 2) headperm(c, g_kD / 128, K); put(f, c, g_kD, K, threads); }
            snprintf(n, sizeof(n), bfl ? "single_blocks.%d.linear2.weight" : "single_transformer_blocks.%d.attn.to_out.weight", si); float *wo = read_f32(n, &No, &Ko);
            { float *c = cols(wo, No, Ko, 0, g_kD); put(f, c, No, g_kD, threads); free(c); }
            put(f, w + (size_t)3 * g_kD * K, 3 * g_kD, K, threads); put(f, w + (size_t)6 * g_kD * K, 3 * g_kD, K, threads); free(w);
            put_down(f, wo, No, Ko, g_kD, threads); free(wo);
        }
        fclose(f); char q[1024]; snprintf(q, sizeof(q), "%s/weights_b%d.f16", out, i); snprintf(p, sizeof(p), "%s/weights_b%d.f16.part", out, i); rename(p, q);
        snprintf(p, sizeof(p), "%s/params_b%d.f32", out, i); f = fopen(p, "wb");
        if (bfl && i < nd) { static const char *pn[4] = { "txt_attn.norm.query_norm", "txt_attn.norm.key_norm", "img_attn.norm.query_norm", "img_attn.norm.key_norm" }; for (int m = 0; m < 4; m++) { snprintf(n, sizeof(n), "double_blocks.%d.%s.scale", i, pn[m]); put_norm(f, n); } }
        else if (bfl) { static const char *pn[2] = { "norm.query_norm", "norm.key_norm" }; for (int m = 0; m < 2; m++) { snprintf(n, sizeof(n), "single_blocks.%d.%s.scale", i - nd, pn[m]); put_norm(f, n); } }
        else if (i < nd) { static const char *pn[4] = { "norm_added_q", "norm_added_k", "norm_q", "norm_k" }; for (int m = 0; m < 4; m++) { snprintf(n, sizeof(n), "transformer_blocks.%d.attn.%s.weight", i, pn[m]); put_norm(f, n); } }
        else { static const char *pn[2] = { "norm_q", "norm_k" }; for (int m = 0; m < 2; m++) { snprintf(n, sizeof(n), "single_transformer_blocks.%d.attn.%s.weight", i - nd, pn[m]); put_norm(f, n); } }
        fclose(f);
        gettimeofday(&b, 0); printf("block %d done, %.1f s\n", i, (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1e6); fflush(stdout);
    }
    return 0;
}

#include "sdxl_conv.inc"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "kconv <model.safetensors | model.gguf> <out dir> [threads] [f16 | q8 | q4] [bundle dir]\n"); return 1; }
    int threads = argc > 3 ? atoi(argv[3]) : 4; if (threads < 1) threads = 1; if (threads > 16) threads = 16;
    g_q = 0; if (argc > 4) { if (!strcmp(argv[4], "q8")) g_q = 1; else if (!strcmp(argv[4], "q4")) g_q = 2; else if (strcmp(argv[4], "f16")) { fprintf(stderr, "unknown format %s (f16, q8 or q4)\n", argv[4]); return 1; } }
    if (argc > 5) g_bundle = argv[5];
    g_fd = open(argv[1], O_RDONLY); if (g_fd < 0) { perror(argv[1]); return 1; }
    char magic[4] = { 0 }; if (pread(g_fd, magic, 4, 0) != 4) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    if (!memcmp(magic, "GGUF", 4)) { if (gguf_open()) { fprintf(stderr, "a damaged GGUF file\n"); return 1; } g_hdr = (char *)""; }
    else
    {
        uint64_t hl = 0; if (pread(g_fd, &hl, 8, 0) != 8 || hl > (64u << 20)) { fprintf(stderr, "not a safetensors or GGUF file\n"); return 1; }
        g_hdr = malloc(hl + 1); if (pread(g_fd, g_hdr, hl, 8) != (ssize_t)hl) { fprintf(stderr, "header read failed\n"); return 1; }
        g_hdr[hl] = 0; g_data0 = 8 + hl;
    }
    {   /* which format the block files are in, for the app (no file: f16) */
        char p[1024]; snprintf(p, sizeof(p), "%s/format.txt", argv[2]); if (g_q) { FILE *f = fopen(p, "w"); if (f) { fputs(g_q == 1 ? "q8\n" : "q4\n", f); fclose(f); } } else unlink(p);
    }
    static const char *pres[] = { "model.diffusion_model.", "diffusion_model.", "net.", "" };
    tinfo_t t0; int found = 0;
    g_pre = ""; if (!find_tensor("blocks.0.attn.wq.weight", &t0)) return krea(argv[2], threads);
    for (int i = 0; i < 2; i++) { g_pre = pres[i]; if (!find_tensor("input_blocks.0.0.weight", &t0) && !find_tensor("label_emb.0.0.weight", &t0)) return sdxl(argv[2], threads); }
    for (int i = 0; i < 4; i++) { g_pre = pres[i]; if (!find_tensor("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", &t0) || !find_tensor("single_blocks.0.linear1.weight", &t0)) {
        return klein(argv[2], threads); } }
    for (int i = 0; i < 4; i++) { g_pre = pres[i]; if (!find_tensor("layers.0.proj.qkv.weight", &t0)) {
        return moss(argv[2], threads); } }
    for (int i = 0; i < 4 && !found; i++) { g_pre = pres[i]; found = !find_tensor("blocks.0.mlp.layer1.weight", &t0); }
    if (!found) { fprintf(stderr, "no Anima blocks in %s\n", argv[1]); return 1; }
    int nb = 0; for (;; nb++) { char n[96]; tinfo_t t; snprintf(n, sizeof(n), "blocks.%d.mlp.layer1.weight", nb); if (find_tensor(n, &t)) break; }
    printf("prefix '%s', %d blocks, dtype %s, out %s\n", g_pre, nb, t0.dtype == 0 ? "bf16" : t0.dtype == 1 ? "f16" : t0.dtype == 2 ? "f32" : t0.dtype == 3 ? "q8_0" : t0.dtype == 4 ? "q4_0" : t0.dtype == 5 ? "f8_e4m3" : "f8_e5m2", g_q == 1 ? "q8" : g_q == 2 ? "q4" : "f16");
    static const char *mats[10] = { "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                                    "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
                                    "mlp.layer1", "mlp.layer2" };
    static const char *norms[4] = { "self_attn.q_norm", "self_attn.k_norm", "cross_attn.q_norm", "cross_attn.k_norm" };
    struct timeval a, b; gettimeofday(&a, 0);
    {   /* the input layer, W_x^T / 16 zero-padded to 512 rows (the stream is carried / 16), and the output layer W_f^T */
        int64_t N, K; float *w = read_f32("x_embedder.proj.1.weight", &N, &K);
        float *wp = calloc((size_t)N * 512, 4); for (int64_t n = 0; n < N; n++) for (int64_t k = 0; k < K; k++) wp[n * 512 + k] = w[n * K + k] / 16.f;
        __fp16 *o = malloc((size_t)N * 512 * 2); layout(wp, o, N, 512, threads);
        char p[1024]; snprintf(p, sizeof(p), "%s/embed.f16", argv[2]); FILE *f = fopen(p, "wb"); fwrite(o, 2, (size_t)N * 512, f); fclose(f);
        free(o); free(wp); free(w);
        w = read_f32("final_layer.linear.weight", &N, &K);
        o = malloc((size_t)N * K * 2); layout(w, o, N, K, threads);
        snprintf(p, sizeof(p), "%s/final.f16", argv[2]); f = fopen(p, "wb"); fwrite(o, 2, (size_t)N * K, f); fclose(f);
        free(o); free(w);
    }
    {   /* the adaLN weights for the app's Adaln (adaln.f16: fp16 row-major one after another; adaln.json: name -> shape, byte offset) */
        char p[1024]; snprintf(p, sizeof(p), "%s/adaln.f16", argv[2]); FILE *f = fopen(p, "wb");
        snprintf(p, sizeof(p), "%s/adaln.json", argv[2]); FILE *j = fopen(p, "w");
        fprintf(j, "{\"tensors\": {"); uint64_t off = 0; int first = 1;
        static const char *kinds[3] = { "self_attn", "cross_attn", "mlp" };
        for (int e = -3; e < nb * 6 + 2; e++)
        {
            char src[128], dst[64];
            if (e == -3) { strcpy(src, "t_embedder.1.linear_1.weight"); strcpy(dst, "t.linear_1.weight"); }
            else if (e == -2) { strcpy(src, "t_embedder.1.linear_2.weight"); strcpy(dst, "t.linear_2.weight"); }
            else if (e == -1) { strcpy(src, "t_embedding_norm.weight"); strcpy(dst, "t.norm.weight"); }
            else if (e < nb * 6) { int i = e / 6, k = (e % 6) / 2, l = e % 2 + 1;
                snprintf(src, sizeof(src), "blocks.%d.adaln_modulation_%s.%d.weight", i, kinds[k], l); snprintf(dst, sizeof(dst), "b%d.%d.%d.weight", i, k, l); }
            else { int l = e - nb * 6 + 1; snprintf(src, sizeof(src), "final_layer.adaln_modulation.%d.weight", l); snprintf(dst, sizeof(dst), "final.%d.weight", l); }
            tinfo_t t; find_tensor(src, &t); int64_t r, c; float *w = read_f32(src, &r, &c); size_t n = (size_t)r * c;
            __fp16 *h = malloc(n * 2); for (size_t q = 0; q < n; q++) h[q] = (__fp16)w[q]; fwrite(h, 2, n, f); free(h); free(w);
            if (t.nd > 1) fprintf(j, "%s\"%s\": {\"shape\": [%lld, %lld], \"offset\": %llu}", first ? "" : ", ", dst, (long long)r, (long long)c, (unsigned long long)off);
            else fprintf(j, "%s\"%s\": {\"shape\": [%lld], \"offset\": %llu}", first ? "" : ", ", dst, (long long)c, (unsigned long long)off);
            first = 0; off += n * 2;
        }
        fprintf(j, "}, \"t_embedder\": \"Timesteps\", \"D\": %d}", (int)t0.shape[1]); fclose(j); fclose(f);
    }
    for (int i = 0; i < nb; i++)
    {
        char p[1024]; snprintf(p, sizeof(p), "%s/weights_b%d.f16.part", argv[2], i);
        FILE *f = fopen(p, "wb"); if (!f) { perror(p); return 1; }
        for (int m = 0; m < 10; m++)
        {
            char n[128]; snprintf(n, sizeof(n), "blocks.%d.%s.weight", i, mats[m]);
            int64_t N, K; float *w = read_f32(n, &N, &K);
            if (N % 64 || K % 32) { fprintf(stderr, "%s: shape %lld x %lld not a multiple of 64 x 32\n", n, (long long)N, (long long)K); return 1; }
            put(f, w, N, K, threads); free(w);
        }
        fclose(f);
        char q[1024]; snprintf(q, sizeof(q), "%s/weights_b%d.f16", argv[2], i); rename(p, q);          /* complete files only */
        snprintf(p, sizeof(p), "%s/norms_b%d.f32", argv[2], i); f = fopen(p, "wb");
        for (int m = 0; m < 4; m++) { char n[128]; snprintf(n, sizeof(n), "blocks.%d.%s.weight", i, norms[m]); int64_t r, c; float *w = read_f32(n, &r, &c); fwrite(w, 4, (size_t)(r * c), f); free(w); }
        fclose(f);
        gettimeofday(&b, 0); printf("block %d done, %.1f s\n", i, (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1e6); fflush(stdout);
    }
    return 0;
}
