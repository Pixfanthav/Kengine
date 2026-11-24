/* kxl.inc (included by kblock_imp.c) -- an SDXL UNet, one layer a call, in the order of the unet.tab the host writes; the host keeps the skips and makes the resnets' time embeddings.
 * w, prm: this layer's slices of unet.w, unet.prm; a resnet's prm is followed by its time embedding [co]. Heads of 64 columns: xl_attention, attention() cut down from 4 blocks a head to 2. */
static inline void xl_kc(uint32_t K) { uint32_t b = K / 32 < 16 ? K / 32 : 16; while ((K / 32) % b) b--; g_kc = b * 32; }
/* y [ceil(h / 2), nw x c] from x [h, sw x c]: pixel (2 r, 2 q), what a stride-2 convolution with a padding of 1 keeps of the same convolution on every pixel. An even pixel is the low half of its vector */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t sw, nw, c, odd; } xdn_t;   /* odd: pixel (2 r + 1, 2 q + 1), a convolution padded only below and right (diffusers' encoder) */
static void xdn_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    xdn_t *a = (xdn_t *)a_; const uint32_t swt = a->sw / 32, nwt = a->nw / 32, cb = a->c / 32; const HVX_Vector lo = Q6_V_vsplat_R(0xffff);
    for (uint32_t t = t0; t < t1; t++)
    {
        const uint32_t r = t / nwt, tc = t % nwt, sr = 2 * r + a->odd; HVX_Vector *d = (HVX_Vector *)blk(a->y, a->c, t, 0);
        for (uint32_t b = 0; b < cb; b++)
            for (uint32_t k = 0; k < 16; k++)
            {
                const uint32_t q0 = 32 * tc + 2 * k, i0 = 2 * q0 + a->odd, i1 = i0 + 2;
                /* past the source's width (an output padded up to 32 columns): 0 */
                const HVX_Vector v0 = i0 >= a->sw ? Q6_V_vzero() : ((const HVX_Vector *)blk((uint16_t *)a->x, a->c, sr * swt + (i0 >> 5), b))[(i0 & 31) >> 1],
                                 v1 = i1 >= a->sw ? Q6_V_vzero() : ((const HVX_Vector *)blk((uint16_t *)a->x, a->c, sr * swt + (i1 >> 5), b))[(i1 & 31) >> 1];
                /* even: each pair's low halves (pixel i0) and v1's low moved up; odd: the high halves (pixel i0 is the second of its pair) */
                d[b * 16 + k] = a->odd ? Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v0, 16), Q6_V_vand_VV(v1, Q6_V_vsplat_R(0xffff0000))) : Q6_V_vor_VV(Q6_V_vand_VV(v0, lo), Q6_Vw_vasl_VwR(v1, 16));
            }
    }
}
/* the transformer blocks' time: q k v, the attention, out and add, up, the gate, down and add */
static uint64_t g_xt[6]; static uint64_t g_xc;
#define XHMXSUM 1   /* SDXL: 1 = the rows' sums from a product with a block of ones (the HMX), 0 = from the softmax workers (HVX); OPT_XHVXSUM picks 0 at run time */
static uint64_t g_xa[3];   /* the attention's time (us): head copies and K transposes, its HMX products, the waits for the softmax */
#define XT(i) { const uint64_t n_ = HAP_perf_get_time_us(); g_xt[i] += n_ - g_xc; g_xc = n_; }
/* SDXL's softmax workers kept for a whole attention call: they wait for the next 32 query rows (gen) instead of being submitted again
 * for each (about 50 000 tiles a UNet pass, the submits about 200 of its 809 ms of attention). One fewer than the
 * HVX contexts, as the main thread takes rows too and a waiting worker keeps its context. */
typedef struct { int gen, quit, next, done; __fp16 *s; uint32_t nb; const uint32_t *mk; HVX_Vector *rsum; spintok_t tok; int kind, nparts; const hp_arg_t *hp; } xsm_t;
/* kind 0: the tile's 16 row pairs; 1 (OPT_XHP): a head's preparation in nparts key ranges (hp_tiles, each fencing its own scatters) */
static void xsm_rows(xsm_t *g)
{
    if (g->kind) { for (;;) { const int k = __atomic_fetch_add(&g->next, 1, __ATOMIC_RELAXED); if (k >= g->nparts) break;
                              const uint32_t n = g->hp->nk; hp_tiles((void *)g->hp, n * (uint32_t)k / (uint32_t)g->nparts, n * (uint32_t)(k + 1) / (uint32_t)g->nparts); } return; }
    for (;;) { const int k = __atomic_fetch_add(&g->next, 1, __ATOMIC_RELAXED); if (k >= 16) break; softmax_rows(g->s, g->nb, g->mk, g->rsum, k, k + 1, -1); }
}
static void xsm_worker(void *p)
{
    xsm_t *g = (xsm_t *)p; int seen = 0;
    for (;;)
    {
        int gen; while ((gen = __atomic_load_n(&g->gen, __ATOMIC_ACQUIRE)) == seen) { }
        if (__atomic_load_n(&g->quit, __ATOMIC_ACQUIRE)) break;
        seen = gen; xsm_rows(g); __atomic_fetch_add(&g->done, 1, __ATOMIC_RELEASE);
    }
    st_done(&g->tok);
}
static void xsm_go(xsm_t *g, __fp16 *s, uint32_t nb, HVX_Vector *rsum)
{ g->kind = 0; g->s = s; g->nb = nb; g->rsum = rsum; __atomic_store_n(&g->next, 0, __ATOMIC_RELAXED); __atomic_store_n(&g->done, 0, __ATOMIC_RELAXED); __atomic_fetch_add(&g->gen, 1, __ATOMIC_RELEASE); }
static void xsm_finish(xsm_t *g, int nwk) { xsm_rows(g); while (__atomic_load_n(&g->done, __ATOMIC_ACQUIRE) < nwk) { } }
static void xsm_prep(xsm_t *g, const hp_arg_t *hp, int nparts)
{ g->kind = 1; g->hp = hp; g->nparts = nparts; __atomic_store_n(&g->next, 0, __ATOMIC_RELAXED); __atomic_store_n(&g->done, 0, __ATOMIC_RELAXED); __atomic_fetch_add(&g->gen, 1, __ATOMIC_RELEASE); }
/* o[:, head] = softmax(q k^T) v over heads of 64; q carries the score's scale. q, k in OUT_HEADQ64, v in OUT_HEADV64, o [M x NH 64] */
static void xl_attention(const uint16_t *q, uint32_t M, const uint16_t *k, const uint16_t *v, uint32_t L, const uint32_t *mk, uint16_t *o, uint32_t heads)
{
    const uint32_t nq = M / 32, nk = L / 32, qhead = nq * 2 * BLK, khead = nk * 2 * BLK, Ch = 64;
    uint8_t *v_q = g_vtcm, *v_kt = v_q + qhead, *v_v = v_kt + khead, *v_o = v_v + khead, *v_s[3], *v_c;
    v_s[0] = v_o + 4 * BLK; v_s[1] = v_s[0] + (nk + 2) * BLK; v_s[2] = v_s[1] + (nk + 2) * BLK; v_c = v_s[2] + (nk + 2) * BLK;
    uint8_t *v_one = v_c + nk * BLK, *v_sum = v_one + nk * BLK; const HVX_Vector one_h = splat_hf(1.f);
    memset(v_one, 0, nk * BLK); { __fp16 *o1 = (__fp16 *)v_one; for (uint32_t kk = 0; kk < L; kk++) o1[(kk >> 5) * 1024 + 32 * ((kk & 31) & ~1) + (kk & 1)] = (__fp16)1.f; }
    const uint32_t qk_act = ((Ch >> 5) << 11) | 0x7C, qk_wgt = ((Ch >> 5) << 11) - 1;
    uint32_t kcb = (L < KC ? L : KC) / 32; while ((L / 32) % kcb) kcb--;
    const uint32_t kc = kcb * 32, av_act = ((kc >> 5) << 11) | 0x7C, av_wgt = ((kc >> 5) << 11) - 1, av_chunk = (kc / 32) * BLK, kt_half = 2 * BLK, kt_group = 2 * kt_half, v_half = (L / 32) * BLK;
    static xsm_t xg; const int nwk = SMJ - 1; memset(&xg, 0, sizeof(xg)); xg.mk = mk; st_init(&xg.tok, nwk);
    for (int t = 0; t < nwk; t++) { worker_pool_job_t job = { xsm_worker, &xg }; worker_pool_submit(g_pool, job); }
    for (uint32_t hd = 0; hd < heads; hd++)
    {
        uint64_t ta_ = HAP_perf_get_time_us();
        if (g_opt & OPT_XHP)
        {   /* the crew and the main thread share the head's copies and transposes (nwk + 1 key ranges) */
            static hp_arg_t ha; ha = (hp_arg_t){ v_q, v_v, v_kt, (const uint8_t *)(q + (size_t)hd * M * Ch), (const uint8_t *)(v + (size_t)hd * L * Ch), k + (size_t)hd * L * Ch,
                                                 qhead, khead, nk, kt_group, kt_half, 1, 2 };
            xsm_prep(&xg, &ha, nwk + 1); xsm_finish(&xg, nwk);
        }
        else {
        hvx_copy(v_q, (const uint8_t *)(q + (size_t)hd * M * Ch), qhead); hvx_copy(v_v, (const uint8_t *)(v + (size_t)hd * L * Ch), khead);
        const uint16_t *kh = k + (size_t)hd * L * Ch;
        for (uint32_t t = 0; t < nk; t++) for (int j = 0; j < 2; j++)
        {
            if (j == 0 && (t & 7) == 0 && t + 8 < nk) prefetch(kh + (size_t)(t + 8) * 2 * BH, 32768);
            transpose_block(v_kt + (t / 2) * kt_group + (t % 2) * kt_half + j * BLK, kh + ((size_t)t * 2 + j) * BH);
        }
        scatter_fence(v_kt);
        }
        { const uint64_t n_ = HAP_perf_get_time_us(); g_xa[0] += n_ - ta_; ta_ = n_; }
        HVX_Vector rs[3][16];
        #define XQK(qt_, dst_) hmx_mpy_row(v_q + (qt_) * 2 * BLK, qk_act, v_kt, qk_wgt, (dst_), kt_half, g_bias, nk)
        #define XAV(qt_) do { uint8_t *src_ = v_s[(qt_) % 3]; HVX_Vector *rr_ = rs[(qt_) % 3]; \
            hmx_mpy_chunks(src_, av_act, v_v, av_wgt, v_o, 0, g_bias_av, av_chunk, L / kc); hmx_mpy_chunks(src_, av_act, v_v + v_half, av_wgt, v_o + BLK, 0, g_bias_av, av_chunk, L / kc); \
            if (g_hmxsum) { hmx_mpy_chunks(src_, av_act, v_one, av_wgt, v_sum, 0, g_bias_av, av_chunk, L / kc); \
            for (int kk = 0; kk < 16; kk++) { HVX_Vector s_ = Q6_V_vsplat_R(Q6_R_vextract_VR(((HVX_Vector *)v_sum)[kk], 0)); HVX_VectorPair w_ = Q6_Wqf32_vmpy_VhfVhf(s_, one_h); rr_[kk] = rows_hf(recip_sf(sfv(Q6_V_lo_W(w_))), recip_sf(sfv(Q6_V_hi_W(w_)))); } } \
            HVX_Vector *ov_ = (HVX_Vector *)v_o; for (int kk = 0; kk < 16; kk++) { const HVX_Vector r_ = rr_[kk], p0_ = hf(Q6_Vqf16_vmpy_VhfVhf(ov_[kk], r_)), p1_ = hf(Q6_Vqf16_vmpy_VhfVhf(ov_[16 + kk], r_)); ov_[kk] = p0_; ov_[16 + kk] = p1_; } \
            hvx_copy((uint8_t *)blk(o, heads * Ch, (qt_), 2 * hd), v_o, 2 * BLK); } while (0)
        XQK(0, v_s[0]);
        for (uint32_t qt = 0; qt < nq; qt++)
        {
            xsm_go(&xg, (__fp16 *)v_s[qt % 3], nk, rs[qt % 3]);
            ta_ = HAP_perf_get_time_us();
            if (qt > 0) XAV(qt - 1);
            if (qt + 1 < nq) XQK(qt + 1, v_s[(qt + 1) % 3]);
            { const uint64_t n_ = HAP_perf_get_time_us(); g_xa[1] += n_ - ta_; ta_ = n_; }
            xsm_finish(&xg, nwk);
            g_xa[2] += HAP_perf_get_time_us() - ta_;
        }
        XAV(nq - 1);
        #undef XQK
        #undef XAV
    }
    __atomic_store_n(&xg.quit, 1, __ATOMIC_RELEASE); __atomic_fetch_add(&xg.gen, 1, __ATOMIC_RELEASE); st_wait(&xg.tok);
}
/* the cross attentions' K and V of the text, the same through a picture's steps, kept on the DSP by block and picture
 * (kblock_xrun's flag 8; flag 4, a picture's first call, forgets them): a slot holds K then V, 96 x up to 1280 each. g_kvb: the transformer
 * block's place in the step (kblock_xrun), g_kvp the picture (kblock_xop), g_kvb -1: not kept */
#define XKVN 160
#define XKVS ((size_t)2 * 96 * 1280)
static uint16_t *g_kvc; static uint8_t g_kvok[XKVN]; static int g_kvb = -1, g_kvp, g_kvon, g_kvbad;
/* the text's rows a picture (xop_one's op 5 sets it from x2Len): 96 for one CLIP chunk of 77 tokens, 32 up from 77 k for k chunks
 * (long prompts, ComfyUI's way: 75 tokens a chunk, each with its own start and end), the 77 k first ones real, the rest masked */
static uint32_t g_xlc = 96;
/* op 3's output width when the levels are padded to 32 each (kblock_xrun: the skip it meets next), 0: twice the input's */
static uint32_t g_xupw;
/* one attention: x's LayerNorm through q, k, v (k, v from the text if ctx), scores, the out projection and its bias, added to x */
static void xl_attn(const uint16_t **wp, const ln_arg_t *lf, const uint16_t *ctx, uint16_t *x, uint32_t T, uint32_t c, const float *bo, uint16_t *s, uint64_t *th, const uint16_t *const *L)
{
    /* L: a runtime LoRA's (la, lb) for q, k, v, o (8 pointers, 0 for none; xl_block reads them after the block's weights) */
    static const uint16_t *const L0[8] = { 0 }; if (!L) L = L0;
    const uint32_t D2 = c, CD = 2048, C = g_xlc, CV = 77 * (g_xlc / 77); const uint16_t *wq = *wp, *wk = WP(wq, (size_t)c * D2), *wv = WP(wk, (size_t)(ctx ? CD : c) * D2), *wo = WP(wv, (size_t)(ctx ? CD : c) * D2);
    *wp = WP(wo, (size_t)D2 * c);
    uint16_t *qb = s, *kb = qb + (size_t)T * D2, *vb = kb + (size_t)T * D2, *ob = vb + (size_t)T * D2, *yb = ob + (size_t)T * D2, *ck = yb + (size_t)T * c, *cv = ck + (size_t)C * D2;
    g_D = c; xl_kc(c); g_xc = HAP_perf_get_time_us();
    if (ctx)
    {
        wset_t w1 = { wq, D2, qb, OUT_HEADQ64, 0, 0, 0, L[0], L[1] }; product_multi(0, ln_fill, (void *)lf, T, c, &w1, 1, 128, 0, th);
        const int slot = g_kvb >= 0 && g_kvb * 2 + g_kvp < XKVN && D2 <= 1280 && C == 96 ? g_kvb * 2 + g_kvp : -1;   /* one chunk's text only */
        if (slot >= 0 && !g_kvc && !g_kvbad) { g_kvc = (uint16_t *)memalign(128, XKVN * XKVS * 2); if (!g_kvc) g_kvbad = 1; }
        uint16_t *kc = slot >= 0 && g_kvc ? g_kvc + (size_t)slot * XKVS : 0;
        if (kc && g_kvok[slot]) { ck = kc; cv = kc + XKVS / 2; }
        else
        {
            xl_kc(CD); product_lo(ctx, C, CD, wk, D2, ck, OUT_HEADQ64, 16, L[2], L[3], th); product_lo(ctx, C, CD, wv, D2, cv, OUT_HEADV64, 16, L[4], L[5], th);
            if (kc) { hvx_copy((uint8_t *)kc, (const uint8_t *)ck, (size_t)C * D2 * 2); hvx_copy((uint8_t *)(kc + XKVS / 2), (const uint8_t *)cv, (size_t)C * D2 * 2); g_kvok[slot] = 1; }
        }
    }
    else { wset_t w3[3] = { { wq, D2, qb, OUT_HEADQ64, 0, 0, 0, L[0], L[1] }, { wk, D2, kb, OUT_HEADQ64, 0, 0, 0, L[2], L[3] }, { wv, D2, vb, OUT_HEADV64, 0, 0, 0, L[4], L[5] } }; product_multi(0, ln_fill, (void *)lf, T, c, w3, 3, 128, 0, th); }
    XT(0); g_kc = 0;
    const uint32_t mks[4] = { T, T, 0, 0 }, mkc[4] = { CV, C, 0, 0 };
    if (g_hmxsum) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - ((PSH + 4) << 10));
    { const uint32_t cm = g_cmwt; if (ctx) g_cmwt = 0;   /* the padding-column mask is the self attention's (its keys are the pixels), not the text's */
      if (ctx) xl_attention(qb, T, ck, cv, C, mkc, ob, c / 64); else xl_attention(qb, T, kb, vb, T, mks, ob, c / 64); g_cmwt = cm; }
    if (g_hmxsum) for (int ch = 0; ch < 32; ch++) ((uint32_t *)g_bias_av)[ch] = (uint32_t)(0x3C00 - (PSH << 10));
    XT(1); xl_kc(D2);
    if (g_opt & OPT_XRES)
    {   /* x += out + bias in the out product's post */
        static HVX_Vector bvo[MAXD / 32]; for (uint32_t b = 0; b < c / 32; b++) bvo[b] = colvec(bo + b * 32);
        g_resb = bvo; g_resg = 0; product_lo(ob, T, D2, wo, c, x, OUT_RESG, 128, L[6], L[7], th); g_resb = 0; g_kc = 0; g_D = c;
    }
    else { product_lo(ob, T, D2, wo, c, yb, OUT_ACT, 128, L[6], L[7], th); g_kc = 0; g_D = c; vk_bias(yb, T / 32, c, bo); vk_add(yb, x, T / 32, c); }
    XT(2);
}
/* a transformer block, x [T x c] in place */
static void xl_block(const uint16_t *w, const float *p, const uint16_t *ctx, uint16_t *x, uint32_t T, uint32_t c, uint16_t *s, uint64_t *th)
{
    static HVX_Vector mv[3][2][MAXD / 32]; const uint32_t tiles = T / 32, F = 4 * c; g_D = c;
    for (int m = 0; m < 3; m++) for (uint32_t cb = 0; cb < c / 32; cb++) { mv[m][1][cb] = colvec(p + (2 * m) * c + cb * 32); mv[m][0][cb] = colvec(p + (2 * m + 1) * c + cb * 32); }
    const float *bo1 = p + 6 * c, *bo2 = bo1 + c, *b1 = bo2 + c, *b2 = b1 + 2 * F;
    ln_arg_t lf[3]; for (int i = 0; i < 3; i++) lf[i] = (ln_arg_t){ x, 0, mv[i][0], mv[i][1] };
    const uint16_t *wp = w;
    /* g_lhdr (op | 0x400): a runtime LoRA after the block's eleven matrices, Linear by Linear in their order: attn1 q k v o, attn2
       q k v o, the GEGLU's two halves (up, gate), down; each { lr, gain } then A^T, { lr, gain } then B^T (lora_next), lr 0 for none */
    const uint16_t *L[22] = { 0 };
    if (g_lhdr)
    {
        const uint16_t *e = w; const uint32_t CD = 2048;
        const uint32_t KN[11][2] = { { c, c }, { c, c }, { c, c }, { c, c }, { c, c }, { CD, c }, { CD, c }, { c, c }, { c, F }, { c, F }, { F, c } };
        for (int i = 0; i < 11; i++) e = WP(e, (size_t)KN[i][0] * KN[i][1]);
        for (int i = 0; i < 11; i++) e = lora_next(e, KN[i][0], KN[i][1], &L[2 * i], &L[2 * i + 1]);
    }
    xl_attn(&wp, &lf[0], 0, x, T, c, bo1, s, th, L);
    xl_attn(&wp, &lf[1], ctx, x, T, c, bo2, s, th, L + 8);
    /* the gated feed-forward: the up product in eight parts [T x c], the first four times gelu of the last four, then the down product a part at a time */
    const uint16_t *wm1 = wp, *wm2 = WP(wm1, (size_t)c * 2 * F); uint16_t *hid = s, *yb = hid + (size_t)T * 2 * F; const size_t pn = (size_t)T * c;
    /* the up product as GEGLU pairs (gelu 4): the gate half (wm1's last F columns, its 64-column groups after the first F's) and the up half
     * by group, up * gelu(gate) made in the post and only the F up columns written, in four parts as the down product reads them */
    {
        static HVX_Vector bv[8 * 40]; const uint32_t cb = c / 32; for (uint32_t i = 0; i < 8 * cb; i++) bv[i] = colvec(b1 + i * 32);
        xl_kc(c); g_splitn = 4;
        wset_t wg[2] = { { WP(wm1, (size_t)c * F), F, hid, OUT_SPLIT4, 0, 0, bv + 4 * cb, L[18], L[19] }, { wm1, F, hid, OUT_SPLIT4, 0, 0, bv, L[16], L[17] } };
        product_multi(0, ln_fill, &lf[2], T, c, wg, 2, 128, 4, th);
    }
    XT(3);
    const int xr = (g_opt & OPT_XRES) != 0; static HVX_Vector bv2[MAXD / 32];
    if (xr) { for (uint32_t b = 0; b < c / 32; b++) bv2[b] = colvec(b2 + b * 32); g_resb = bv2; g_resg = 0; g_resx = x; }   /* the last part's post adds bias and x */
    XT(4); for (uint32_t i = 0; i < 4; i++) { wset_t w2 = { wm2, c, yb, i ? (xr && i == 3 ? OUT_ADDRESG : OUT_ADD) : OUT_ACT, F, i * c }; product_multi(hid + i * pn, 0, 0, T, c, &w2, 1, 128, 0, th); }
    if (L[20])
    {   /* down's LoRA on its own (the product goes a K part at a time): t = hid A^T by parts, then t B^T added where down's result went */
        const uint32_t lr = L[21][-64]; uint16_t *tl = yb + (size_t)T * c;
        for (uint32_t i = 0; i < 4; i++) { wset_t wa = { L[20], lr, tl, i ? OUT_ADD : OUT_ACT, F, i * c }; product_multi(hid + i * pn, 0, 0, T, c, &wa, 1, 128, 0, th); }
        xl_kc(lr); product_to(tl, T, lr, L[21], c, xr ? x : yb, OUT_ADD, 128, 0, th); xl_kc(c);
    }
    g_kc = 0; if (xr) g_resb = 0; else { vk_bias(yb, tiles, c, b2); vk_add(yb, x, tiles, c); } XT(5);
}
static uint16_t *g_xscr; static size_t g_xscrn;
/* op 0 convolution; 1 resnet; 2 down; 3 up; 4 norm + proj_in; 5 transformer block (x2 the text [96 x 2048]); 6 proj_out + x2; 9 the last norm and convolution */
static int xop_one(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, const uint16_t *x2, int x2Len, uint16_t *y, int yLen,
               int op, int h, int wd, int ci, int co, uint32_t *times, int timesLen)
{
    const int vwp = (op >> 16) & 0xfff, lo = (op & 0x400) != 0; op &= 0xff;   /* bits 16..27:
                                                                         the image's real width (the columns past it padding, kept 0), 0: wd */
    const size_t M = (size_t)h * wd; (void)wLen; const uint32_t mc = ci > co ? ci : co;
    if (wd % 32 || ci % 32 || co % 32 || (size_t)xLen < M * ci || timesLen < 2 || mc > 4096) return AEE_EBADPARM;
    if (op == 5 && x2Len >= 96 * 2048) g_xlc = (uint32_t)(x2Len / 2048);
    const size_t need = op == 5 ? M * ci * 9 + (size_t)g_xlc * 4 * ci + (lo ? M * 512 : 0) : op == 3 ? 4 * M * mc : 2 * M * mc;
    if (g_xscrn < need) { free(g_xscr); g_xscr = (uint16_t *)memalign(128, need * 2); g_xscrn = g_xscr ? need : 0; if (!g_xscr) return AEE_ENOMEMORY; }
    const uint64_t t0 = HAP_perf_get_time_us(); eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_lhdr = lo; g_lora = 1.f;
    g_vkv = 0; g_smj = 4; g_hmxsum = (g_opt & OPT_XHVXSUM) ? 0 : XHMXSUM; g_convsplit = 1; vd_t d; memset(&d, 0, sizeof(d)); d.w = w; d.p = prm; d.h = h; d.wd = wd; d.vw = vwp && vwp < wd ? (uint32_t)vwp : (uint32_t)wd; d.run = 1;
    uint16_t *b = g_xscr, *c = b + M * mc; const uint32_t tiles = (uint32_t)(M / 32);
    switch (op)
    {
    case 0: vd_conv(&d, x, y, ci, co, 3); break;
    case 1: /* the resnet's time in g_xt : norm 1, conv 1 + bias, norm 2, conv 2, the skip and add, the convs' HMX */
            { const uint64_t th0 = d.th, tq0 = g_tqk, tv0 = g_tav; g_xc = HAP_perf_get_time_us();
              /* g_xt: 0 norms, 1 conv 1, 2 conv 2, 3 the convs' activation fills, 4 their waits for weights, 5 their HMX */
              vd_gn(&d, x, b, ci, 1e-5f, 1); XT(0) if (g_opt & OPT_XTEMB) { g_vdb2 = prm + prmLen - co; vd_conv(&d, b, c, ci, co, 3); g_vdb2 = 0; } else { vd_conv(&d, b, c, ci, co, 3); vk_bias(c, tiles, co, prm + prmLen - co); } XT(1)
              vd_gn(&d, c, b, co, 1e-5f, 1); XT(0) vd_conv(&d, b, y, co, co, 3); XT(2)
              if (ci != co) { vd_conv(&d, x, b, ci, co, 1); vk_add(b, y, tiles, co); } else vk_add(x, y, tiles, co);
              XT(0) g_xt[3] += g_tqk - tq0; g_xt[4] += g_tav - tv0; g_xt[5] += d.th - th0; }
            break;
    case 2: if ((g_opt & OPT_XDN) && !(wd % 64)) { g_xdn = 1; vd_conv(&d, x, y, ci, co, 3); g_xdn = 0; }   /* only the kept pixels */
            else { vd_conv(&d, x, b, ci, co, 3); const uint32_t nw = ((uint32_t)wd / 2 + 31) & ~31u, nh = (h + 1) / 2; xdn_t a = { b, y, (uint32_t)wd, nw, (uint32_t)co }; tiles_parallel_range(xdn_tiles, &a, 0, nh * (nw / 32)); } break;
    case 3: { const uint32_t uw = g_xupw ? g_xupw : 2 * (uint32_t)wd; vk_up(x, b, h, wd, uw, ci); d.h = 2 * h; d.wd = uw; d.vw = 2 * d.vw; vd_conv(&d, b, y, ci, co, 3); } break;
    case 4: vd_gn(&d, x, b, ci, 1e-6f, 0); vd_conv(&d, b, y, ci, co, 1); break;
    case 5: if (x2Len < 96 * 2048) { err = AEE_EBADPARM; break; } hvx_copy((uint8_t *)y, (const uint8_t *)x, M * ci * 2);
            /* a padded width: the self attention's keys in the padding columns masked (g_cmwt), and those columns put back to 0 after */
            if (d.vw < (uint32_t)wd) { g_cmwt = (uint32_t)wd / 32; g_cmvw = d.vw; }
            xl_block(w, prm, x2, y, (uint32_t)M, ci, b, &d.th); g_cmwt = 0;
            if (d.vw < (uint32_t)wd) vk_pad(y, (uint32_t)h, (uint32_t)wd, d.vw, (uint32_t)ci); break;
    case 6: vd_conv(&d, x, y, ci, co, 1); vk_add(x2, y, tiles, co); break;
    case 9: vd_gn(&d, x, b, ci, 1e-5f, 1); vd_conv(&d, b, y, ci, co, 3); break;
    case 10: hvx_copy((uint8_t *)y, (const uint8_t *)x, (uint32_t)xLen * 2); break;   /* kblock_xrun's copies: xLen elements, x -> y */
    case 11: /* kblock_xrun's concat, one picture: y's tiles = x's (ci channels) then x2's (co channels) */
            { const size_t tl = M / 32; const uint32_t bx = (uint32_t)(ci / 32) * 2048, bk = (uint32_t)(co / 32) * 2048; uint8_t *yy = (uint8_t *)y; const uint8_t *xx = (const uint8_t *)x, *kk = (const uint8_t *)x2;
              for (size_t t = 0; t < tl; t++) { hvx_copy(yy + t * (bx + bk), xx + t * bx, bx); hvx_copy(yy + t * (bx + bk) + bx, kk + t * bk, bk); } }
            break;
    default: err = AEE_EBADPARM;
    }
    g_kc = 0; g_convsplit = 0; eng_end(&eg); (void)h_; (void)yLen;
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t0); times[1] = (uint32_t)d.th;
    if (timesLen >= 8) for (int i = 0; i < 6; i++) { times[2 + i] = (uint32_t)g_xt[i]; g_xt[i] = 0; }
    if (timesLen >= 11) for (int i = 0; i < 3; i++) { times[8 + i] = (uint32_t)g_xa[i]; g_xa[i] = 0; }
    return err;
}
int kblock_xop(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, const uint16_t *x2, int x2Len, uint16_t *y, int yLen,
               int op, int h, int wd, int ci, int co, uint32_t *times, int timesLen)
{ g_need8 = 1;   /* VTCM laid out for 8 MB */
    /* op | 0x100: the cond and uncond pictures in one call, with the layer's weights (one cache maintenance of them for both: each call's
     * was about 4 ms, 21 s of 20 steps at two calls a layer). x, x2, y hold the two one after the other; prm is the
     * layer's parameters, then (resnets) the two time embeddings, co each: picture 1's is copied over picture 0's for its turn */
    if (!(op & 0x100)) return xop_one(h_, w, wLen, prm, prmLen, x, xLen, x2, x2Len, y, yLen, op, h, wd, ci, co, times, timesLen);
    const int vwb = op & 0x0fff0400; op &= 0xff;
    const size_t M = (size_t)h * wd; const int nh = op == 2 ? (h + 1) / 2 : op == 3 ? 2 * h : h, nw = op == 2 ? (wd / 2 + 31) & ~31 : op == 3 ? (g_xupw ? (int)g_xupw : 2 * wd) : wd; const size_t Mo = (size_t)nh * nw;
    const int pl = op == 1 ? prmLen - co : prmLen; uint32_t ts[11] = { 0 }; int err = 0;
    for (int i = 0; i < 2 && !err; i++)
    {
        uint32_t t1[11] = { 0 };
        if (i == 1 && op == 1) memcpy((float *)prm + pl - co, prm + pl, (size_t)co * 4);
        g_kvp = i;
        err = xop_one(h_, w, wLen, prm, pl, x + (size_t)i * M * ci, xLen / 2, x2 ? x2 + (size_t)i * (x2Len / 2) : x2, x2Len / 2, y + (size_t)i * Mo * co, yLen / 2, op | vwb, h, wd, ci, co, t1, timesLen < 11 ? timesLen : 11);
        for (int k = 0; k < 11; k++) ts[k] += t1[k];
    }
    for (int k = 0; k < timesLen && k < 11; k++) times[k] = ts[k];
    return err;
}
/* kblock_xrun: a run of the UNet's layers in one call, both pictures (the host's layer loop of xl_gen moved here: a call cost about
 * 8.6 ms outside the DSP whatever it carried, 115 of them a step). tab: [0] flags (1: the step's first run: state reset,
 * zin copied in; 2: its last: the result copied to out), [1] the ops, [2] H, [3] W, then 7 words an op: op (unet.tab's: 7 keeps a skip,
 * 8 concatenates the last kept), ci, co, the weights' offset and length in w (elements), the parameters' in prm (floats; a resnet's
 * end with the two time embeddings, as kblock_xop's op | 0x100). xa, xb: the two activation buffers the layers alternate between,
 * st0: the transformer's input kept for its proj_out, sk: the skips, stacked. All four are the DSP's own between a step's calls (the
 * host never reads or writes them), so nothing about them depends on cache maintenance. times: [0] all, [1] ops done, [2 + op] by op */
#define XRSK 16
static struct { int par, hh, wd, cc, ns, n5, vw, skc[XRSK], skw[XRSK]; size_t sko[XRSK + 1]; } g_xr;   /* vw: the real width at this level (tab[0] bits 16..27) */
int kblock_xrun(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint32_t *tab, int tabLen, const uint16_t *ctx, int ctxLen,
                const uint16_t *zin, int zinLen, uint16_t *xa, int xaLen, uint16_t *xb, int xbLen, uint16_t *st0, int st0Len, uint16_t *sk, int skLen,
                uint16_t *out, int outLen, uint32_t *times, int timesLen, int unused)
{ g_need8 = 1;   /* VTCM laid out for 8 MB */
    (void)unused; if (tabLen < 4 || timesLen < 12) return AEE_EBADPARM;
    const uint32_t flags = tab[0], n = tab[1], H = tab[2], W = tab[3]; if ((uint32_t)tabLen < 4 + 7 * n || ctxLen < 2 * 96 * 2048) return AEE_EBADPARM;
    const uint64_t t0 = HAP_perf_get_time_us(); int err = 0; uint32_t top[10] = { 0 }, tc[11], done = 0;
    #define XRCOPY(dst_, src_, n_) err = xop_one(h_, w, 0, prm, 0, (src_), (int)(n_), 0, 0, (dst_), (int)(n_), 10, 1, 32, 32, 32, tc, 2)
    if (flags & 1)
    {
        memset(&g_xr, 0, sizeof(g_xr)); g_xr.hh = (int)H; g_xr.wd = (int)W; g_xr.cc = 32; g_xr.vw = (flags >> 16) & 0xfff ? (int)((flags >> 16) & 0xfff) : (int)W;
        if (flags & 4) memset(g_kvok, 0, sizeof(g_kvok));    /* a new picture: its text's K and V made again */
        g_kvon = (flags & 8) != 0;
        const size_t zn = (size_t)2 * H * W * 32; if ((size_t)zinLen < zn || (size_t)xaLen < zn) return AEE_EBADPARM;
        XRCOPY(xa, zin, zn);
    }
    for (uint32_t i = 0; i < n && !err; i++, done++)
    {
        const uint32_t *e = tab + 4 + 7 * i; const int op = (int)(e[0] & 0xff), lob = (int)(e[0] & 0x400), ci = (int)e[1], co = (int)e[2];
        uint16_t *X = g_xr.par ? xb : xa, *Y = g_xr.par ? xa : xb; const int yl = g_xr.par ? xaLen : xbLen, hh = g_xr.hh, wd = g_xr.wd, cc = g_xr.cc; const size_t M = (size_t)hh * wd;
        if (op == 7)
        {
            const size_t nel = 2 * M * cc; if (g_xr.ns >= XRSK || g_xr.sko[g_xr.ns] + nel > (size_t)skLen) { err = AEE_ENOMEMORY; break; }
            XRCOPY(sk + g_xr.sko[g_xr.ns], X, nel); g_xr.skc[g_xr.ns] = cc; g_xr.skw[g_xr.ns] = wd; g_xr.sko[g_xr.ns + 1] = g_xr.sko[g_xr.ns] + nel; g_xr.ns++; continue;
        }
        if (op == 8)
        {
            if (g_xr.ns <= 0) { err = AEE_EBADPARM; break; }
            g_xr.ns--; const int cs = g_xr.skc[g_xr.ns], nc = cc + cs; const uint16_t *kp = sk + g_xr.sko[g_xr.ns]; if (2 * M * nc > (size_t)yl) { err = AEE_ENOMEMORY; break; }
            for (int c = 0; c < 2 && !err; c++) err = xop_one(h_, w, 0, prm, 0, X + (size_t)c * M * cc, (int)(M * cc), kp + (size_t)c * M * cs, (int)(M * cs), Y + (size_t)c * M * nc, (int)(M * nc), 11, hh, wd, cc, cs, tc, 2);
            g_xr.par ^= 1; g_xr.cc = nc; continue;
        }
        if (op < 0 || op > 9 || (size_t)e[3] + e[4] > (size_t)wLen || (size_t)e[5] + e[6] > (size_t)prmLen) { err = AEE_EBADPARM; break; }
        g_xupw = op == 3 && g_xr.ns > 0 ? (uint32_t)g_xr.skw[g_xr.ns - 1] : 0;   /* an upsample meets the next skip's width (each level padded to 32 on its own) */
        const int nh = op == 2 ? (hh + 1) / 2 : op == 3 ? 2 * hh : hh, nw = op == 2 ? (wd / 2 + 31) & ~31 : op == 3 ? (g_xupw ? (int)g_xupw : 2 * wd) : wd; const size_t Mo = (size_t)nh * nw;
        if (2 * Mo * co > (size_t)yl) { err = AEE_ENOMEMORY; break; }
        if (op == 4) { if (2 * M * ci > (size_t)st0Len) { err = AEE_ENOMEMORY; break; } XRCOPY(st0, X, 2 * M * ci); if (err) break; }
        const uint16_t *x2 = op == 5 ? ctx : st0; const int x2l = op == 5 ? ctxLen : op == 6 ? (int)(2 * M * ci) : 0; uint32_t t[11] = { 0 };
        g_kvb = op == 5 && g_kvon ? g_xr.n5++ : -1;
        err = kblock_xop(h_, w + e[3], (int)e[4], prm + e[5], (int)e[6], X, (int)(2 * M * ci), x2, x2l, Y, (int)(2 * Mo * co), op | 0x100 | lob | (g_xr.vw << 16), hh, wd, ci, co, t, 11);
        g_kvb = -1;
        g_xupw = 0; top[op] += t[0]; g_xr.par ^= 1; g_xr.hh = nh; g_xr.wd = nw; g_xr.cc = co; if (op == 2) g_xr.vw = (g_xr.vw + 1) / 2; else if (op == 3) g_xr.vw *= 2;
    }
    if (!err && (flags & 2))
    {
        const size_t on = (size_t)2 * H * W * 64; if ((size_t)outLen < on) return AEE_EBADPARM;
        XRCOPY(out, g_xr.par ? xb : xa, on);
    }
    #undef XRCOPY
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t0); times[1] = done; for (int k = 0; k < 10 && 2 + k < timesLen; k++) times[2 + k] = top[k];
    return err;
}

/* FLUX.2's VAE encoder on the DSP (kblock_xvae's stop bit 30): x the picture [8h x 8wd x 32] in the activation layout (its three channels then 0), y the
 * latent [h x wd x 64] (quant_conv's: the mean is its first 32). conv_in, four levels of two resnets and (the first three) a stride-2 convolution padded below
 * and right (made as the full convolution's odd pixels), the middle (resnet, attention, resnet), norm + silu, conv_out, quant_conv; weights in that order. stop: the stage to stop after (17 the whole) */
static int xvae_enc(const uint16_t *w, const float *prm, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int stop,
                    uint32_t *times, int timesLen)
{
    const size_t Mf = (size_t)64 * h * wd, part = Mf * 256;
    if (wd % 32 || h % 2 || (size_t)xLen < Mf * 32 || (size_t)scrLen < 2 * part + part / 2 || (size_t)yLen < (size_t)h * wd * 64 || timesLen < 32 || stop > 17 || stop < 0) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_vkv = 0; g_tqk = g_tpost = 0; g_lora = 1.f; g_smj = 4; g_hmxsum = 1;
    vd_t d; memset(&d, 0, sizeof(d)); d.w = w; d.p = prm; d.a = scr; d.b = scr + part / 2; d.c = d.b + part; d.h = 8 * h; d.wd = 8 * wd; d.vw = d.wd; d.run = 1; d.cap = part / 2;
    hvx_copy((uint8_t *)d.a, (const uint8_t *)x, Mf * 32 * 2);
    static const uint32_t EC[4] = { 128, 256, 512, 512 };
    uint64_t t = HAP_perf_get_time_us(); int k = 0; uint32_t ci = 128;
#define XS(...) { __VA_ARGS__; { const uint64_t n_ = HAP_perf_get_time_us(); times[k] = (uint32_t)(n_ - t); t = n_; } if (k == stop) goto enc_done; k++; }
#define XSWAP { uint16_t *s_ = d.a; d.a = d.c; d.c = s_; }
    XS(vd_conv(&d, d.a, d.c, 32, 128, 3); XSWAP)
    for (int l = 0; l < 4; l++)
    {
        XS(vd_res(&d, ci, EC[l])) ci = EC[l]; XS(vd_res(&d, ci, ci))
        if (l < 3) XS(vd_conv(&d, d.a, d.c, ci, ci, 3);
                      { const uint32_t nw = d.wd / 2, nh = d.h / 2; xdn_t a = { d.c, d.a, d.wd, nw, ci, 1 }; tiles_parallel_range(xdn_tiles, &a, 0, nh * (nw / 32)); d.h = nh; d.wd = nw; d.vw = nw; })
    }
    XS(vd_res(&d, 512, 512)) XS(vd_attn(&d, 512)) XS(vd_res(&d, 512, 512))
    XS(vd_gn(&d, d.a, d.b, 512, 1e-6f, 1); { uint16_t *s_ = d.a; d.a = d.b; d.b = s_; })
    XS(vd_conv(&d, d.a, d.c, 512, 64, 3); XSWAP)
    XS(vd_conv(&d, d.a, d.c, 64, 64, 1); XSWAP)
enc_done:;
#undef XS
#undef XSWAP
    { static const uint16_t EO[18] = { 128, 128, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 64, 64 };
      const size_t n = (size_t)d.h * d.wd * EO[k < 18 ? k : 17]; if ((size_t)yLen >= n) hvx_copy((uint8_t *)y, (const uint8_t *)d.a, n * 2); }
    times[22] = (uint32_t)d.th; times[23] = (uint32_t)d.tp; times[24] = (uint32_t)d.tg; times[25] = (uint32_t)d.ta;
    eng_end(&eg); return 0;
}
/* ------------------------------------------------------------------ the SDXL VAE's decoder (the fp16-fix weights: the original's values pass 65504). x [h, wd x 32]: the latent in channels 0..3; y: stage stop's output, the last the picture's 64 channels.
 * scr: an image of 128 channels and two of 256 at the final size (8 h x 8 wd), the last levels' only full ones: as kblock_vdec. Stages as the host lists them */
int kblock_xvae(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen,
                int h, int wd, int stop, uint32_t *times, int timesLen)
{ g_need8 = 1;   /* VTCM laid out for 8 MB */
    const int enc = (stop >> 30) & 1; stop &= 0x3fffffff;   /* bit 30: FLUX.2's VAE encoder instead (Klein's references) */
    const int vw0 = (stop >> 8) & 0xfff; stop &= 255;   /* stop bits 8..19: the latent's real width, the columns past it padding (0 = wd) */
    const size_t Mf = (size_t)64 * h * wd, part = Mf * 256;
    if (enc) return xvae_enc(w, prm, x, xLen, scr, scrLen, y, yLen, h, wd, stop, times, timesLen);
    if (wd % 32 || h % 8 || (size_t)xLen < (size_t)h * wd * 32 || (size_t)scrLen < 2 * part + part / 2 || timesLen < 32 || stop > 21 || stop < 0) return AEE_EBADPARM;
    (void)h_; (void)wLen; (void)prmLen; eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_vkv = 0; g_tqk = g_tpost = 0; g_lora = 1.f; g_smj = 4; g_hmxsum = 1;
    uint16_t *const small = scr; vd_t d; memset(&d, 0, sizeof(d)); d.w = w; d.p = prm; d.a = small; d.b = scr + part / 2; d.c = d.b + part; d.h = h; d.wd = wd; d.vw = vw0 && vw0 < wd ? (uint32_t)vw0 : (uint32_t)wd; d.run = 1;
    hvx_copy((uint8_t *)d.a, (const uint8_t *)x, (size_t)h * wd * 32 * 2);
    static const uint16_t OUTC[22] = { 32, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128, 64 };
    uint64_t t = HAP_perf_get_time_us(); int k = 0;
#define XS(...) { __VA_ARGS__; { const uint64_t n_ = HAP_perf_get_time_us(); times[k] = (uint32_t)(n_ - t); t = n_; } if (k == stop) goto done; k++; }
#define XSWAP { uint16_t *s_ = d.a; d.a = d.c; d.c = s_; }
#define XUP(c_, last_) XS({ const uint32_t sw = d.wd, nw = 2 * d.wd; \
        if ((last_) && d.a != small) { hvx_copy((uint8_t *)small, (const uint8_t *)d.a, (size_t)d.h * d.wd * (c_) * 2); uint16_t *big = d.a, *other = d.b == small ? d.c : d.b; d.a = small; d.b = big; d.c = other; } \
        vk_up(d.a, d.c, d.h, sw, nw, (c_)); XSWAP if (last_) { uint16_t *s_ = d.b; d.b = d.c; d.c = s_; } \
        d.h *= 2; d.wd = nw; d.vw *= 2; vd_conv(&d, d.a, d.c, (c_), (c_), 3); XSWAP if (last_) { uint16_t *s_ = d.b; d.b = d.c; d.c = s_; } })
    XS(vd_conv(&d, d.a, d.c, 32, 32, 1); XSWAP)
    XS(vd_conv(&d, d.a, d.c, 32, 512, 3); XSWAP)
    XS(vd_res(&d, 512, 512)) XS(vd_attn(&d, 512)) XS(vd_res(&d, 512, 512))
    XS(vd_res(&d, 512, 512)) XS(vd_res(&d, 512, 512)) XS(vd_res(&d, 512, 512)) XUP(512, 0)
    XS(vd_res(&d, 512, 512)) XS(vd_res(&d, 512, 512)) XS(vd_res(&d, 512, 512)) XUP(512, 0)
    XS(vd_res(&d, 512, 256)) XS(vd_res(&d, 256, 256)) XS(vd_res(&d, 256, 256)) XUP(256, 1)
    XS(vd_res(&d, 256, 128)) XS(vd_res(&d, 128, 128)) XS(vd_res(&d, 128, 128))
    XS(vd_gn(&d, d.a, d.b, 128, 1e-6f, 0); { uint16_t *s_ = d.a; d.a = d.b; d.b = s_; })
    XS(vk_silu(d.a, d.b, d.h * (d.wd / 32), 128); vd_conv(&d, d.b, d.c, 128, 64, 3); XSWAP)
done:;
#undef XS
#undef XSWAP
#undef XUP
    { const size_t M = (size_t)d.h * d.wd, n = M * OUTC[k]; if ((size_t)yLen >= n) hvx_copy((uint8_t *)y, (const uint8_t *)d.a, n * 2); }
    times[22] = (uint32_t)d.th; times[23] = (uint32_t)d.tp; times[24] = (uint32_t)d.tg; times[25] = (uint32_t)d.ta;
    eng_end(&eg); return 0;
}
