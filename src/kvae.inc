/* kvae.inc (included by kblock_imp.c) -- convolutions on the HMX: a 3 x 3 convolution is a product whose activation row, for each pixel, is its nine
 * neighbours' channels side by side, K = 9 C. A pixel is a row of the activation layout, the image's rows one after another (width a multiple of
 * 32, so the row above is a whole number of tiles away); a tile's activation is made as it goes in (conv_fill), so nothing but the output leaves VTCM. */
typedef struct { const uint16_t *x; uint32_t h, w, c; int dyr; } conv_arg_t;   /* dyr: 0 all nine neighbours, else only row dy = dyr - 2 (vd_conv's split) */
/* these passes read each tile once from DDR: without a fetch ahead the misses are half their time (the norm passes of a 1024^2 picture, 1327 -> 664 ms on v81) */
static inline void vk_pf(const void *p, size_t bytes) { Q6_l2fetch_AP((void *)p, ((uint64_t)8192 << 32) | ((uint64_t)8192 << 16) | (uint32_t)((bytes + 8191) / 8192)); }
/* the activation's block run of one neighbour for tile t: dy rows, dx columns away; past the picture 0 */
static void conv_neighbour(const conv_arg_t *a, HVX_Vector *d, uint32_t t, int dy, int dx)
{
    const uint32_t wt = a->w / 32, cb = a->c / 32, row = t / wt, col = t % wt; const size_t bytes = (size_t)cb * BLK;
    if ((dy < 0 && row == 0) || (dy > 0 && row + 1 >= a->h)) { memset(d, 0, bytes); return; }
    const uint32_t ts = t + (uint32_t)dy * wt;      /* the same x, dy rows away */
    const HVX_Vector *s = (const HVX_Vector *)blk((uint16_t *)a->x, a->c, ts, 0);
    if (!dx) { hvx_copy((uint8_t *)d, (const uint8_t *)s, bytes); return; }
    /* vector k of a block holds rows 2k (low half of each word) and 2k + 1 (high half): one pixel along, a word's halves come from k and k +- 1 */
    const HVX_Vector *near = 0;                    /* the tile the first or the last row comes from */
    if (dx > 0 && col + 1 < wt) near = s + (size_t)cb * 16;
    if (dx < 0 && col > 0) near = s - (size_t)cb * 16;
    for (uint32_t b = 0; b < cb; b++)
    {
        const HVX_Vector *v = s + b * 16; HVX_Vector *o = d + b * 16;
        if (dx > 0)
        {
            for (int k = 0; k < 15; k++) o[k] = Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v[k], 16), Q6_Vw_vasl_VwR(v[k + 1], 16));
            o[15] = Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v[15], 16), near ? Q6_Vw_vasl_VwR(near[b * 16], 16) : Q6_V_vzero());
        }
        else
        {
            for (int k = 15; k > 0; k--) o[k] = Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v[k - 1], 16), Q6_Vw_vasl_VwR(v[k], 16));
            o[0] = Q6_V_vor_VV(near ? Q6_Vuw_vlsr_VuwR(near[b * 16 + 15], 16) : Q6_V_vzero(), Q6_Vw_vasl_VwR(v[0], 16));
        }
    }
}
typedef struct { const conv_arg_t *a; uint8_t *dst; uint32_t m0; } conv_fill_t;
static void conv_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    conv_fill_t *f = (conv_fill_t *)a_; const uint32_t cb = f->a->c / 32;
    const uint32_t wt = f->a->w / 32, T = f->a->h * wt;
    for (uint32_t t = t0; t < t1; t++)
    {
        const uint32_t tn = f->m0 + t + 1;                                          /* the next tile's three rows */
        if (tn < T) { const size_t b_ = (size_t)cb * BLK; vk_pf(blk((uint16_t *)f->a->x, f->a->c, tn + 1 < T ? tn + 1 : tn, 0), b_); if (tn >= wt) vk_pf(blk((uint16_t *)f->a->x, f->a->c, tn - wt + 1, 0), b_); if (tn + wt + 1 < T) vk_pf(blk((uint16_t *)f->a->x, f->a->c, tn + wt + 1, 0), b_); }
        if (f->a->dyr)
        {   /* one row of the kernel: its three neighbours, K = 3 C */
            uint8_t *d = f->dst + (size_t)t * 3 * cb * BLK; const int dy = f->a->dyr - 2;
            for (int dx = -1; dx <= 1; dx++) conv_neighbour(f->a, (HVX_Vector *)(d + (size_t)(dx + 1) * cb * BLK), f->m0 + t, dy, dx);
            continue;
        }
        uint8_t *d = f->dst + (size_t)t * 9 * cb * BLK;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
            conv_neighbour(f->a, (HVX_Vector *)(d + (size_t)(3 * (dy + 1) + dx + 1) * cb * BLK), f->m0 + t, dy, dx);
    }
}
static void conv_fill(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles)
{ conv_fill_t f = { (const conv_arg_t *)arg, v_act, m0 }; tiles_parallel_range(conv_tiles, &f, 0, tiles); }
/* a stride-2 3 x 3 convolution's activation for output pixel (r, q) only -- input pixels (2 r + dy, 2 q + dx), the same values
 * conv_neighbour gives those rows (so the kept outputs are the full convolution's, bit for bit). Output tile t: row r = t / (w / 64), its 32
 * pixels from q0 = 32 (t % (w / 64)); input pixel p = 2 q + dx is half p & 1 of vector (p & 31) / 2 of input tile p / 32 */
static void dconv_neighbour(const conv_arg_t *a, HVX_Vector *d, uint32_t t, int dy, int dx)
{
    const uint32_t swt = a->w / 32, nwt = a->w / 64, cb = a->c / 32, r = t / nwt, tc = t % nwt; const int sr = 2 * (int)r + dy;
    if (sr < 0 || sr >= (int)a->h) { memset(d, 0, (size_t)cb * BLK); return; }
    const HVX_Vector lo = Q6_V_vsplat_R(0xffff), hi = Q6_V_vsplat_R(0xffff0000);
    for (uint32_t b = 0; b < cb; b++)
        for (uint32_t k = 0; k < 16; k++)
        {
            const int p0 = 64 * (int)tc + 4 * (int)k + dx, p1 = p0 + 2;
            const HVX_Vector v0 = p0 < 0 ? Q6_V_vzero() : ((const HVX_Vector *)blk((uint16_t *)a->x, a->c, (uint32_t)sr * swt + ((uint32_t)p0 >> 5), b))[((uint32_t)p0 & 31) >> 1];
            const HVX_Vector v1 = ((const HVX_Vector *)blk((uint16_t *)a->x, a->c, (uint32_t)sr * swt + ((uint32_t)p1 >> 5), b))[((uint32_t)p1 & 31) >> 1];
            d[b * 16 + k] = dx & 1 ? Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v0, 16), Q6_V_vand_VV(v1, hi)) : Q6_V_vor_VV(Q6_V_vand_VV(v0, lo), Q6_Vw_vasl_VwR(v1, 16));
        }
}
static void dconv_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    conv_fill_t *f = (conv_fill_t *)a_; const uint32_t cb = f->a->c / 32;
    for (uint32_t t = t0; t < t1; t++)
    {
        if (f->a->dyr)
        {
            uint8_t *d = f->dst + (size_t)t * 3 * cb * BLK; const int dy = f->a->dyr - 2;
            for (int dx = -1; dx <= 1; dx++) dconv_neighbour(f->a, (HVX_Vector *)(d + (size_t)(dx + 1) * cb * BLK), f->m0 + t, dy, dx);
            continue;
        }
        uint8_t *d = f->dst + (size_t)t * 9 * cb * BLK;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
            dconv_neighbour(f->a, (HVX_Vector *)(d + (size_t)(3 * (dy + 1) + dx + 1) * cb * BLK), f->m0 + t, dy, dx);
    }
}
static void dconv_fill(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles)
{ conv_fill_t f = { (const conv_arg_t *)arg, v_act, m0 }; tiles_parallel_range(dconv_tiles, &f, 0, tiles); }
static int g_xdn;   /* vd_conv: the stride-2 form (SDXL's down op, OPT_XDN) */

/* ------------------------------------------------------------------ the decoder's other layers, on images [h, w x c] in the activation layout. vw <= w: the picture's own width,
 * the columns past it are padding, kept 0 so that a convolution sees what PyTorch's zero padding gives (vk_pad). */
/* the lanes of pixels past vw % 32 in vector k cleared; only the last tile of each row has them */
static void vk_padmask(HVX_Vector *m, uint32_t vw)
{
    const uint32_t r = vw % 32;
    for (uint32_t k = 0; k < 16; k++)
    {
        uint16_t t[64] __attribute__((aligned(128)));
        for (uint32_t l = 0; l < 64; l++) t[l] = !r || 2 * k + (l & 1) < r ? 0xffff : 0;
        m[k] = *(HVX_Vector *)t;
    }
}
static void vk_pad(uint16_t *y, uint32_t h, uint32_t w, uint32_t vw, uint32_t c)
{
    if (vw == w) return;
    HVX_Vector m[16]; vk_padmask(m, vw); const uint32_t wt = w / 32, last = vw / 32, cb = c / 32;
    for (uint32_t r = 0; r < h; r++) for (uint32_t tc = last; tc < wt; tc++)
    {
        HVX_Vector *v = (HVX_Vector *)blk(y, c, r * wt + tc, 0);
        if (tc > last || !(vw % 32)) memset(v, 0, (size_t)cb * BLK);
        else for (uint32_t b = 0; b < cb; b++) for (int k = 0; k < 16; k++) v[b * 16 + k] = Q6_V_vand_VV(v[b * 16 + k], m[k]);
    }
}
/* y = silu(GroupNorm(x)): groups of c / 32 channels, the mean and variance over the picture and the group */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t c, gs; double sum[32], sq[32]; qurt_mutex_t mu; const HVX_Vector *mean, *rstd, *gv, *bv; int silu; } vgn_t;
static void vgn_stats(void *a_, uint32_t t0, uint32_t t1)
{
    vgn_t *a = (vgn_t *)a_; const uint32_t cb = a->c / 32; double s[32] = { 0 }, q[32] = { 0 }; const HVX_Vector one = splat_hf(1.f);
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < cb; b++)
    {
        const HVX_Vector *v = (const HVX_Vector *)blk((uint16_t *)a->x, a->c, t, b); HVX_Vector e = Q6_V_vzero(), f = Q6_V_vzero();
        if (!(b & 3) && (t + 1 < t1 || b + 4 < cb)) vk_pf(blk((uint16_t *)a->x, a->c, t, b + 4), 8192);
        for (int k = 0; k < 16; k++)
        {
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v[k], one); HVX_Vector lo = sfv(Q6_V_lo_W(p)), hi = sfv(Q6_V_hi_W(p));
            e = sfv(Q6_Vqf32_vadd_VsfVsf(e, lo)); e = sfv(Q6_Vqf32_vadd_VsfVsf(e, hi));
            f = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(lo, lo), f)); f = sfv(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(hi, hi), f));
        }
        float fe[32] __attribute__((aligned(128))), ff[32] __attribute__((aligned(128))); *(HVX_Vector *)fe = e; *(HVX_Vector *)ff = f;
        for (uint32_t j = 0; j < 32; j++) { const uint32_t g = (b * 32 + j) / a->gs; s[g] += fe[j]; q[g] += ff[j]; }
    }
    qurt_mutex_lock(&a->mu); for (int g = 0; g < 32; g++) { a->sum[g] += s[g]; a->sq[g] += q[g]; } qurt_mutex_unlock(&a->mu);   /* no spin lock: QuRT doesn't time-slice, spinners could keep a preempted holder off every thread */
}
static void vgn_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vgn_t *a = (vgn_t *)a_; const uint32_t cb = a->c / 32;
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < cb; b++)
    {
        const HVX_Vector *xi = (const HVX_Vector *)blk((uint16_t *)a->x, a->c, t, b); HVX_Vector *yo = (HVX_Vector *)blk(a->y, a->c, t, b);
        if (!(b & 3) && (t + 1 < t1 || b + 4 < cb)) vk_pf(blk((uint16_t *)a->x, a->c, t, b + 4), 8192);
        for (int k = 0; k < 16; k++)
        {
            HVX_Vector n = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vsub_VhfVhf(xi[k], a->mean[b])), a->rstd[b]));
            n = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(n, a->gv[b])), a->bv[b]));
            yo[k] = a->silu ? hf(Q6_Vqf16_vmpy_VhfVhf(n, sigmoid_hf(n))) : n;
        }
    }
}
static void vk_gn(const uint16_t *x, uint16_t *y, uint32_t h, uint32_t w, uint32_t vw, uint32_t ch, const float *gam, const float *bet, float eps, int silu)
{
    static vgn_t a; memset(&a, 0, sizeof(a)); a.x = x; a.y = y; a.c = ch; a.gs = ch / 32; a.silu = silu;
    const uint32_t tiles = h * (w / 32), cb = ch / 32; qurt_mutex_init(&a.mu); tiles_parallel_range(vgn_stats, &a, 0, tiles); qurt_mutex_destroy(&a.mu);
    const double n = (double)h * vw * a.gs; float me[32], rs[32];
    for (int g = 0; g < 32; g++) { const double mm = a.sum[g] / n, va = a.sq[g] / n - mm * mm; me[g] = (float)mm; rs[g] = (float)(1.0 / sqrt((va > 0 ? va : 0) + eps)); }
    static HVX_Vector mean[128], rstd[128], gv[128], bv[128];                   /* not on the small RPC stack; c <= 4096 */
    for (uint32_t b = 0; b < cb; b++)
    {
        float tm[32], tr[32]; for (uint32_t j = 0; j < 32; j++) { const uint32_t g = (b * 32 + j) / a.gs; tm[j] = me[g]; tr[j] = rs[g]; }
        mean[b] = colvec(tm); rstd[b] = colvec(tr); gv[b] = colvec(gam + b * 32); bv[b] = colvec(bet + b * 32);
    }
    a.mean = mean; a.rstd = rstd; a.gv = gv; a.bv = bv; tiles_parallel_range(vgn_tiles, &a, 0, tiles);
    vk_pad(y, h, w, vw, ch);
}
/* y = silu(x), the last layer's: after the reference's norm_out stage */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t c; } vsilu_t;
static void vsilu_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vsilu_t *a = (vsilu_t *)a_;
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < a->c / 32; b++)
    {
        const HVX_Vector *xi = (const HVX_Vector *)blk((uint16_t *)a->x, a->c, t, b); HVX_Vector *yo = (HVX_Vector *)blk(a->y, a->c, t, b);
        for (int k = 0; k < 16; k++)
            yo[k] = hf(Q6_Vqf16_vmpy_VhfVhf(xi[k], sigmoid_hf(xi[k])));
    }
}
static void vk_silu(const uint16_t *x, uint16_t *y, uint32_t tiles, uint32_t c) { vsilu_t a = { x, y, c }; tiles_parallel_range(vsilu_tiles, &a, 0, tiles); }
/* y += x (with x null: y += the columns' bias) */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t c; const HVX_Vector *bv; } vadd_t;
static void vadd_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vadd_t *a = (vadd_t *)a_; const uint32_t cb = a->c / 32;
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < cb; b++)
    {
        HVX_Vector *y = (HVX_Vector *)blk(a->y, a->c, t, b); const HVX_Vector *x = a->x ? (const HVX_Vector *)blk((uint16_t *)a->x, a->c, t, b) : 0;
        if (!(b & 3) && (t + 1 < t1 || b + 4 < cb)) { vk_pf(blk(a->y, a->c, t, b + 4), 8192); if (x) vk_pf(blk((uint16_t *)a->x, a->c, t, b + 4), 8192); }
        for (int k = 0; k < 16; k++) y[k] = hf(Q6_Vqf16_vadd_VhfVhf(y[k], x ? x[k] : a->bv[b]));
    }
}
static void vk_add(const uint16_t *x, uint16_t *y, uint32_t tiles, uint32_t c) { vadd_t a = { x, y, c, 0 }; tiles_parallel_range(vadd_tiles, &a, 0, tiles); }
static void vk_bias(uint16_t *y, uint32_t tiles, uint32_t c, const float *bias)
{ static HVX_Vector bv[128]; for (uint32_t b = 0; b < c / 32; b++) bv[b] = colvec(bias + b * 32); vadd_t a = { 0, y, c, bv }; tiles_parallel_range(vadd_tiles, &a, 0, tiles); }
/* y [2h, w x c] from x [h, sw x c], nearest: a tile's even column takes the first 16 pixels of the tile under it, the odd the last; pixel q is half q & 1 of vector q / 2, doubled */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t w, sw, c; } vup_t;
static void vup_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vup_t *a = (vup_t *)a_; const uint32_t wt = a->w / 32, swt = a->sw / 32, cb = a->c / 32; const HVX_Vector lo = Q6_V_vsplat_R(0xffff), hi = Q6_V_vsplat_R(0xffff0000);
    for (uint32_t t = t0; t < t1; t++)
    {
        const uint32_t r = t / wt, tc = t % wt, ts = (r / 2) * swt + tc / 2, off = tc & 1 ? 16 : 0;
        for (uint32_t b = 0; b < cb; b++)
        {
            const HVX_Vector *s = (const HVX_Vector *)blk((uint16_t *)a->x, a->c, ts, b); HVX_Vector *d = (HVX_Vector *)blk(a->y, a->c, t, b);
            for (uint32_t k = 0; k < 16; k++)
            {
                const uint32_t q = off + k; const HVX_Vector v = s[q / 2];
                d[k] = q & 1 ? Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v, 16), Q6_V_vand_VV(v, hi)) : Q6_V_vor_VV(Q6_V_vand_VV(v, lo), Q6_Vw_vasl_VwR(v, 16));
            }
        }
    }
}
static void vk_up(const uint16_t *x, uint16_t *y, uint32_t h, uint32_t sw, uint32_t w, uint32_t c) { vup_t a = { x, y, w, sw, c }; tiles_parallel_range(vup_tiles, &a, 0, 2 * h * (w / 32)); }

/* ------------------------------------------------------------------ the decoder's stages */
/* y = silu(x / rms over the channels * gamma), pixel by pixel: the Qwen-Image VAE's norm (RMS_norm: F.normalize * sqrt(c) * gamma) */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t c; const HVX_Vector *gv; int silu; } vrn_t;
static void vrn_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vrn_t *a = (vrn_t *)a_; const uint32_t cb = a->c / 32;
    for (uint32_t t = t0; t < t1; t++)
    {
        if (t + 1 < t1) vk_pf(blk((uint16_t *)a->x, a->c, t + 1, 0), (size_t)cb * BLK);
        HVX_Vector rstd[16]; row_stats(blk((uint16_t *)a->x, a->c, t, 0), cb, (float)a->c, 1e-12f, 0, rstd);
        for (uint32_t b = 0; b < cb; b++)
        {
            const HVX_Vector *xi = (const HVX_Vector *)blk((uint16_t *)a->x, a->c, t, b); HVX_Vector *yo = (HVX_Vector *)blk(a->y, a->c, t, b);
            for (int k = 0; k < 16; k++)
            {
                const HVX_Vector n = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(xi[k], rstd[k])), a->gv[b]));
                yo[k] = a->silu ? hf(Q6_Vqf16_vmpy_VhfVhf(n, sigmoid_hf(n))) : n;
            }
        }
    }
}
typedef struct { const uint16_t *w; const float *p; uint16_t *a, *b, *c; uint32_t h, wd, vw; uint64_t th; int run; uint64_t tp, tb, tg, ta; int qw; size_t cap; } vd_t;   /* cap: the fewest elements a, b, c hold (vd_attn's room), 0 unknown; tp, tb, tg, ta: products, bias and padding, norm, add (us) */
static inline uint32_t vd_cp(uint32_t co) { return (co + 63) & ~63u; }   /* the N a product makes: co in whole groups of 64 */
static int g_convsplit;   /* SDXL's calls (kblock_xop): wide 3 x 3 convolutions in three K parts */
static int g_vdadd;   /* vd_conv: OUT_ADD into y */
static const float *g_vdb2;   /* vd_conv: a second bias added to the first before it becomes fp16 (SDXL's time embedding) */
/* y = a kernel-size-kz convolution of x ([h w x ci], activation layout), with its bias. w: co x 9 ci (kz 3, neighbours in the order row by row) or co x ci
 * in the weight layout; a whole number of 9 ci or ci columns goes in, cp comes out and is closed up to co */
static void vd_conv(vd_t *d, const uint16_t *x, uint16_t *y, uint32_t ci, uint32_t co, int kz)
{
    const uint32_t K = kz == 3 ? 9 * ci : ci, cp = vd_cp(co); const uint16_t *w = d->w; const float *bias = d->p;
    d->w = WP(w, (size_t)K * cp); d->p += co; if (!d->run) return;
    g_D = ci; uint32_t kcb = K / 32 < 16 ? K / 32 : 16; while ((K / 32) % kcb) kcb--;
    const int dn = g_xdn && kz == 3; const uint32_t oh = dn ? (d->h + 1) / 2 : d->h, ow = dn ? d->wd / 2 : d->wd;   /* the stride-2 form */
    act_fill_t cf = dn ? dconv_fill : conv_fill;
    g_kc = kcb * 32; const uint32_t M = oh * ow;
    static HVX_Vector bv[128]; float bz[32];                                    /* the bias on the output blocks, in VTCM; past co 0 */
    for (uint32_t b = 0; b < cp / 32; b++) { for (uint32_t j = 0; j < 32; j++) bz[j] = b * 32 + j < co ? bias[b * 32 + j] + (g_vdb2 ? g_vdb2[b * 32 + j] : 0.f) : 0.f; bv[b] = colvec(bz); }
    wset_t ws = { w, cp, y, g_vdadd ? OUT_ADD : OUT_ACT, 0, 0, bv }; g_outc = co == cp ? 0 : co; uint64_t c0 = HAP_perf_get_time_us();
    if (kz == 3 && g_convsplit && ci > 640)
    {   /* a wide input (SDXL's up blocks, 960..2560 channels) in the kernel's three rows, K = 3 ci each, added up: with all nine a 64-column
           weight group was 2.9 MB twice over in VTCM, one 32-pixel tile fitted beside it, and the 59 MB of weights were read 32 times
           (half of the resnets' conv time waiting for weights) */
        const uint32_t K3 = 3 * ci; kcb = K3 / 32 < 16 ? K3 / 32 : 16; while ((K3 / 32) % kcb) kcb--; g_kc = kcb * 32;
        for (int r = 0; r < 3; r++)
        {
            conv_arg_t a = { x, d->h, d->wd, ci, r + 1 }; wset_t w3 = { w, cp, y, r ? OUT_ADD : OUT_ACT, K, (uint32_t)r * K3, r ? 0 : bv };
            product_multi(0, cf, &a, M, K3, &w3, 1, 128, 0, &d->th);
        }
    }
    else if (kz == 3) { conv_arg_t a = { x, d->h, d->wd, ci, 0 }; product_multi(0, cf, &a, M, K, &ws, 1, 128, 0, &d->th); }
    else product_multi(x, 0, 0, M, K, &ws, 1, 128, 0, &d->th);
    g_kc = 0; g_outc = 0; uint64_t c1 = HAP_perf_get_time_us(); d->tp += c1 - c0;
    vk_pad(y, oh, ow, dn ? (d->vw + 1) / 2 : d->vw, co); d->tb += HAP_perf_get_time_us() - c1;
}
static void vd_gn(vd_t *d, const uint16_t *x, uint16_t *y, uint32_t c, float eps, int silu)
{
    if (d->qw)
    {
        const float *g = d->p; d->p += c; if (!d->run) return;
        const uint64_t c0 = HAP_perf_get_time_us(); static HVX_Vector gv[16]; for (uint32_t b = 0; b < c / 32; b++) gv[b] = colvec(g + b * 32);
        vrn_t a = { x, y, c, gv, silu }; tiles_parallel_range(vrn_tiles, &a, 0, d->h * (d->wd / 32)); vk_pad(y, d->h, d->wd, d->vw, c); d->tg += HAP_perf_get_time_us() - c0; return;
    }
    const float *g = d->p; d->p += 2 * c; if (d->run) { uint64_t c0 = HAP_perf_get_time_us(); vk_gn(x, y, d->h, d->wd, d->vw, c, g, g + c, eps, silu); d->tg += HAP_perf_get_time_us() - c0; }
}
/* a = a resnet of a: norm, silu, conv, norm, silu, conv, plus a or its 1 x 1 shortcut */
static void vd_res(vd_t *d, uint32_t ci, uint32_t co)
{
    /* with ci == co the residual is a itself: the second convolution adds into it in its post (a + (conv + bias), vk_add's
       sum the other way round), its padding zeroed after as before; not with the kernel's rows split (their sum's order would change) */
    const int vr = (g_opt & OPT_VRES) && ci == co && vd_cp(co) == co && !(g_convsplit && co > 640);
    vd_gn(d, d->a, d->b, ci, 1e-6f, 1); vd_conv(d, d->b, d->c, ci, co, 3);
    vd_gn(d, d->c, d->b, co, 1e-6f, 1);
    if (vr) { g_vdadd = 1; vd_conv(d, d->b, d->a, co, co, 3); g_vdadd = 0; return; }
    vd_conv(d, d->b, d->c, co, co, 3);
    if (ci != co) vd_conv(d, d->a, d->b, ci, co, 1);
    if (d->run) { uint64_t c0 = HAP_perf_get_time_us(); vk_add(ci != co ? d->b : d->a, d->c, d->h * (d->wd / 32), co); d->ta += HAP_perf_get_time_us() - c0; }
    if (d->run) { uint16_t *t = d->a; d->a = d->c; d->c = t; }
}
/* ------------------------------------------------------------------ the middle attention: one head of all the channels, every token a key. Its 128-per-head path doesn't fit (attention()), so
 * it goes through the products: scores q K^T in slabs of rows, softmax, then P V. K^T and V are made in the weight layout from the activation layout: an element of K at (n, k) sits in block (n / 32, k / 32)
 * at (n % 32, k % 32) there and at (k % 32, n % 32) in the weights, so each block is transposed; V keeps its blocks' insides and swaps their order. */
typedef struct { uint16_t *x; uint16_t *y; uint32_t cb, nk; float f; } va_t;
static void va_kt(void *a_, uint32_t t0, uint32_t t1)
{
    va_t *a = (va_t *)a_;
    for (uint32_t b = t0; b < t1; b++)
    {
        uint16_t *p = a->x + (size_t)b * BH;
        for (uint32_t i = 0; i < 32; i++) for (uint32_t j = 0; j < i; j++)
        { uint16_t *u = p + (i / 2) * 64 + j * 2 + (i & 1), *v = p + (j / 2) * 64 + i * 2 + (j & 1); const uint16_t t = *u; *u = *v; *v = t; }
    }
}
static void va_scale(void *a_, uint32_t t0, uint32_t t1)
{
    va_t *a = (va_t *)a_; HVX_Vector *v = (HVX_Vector *)a->x; const HVX_Vector f = splat_hf(a->f);
    for (uint32_t i = t0; i < t1; i++) { HVX_Vector *b = v + (size_t)i * 16; for (int k = 0; k < 16; k++) b[k] = hf(Q6_Vqf16_vmpy_VhfVhf(b[k], f)); }
}
typedef struct { uint16_t *s; uint32_t nk, wt, vw; HVX_Vector *rs; } vsm_t;
static void vsm_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vsm_t *a = (vsm_t *)a_; const uint32_t mk[4] = { 0, 0, 0, 0 };
    for (uint32_t t = t0; t < t1; t++)
    {
        HVX_Vector *v = (HVX_Vector *)blk(a->s, a->nk * 32, t, 0);
        if (a->vw != a->wt * 32)
        {   /* the padding's keys out of the softmax: key block b is column b % wt of the picture's tiles, lanes 2 j and 2 j + 1 key j */
            const uint32_t last = a->vw / 32, r = a->vw % 32; uint16_t lm[64] __attribute__((aligned(128))); for (uint32_t l = 0; l < 64; l++) lm[l] = (l >> 1) >= r ? 0xffff : 0;
            const HVX_VectorPred q = Q6_Q_vand_VR(*(HVX_Vector *)lm, 0x01010101); const HVX_Vector lowest = Q6_Vh_vsplat_R(0xfbff);
            for (uint32_t b = 0; b < a->nk; b++)
            {
                const uint32_t c = b % a->wt; if (c < last) continue;
                for (int k = 0; k < 16; k++) v[b * 16 + k] = c > last || !r ? lowest : Q6_V_vmux_QVV(q, lowest, v[b * 16 + k]);
            }
        }
        softmax_rows((__fp16 *)v, a->nk, mk, a->rs + (size_t)t * 16, 0, 16, -1);
    }
}
/* the scores' key blocks b0.. of a slab into VTCM's tiles (act_fill_t), for P V in pieces of keys: a whole row of 64 K keys is more weights than VTCM holds */
typedef struct { const uint16_t *s; uint32_t nk, b0, nb; uint8_t *dst; uint32_t m0; } vpf_t;
static void vpf_tiles(void *a_, uint32_t t0, uint32_t t1)
{ vpf_t *f = (vpf_t *)a_; for (uint32_t t = t0; t < t1; t++) hvx_copy(f->dst + (size_t)t * f->nb * BLK, (const uint8_t *)(f->s + ((size_t)(f->m0 + t) * f->nk + f->b0) * BH), f->nb * BLK); }
static void vpf_fill(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles)
{ vpf_t f = *(vpf_t *)arg; f.dst = v_act; f.m0 = m0; tiles_parallel_range(vpf_tiles, &f, 0, tiles); }
typedef struct { uint16_t *x; uint16_t *y; uint32_t cb, nk, b0, nb; } vvs_t;
static void vvs_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vvs_t *a = (vvs_t *)a_;
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < a->cb; b++)
        hvx_copy((uint8_t *)(a->y + ((size_t)b * a->nb + t) * BH), (const uint8_t *)(a->x + ((size_t)(a->b0 + t) * a->cb + b) * BH), BLK);
}
typedef struct { uint16_t *o; uint32_t cb; const HVX_Vector *rs; } vdiv_t;
static void vdiv_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vdiv_t *a = (vdiv_t *)a_;
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < a->cb; b++)
    { HVX_Vector *d = (HVX_Vector *)(a->o + ((size_t)t * a->cb + b) * BH); for (int k = 0; k < 16; k++) d[k] = hf(Q6_Vqf16_vmpy_VhfVhf(d[k], a->rs[(size_t)t * 16 + k])); }
}
/* a = a + attention(a), ch channels. b and c are scratch. */
static void vd_attn(vd_t *d, uint32_t ch)
{
    const uint32_t M = d->h * d->wd, cb = ch / 32, nk = M / 32; const size_t A = (size_t)M * ch;
    uint32_t SL = 512;   /* query rows a slab: as many as c's room after q, k, v, V's layout and the output holds scores for (K read 8x fewer times at 1024^2) */
    if ((g_opt & OPT_VSL) && d->cap > 5 * A) { const size_t r = (d->cap - 5 * A) / M; SL = r >= M ? M : (uint32_t)(r & ~(size_t)511); if (SL < 512) SL = 512; if (SL > 4096) SL = 4096; }
    uint16_t *qb = d->c, *kb = qb + A, *vb = kb + A, *vw = vb + A, *ob = vw + A, *sb = ob + A; va_t va;
    vd_gn(d, d->a, d->b, ch, 1e-6f, 0);
    vd_conv(d, d->b, qb, ch, ch, 1); vd_conv(d, d->b, kb, ch, ch, 1); vd_conv(d, d->b, vb, ch, ch, 1);
    if (d->run)
    {
        va = (va_t){ qb, 0, cb, nk, 1.44269504f / sqrtf((float)ch) };      /* softmax_rows' exponential is 2^x */ tiles_parallel_range(va_scale, &va, 0, nk * cb);
        va = (va_t){ kb, 0, cb, nk, 0.f }; tiles_parallel_range(va_kt, &va, 0, nk * cb);
        static HVX_Vector *rs; rs = (HVX_Vector *)memalign(128, (size_t)(SL / 32) * 16 * 128);
        const int sum0 = g_hmxsum; uint8_t *bias0 = g_bias;
        const uint32_t VKCN = (g_opt & OPT_VKC) ? 64 : 512;   /* the key chunk of P V in blocks of 32 keys (the partial sums add up in fp16 through OUT_ADD) */
        const int vda = (g_opt & OPT_VDA) != 0;   /* V reordered once, chunk b0 at vw + b0 * cb blocks (vw holds A = nk * cb blocks) */
        if (vda) for (uint32_t b0 = 0; b0 < nk; b0 += VKCN)
        {
            const uint32_t nb = nk - b0 < VKCN ? nk - b0 : VKCN;
            vvs_t vs = { vb, vw + (size_t)b0 * cb * BH, cb, nk, b0, nb }; tiles_parallel_range(vvs_tiles, &vs, 0, nb);
        }
        for (uint32_t m0 = 0; m0 < M; m0 += SL)
        {
            const uint32_t ms = M - m0 < SL ? M - m0 : SL;
            g_D = ch; g_kc = cb * 32; { wset_t ws = { kb, M, sb, OUT_ACT }; product_multi(qb + (size_t)m0 * ch, 0, 0, ms, ch, &ws, 1, 128, 0, &d->th); } g_kc = 0;
            g_hmxsum = 0; { vsm_t sm = { sb, nk, d->wd / 32, d->vw, rs }; tiles_parallel_range(vsm_tiles, &sm, 0, ms / 32); } g_hmxsum = sum0;
            g_bias = g_bias_av;
            for (uint32_t b0 = 0; b0 < nk; b0 += VKCN)
            {
                const uint32_t nb = nk - b0 < VKCN ? nk - b0 : VKCN; uint32_t kcb = nb < 16 ? nb : 16; while (nb % kcb) kcb--;
                uint16_t *vc = vda ? vw + (size_t)b0 * cb * BH : vw;
                if (!vda) { vvs_t vs = { vb, vw, cb, nk, b0, nb }; tiles_parallel_range(vvs_tiles, &vs, 0, nb); }
                vpf_t pf = { sb, nk, b0, nb, 0, 0 }; g_D = nb * 32 < MAXD ? nb * 32 : MAXD; g_kc = kcb * 32;
                wset_t ws = { vc, ch, ob + (size_t)m0 * ch, b0 ? OUT_ADD : OUT_ACT }; product_multi(0, vpf_fill, &pf, ms, nb * 32, &ws, 1, 128, 0, &d->th);
            }
            g_bias = bias0; g_kc = 0;
            { vdiv_t dv = { ob + (size_t)m0 * ch, cb, rs }; tiles_parallel_range(vdiv_tiles, &dv, 0, ms / 32); }
        }
        free(rs);
    }
    vd_conv(d, ob, d->b, ch, ch, 1);
    if (d->run) { vk_add(d->a, d->b, d->h * (d->wd / 32), ch); uint16_t *t = d->a; d->a = d->b; d->b = t; }
}
/* w, prm from the host's converter; x: the 32-channel input (or stage start's input); scr: three images of 192 channels at the final size. y: stage stop's output, or the picture's 64 channels.
 * Only start..stop run, the others just pass their weights by, so that each stage can be checked from its reference input. */
int kblock_vdec(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen,
                int h, int wd, int vw, int start, int stop, uint32_t *times, int timesLen)
{   /* no g_need8: its convs and attention size their tiles from VTCM_BYTES (product_multi, attention), so it runs on 4 MB too */
    g_vkv = stop >> 16; stop &= 255;
    /* h, wd: the input's; each of the three upsamples doubles h and vw, wd the next multiple of 32 */
    const size_t part = (size_t)8 * h * ((8 * vw + 31) & ~31) * 192;
    if (wd % 32 || (uint32_t)vw > (uint32_t)wd || (size_t)scrLen < 2 * part + part / 2 || timesLen < 32 || stop > 21 || start > stop) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_tqk = g_tpost = 0; uint16_t *const small = scr; vd_t d = { w, prm, small, scr + part / 2, scr + part / 2 + part,   /* one image of 96 channels at the final size, two of 192: the last levels' only full ones */ (uint32_t)h, (uint32_t)wd, (uint32_t)vw, 0, 0 }; d.cap = part / 2;
    d.qw = (g_vkv >> 8) & 1;
    static const uint16_t QOUTC[22] = { 32, 384, 384, 384, 384, 384, 384, 384, 192, 384, 384, 384, 192, 192, 192, 192, 96, 96, 96, 96, 96, 64 };
    static const uint16_t OUTC[22] = { 32, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 192, 192, 192, 192, 96, 96, 96, 96, 64 };
    (void)wLen; (void)prmLen; uint64_t t = HAP_perf_get_time_us(); int k = 0;
#define VD_STAGE(...) { d.run = k >= start && k <= stop; if (d.run && k == start) hvx_copy((uint8_t *)d.a, (const uint8_t *)x, (size_t)xLen * 2); __VA_ARGS__; \
                         if (d.run) { const uint64_t n = HAP_perf_get_time_us(); times[k] = (uint32_t)(n - t); t = n; } if (k == stop) goto done; k++; }
    if (d.qw)
    {
        #define VQ_UP(ci_, co_, last_) VD_STAGE({ const uint32_t sw = d.wd, nw = (2 * d.vw + 31) & ~31u; \
            if (d.run && (last_) && d.a != small) { hvx_copy((uint8_t *)small, (const uint8_t *)d.a, (size_t)d.h * d.wd * (ci_) * 2); uint16_t *big = d.a, *other = d.b == small ? d.c : d.b; d.a = small; d.b = big; d.c = other; } \
            if (d.run) { vk_up(d.a, d.c, d.h, sw, nw, (ci_)); uint16_t *s = d.a; d.a = d.c; d.c = s; if (last_) { s = d.b; d.b = d.c; d.c = s; } } \
            d.h *= 2; d.wd = nw; d.vw *= 2; if (d.run) vk_pad(d.a, d.h, d.wd, d.vw, (ci_)); vd_conv(&d, d.a, d.c, (ci_), (co_), 3); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; if (last_) { s = d.b; d.b = d.c; d.c = s; } } })
        VD_STAGE(vd_conv(&d, d.a, d.c, 32, 32, 1); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; })
        VD_STAGE(vd_conv(&d, d.a, d.c, 32, 384, 3); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; })
        VD_STAGE(vd_res(&d, 384, 384)) VD_STAGE(vd_attn(&d, 384)) VD_STAGE(vd_res(&d, 384, 384))
        VD_STAGE(vd_res(&d, 384, 384)) VD_STAGE(vd_res(&d, 384, 384)) VD_STAGE(vd_res(&d, 384, 384))
        VQ_UP(384, 192, 0)
        VD_STAGE(vd_res(&d, 192, 384)) VD_STAGE(vd_res(&d, 384, 384)) VD_STAGE(vd_res(&d, 384, 384))
        VQ_UP(384, 192, 0)
        VD_STAGE(vd_res(&d, 192, 192)) VD_STAGE(vd_res(&d, 192, 192)) VD_STAGE(vd_res(&d, 192, 192))
        VQ_UP(192, 96, 1)
        VD_STAGE(vd_res(&d, 96, 96)) VD_STAGE(vd_res(&d, 96, 96)) VD_STAGE(vd_res(&d, 96, 96))
        VD_STAGE(vd_gn(&d, d.a, d.b, 96, 1e-12f, 1); if (d.run) { uint16_t *s = d.a; d.a = d.b; d.b = s; })
        VD_STAGE(vd_conv(&d, d.a, d.c, 96, 64, 3); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; })
    }
    VD_STAGE(vd_conv(&d, d.a, d.c, 32, 32, 1); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; })
    VD_STAGE(vd_conv(&d, d.a, d.c, 32, 384, 3); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; })
    VD_STAGE(vd_res(&d, 384, 384))
    VD_STAGE(vd_attn(&d, 384))
    VD_STAGE(vd_res(&d, 384, 384))
    static const uint16_t CO[4] = { 384, 384, 192, 96 }; uint32_t ci = 384;
    for (int i = 0; i < 4; i++)
    {
        VD_STAGE(vd_res(&d, ci, CO[i])) VD_STAGE(vd_res(&d, CO[i], CO[i])) VD_STAGE(vd_res(&d, CO[i], CO[i]))
        ci = CO[i]; if (i == 3) break;
        VD_STAGE({ const uint32_t sw = d.wd, nw = (2 * d.vw + 31) & ~31u;
                   if (d.run && i == 2 && d.a != small)
                   {   /* the last upsample: its input into the small image, so that the two large ones hold the 192 channels at full size */
                       hvx_copy((uint8_t *)small, (const uint8_t *)d.a, (size_t)d.h * d.wd * ci * 2); uint16_t *big = d.a, *other = d.b == small ? d.c : d.b; d.a = small; d.b = big; d.c = other;
                   }
                   if (d.run) { vk_up(d.a, d.c, d.h, sw, nw, ci); uint16_t *s = d.a; d.a = d.c; d.c = s; if (i == 2) { s = d.b; d.b = d.c; d.c = s; } }
                   d.h *= 2; d.wd = nw; d.vw *= 2; if (d.run) vk_pad(d.a, d.h, d.wd, d.vw, ci); vd_conv(&d, d.a, d.c, ci, ci, 3); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; if (i == 2) { s = d.b; d.b = d.c; d.c = s; } } })
    }
    VD_STAGE(vd_gn(&d, d.a, d.b, 96, 1e-6f, 0); if (d.run) { uint16_t *s = d.a; d.a = d.b; d.b = s; })
    VD_STAGE(if (d.run) vk_silu(d.a, d.b, d.h * (d.wd / 32), 96); vd_conv(&d, d.b, d.c, 96, 64, 3); if (d.run) { uint16_t *s = d.a; d.a = d.c; d.c = s; })     /* 64 columns, the first 3 the picture */
done:;
#undef VD_STAGE
    const uint32_t oc = (d.qw ? QOUTC : OUTC)[k], M = d.h * d.wd;
    if ((size_t)yLen >= (size_t)M * oc) hvx_copy((uint8_t *)y, (const uint8_t *)d.a, (size_t)M * oc * 2);
    times[22] = (uint32_t)d.th; times[23] = M; times[24] = (uint32_t)d.tp; times[25] = 0;   /* (unused slot: stays 0) */ times[26] = (uint32_t)d.tb; times[27] = (uint32_t)d.tg; times[28] = (uint32_t)d.ta; times[29] = (uint32_t)g_tqk; times[30] = (uint32_t)g_tpost;
    eng_end(&eg); return 0;
}

/* ------------------------------------------------------------------ Real-ESRGAN's SRVGGNetCompact: n 3 x 3 convolutions of 64 channels, a PReLU between them. x [h, wd x 32]: the picture in channels 0..2; y [h, wd x 64]: the last convolution's output (the pixel shuffle is the host's). */
typedef struct { uint16_t *y; const HVX_Vector *av; } vpr_t;
static void vpr_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vpr_t *a = (vpr_t *)a_; const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < 2; b++)
    {
        HVX_Vector *v = (HVX_Vector *)blk(a->y, 64, t, b); const HVX_Vector al = a->av[b];
        if (t + 1 < t1) vk_pf(blk(a->y, 64, t + 1, b), BLK);
        for (int k = 0; k < 16; k++) { const HVX_Vector x = v[k]; v[k] = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(zero, x), hf(Q6_Vqf16_vmpy_VhfVhf(x, al)), x); }
    }
}
int kblock_esr(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen,
               int h, int wd, int vw, int n, uint32_t *times, int timesLen)
{ g_need8 = 1;   /* VTCM laid out for 8 MB */
    const size_t M = (size_t)h * wd;
    if (wd % 32 || (uint32_t)vw > (uint32_t)wd || n < 2 || (size_t)xLen < M * 32 || (size_t)yLen < M * 64 || (size_t)scrLen < 2 * M * 64 || timesLen < 4) return AEE_EBADPARM;
    (void)wLen; (void)prmLen; uint64_t t0 = HAP_perf_get_time_us();
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_vkv = 0; vd_t d; memset(&d, 0, sizeof(d)); d.w = w; d.p = prm; d.h = h; d.wd = wd; d.vw = vw; d.run = 1;
    uint16_t *a = scr, *c = scr + M * 64; const uint16_t *in = x; uint64_t tpr = 0;
    for (int i = 0; i < n; i++)
    {
        uint16_t *o = i + 1 == n ? y : c; vd_conv(&d, in, o, i ? 64 : 32, 64, 3);
        if (i + 1 < n)
        {
            const uint64_t c0 = HAP_perf_get_time_us(); static HVX_Vector av[2];
            for (int b = 0; b < 2; b++) av[b] = colvec(d.p + b * 32);
            d.p += 64; vpr_t pa = { o, av }; tiles_parallel_range(vpr_tiles, &pa, 0, (uint32_t)(M / 32)); vk_pad(o, d.h, d.wd, d.vw, 64);
            tpr += HAP_perf_get_time_us() - c0; in = o; uint16_t *s = a; a = o; c = s;
        }
    }
    eng_end(&eg);
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t0); times[1] = (uint32_t)d.th; times[2] = (uint32_t)d.tp; times[3] = (uint32_t)tpr;
    return 0;
}

/* ------------------------------------------------------------------ the Qwen-Image VAE's encoder, for img2img. */
/* y [h / 2, nw x c] from x [h, sw x c]: pixel (2 r + 1, 2 q + 1), what a stride-2 convolution after a zero pad of one on the right and below keeps of the same convolution run on every pixel.
 * An odd pixel is the high half of its vector: a word of y is the high halves of two vectors of x */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t h, sw, nw, c; } vdn_t;
static void vdn_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    vdn_t *a = (vdn_t *)a_; const uint32_t swt = a->sw / 32, nwt = a->nw / 32, cb = a->c / 32; const HVX_Vector hi = Q6_V_vsplat_R(0xffff0000);
    for (uint32_t t = t0; t < t1; t++)
    {
        const uint32_t r = t / nwt, tc = t % nwt, sr = 2 * r + 1; HVX_Vector *d = (HVX_Vector *)blk(a->y, a->c, t, 0);
        if (sr >= a->h) { memset(d, 0, (size_t)cb * BLK); continue; }
        for (uint32_t b = 0; b < cb; b++)
            for (uint32_t k = 0; k < 16; k++)
            {
                const uint32_t q0 = 32 * tc + 2 * k, i0 = 2 * q0 + 1, i1 = i0 + 2; HVX_Vector v0 = Q6_V_vzero(), v1 = Q6_V_vzero();
                if (i0 < a->sw) v0 = ((const HVX_Vector *)blk((uint16_t *)a->x, a->c, sr * swt + (i0 >> 5), b))[(i0 & 31) >> 1];
                if (i1 < a->sw) v1 = ((const HVX_Vector *)blk((uint16_t *)a->x, a->c, sr * swt + (i1 >> 5), b))[(i1 & 31) >> 1];
                d[b * 16 + k] = Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(v0, 16), Q6_V_vand_VV(v1, hi));
            }
    }
}
/* x [h, wd x 32]: the picture in channels 0..2; y [h / 8, wd' x 32]: mean in channels 0..15, wd' the eighth's width up to a multiple of 32. scr: three images of 128 channels */
int kblock_qenc(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen,
                int h, int wd, int vw, uint32_t *times, int timesLen)
{ g_need8 = 1;   /* VTCM laid out for 8 MB */
    const size_t M = (size_t)h * wd, part = M * 128; const uint32_t ow = ((uint32_t)vw / 8 + 31) & ~31u;
    if (wd % 32 || h % 8 || vw % 8 || (uint32_t)vw > (uint32_t)wd || (size_t)xLen < M * 32 || (size_t)yLen < (size_t)(h / 8) * ow * 32 || (size_t)scrLen < 3 * part || timesLen < 4) return AEE_EBADPARM;
    (void)wLen; (void)prmLen; const uint64_t t0 = HAP_perf_get_time_us();
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_vkv = 0; vd_t d; memset(&d, 0, sizeof(d)); d.w = w; d.p = prm; d.a = scr; d.b = scr + part; d.c = scr + 2 * part; d.h = h; d.wd = wd; d.vw = vw; d.run = 1; d.qw = 1;
    #define QE_SWAP { uint16_t *s_ = d.a; d.a = d.c; d.c = s_; }
    #define QE_DOWN(c_) { vd_conv(&d, d.a, d.c, (c_), (c_), 3); const uint32_t nw_ = (d.vw / 2 + 31) & ~31u; vdn_t da_ = { d.c, d.a, d.h, d.wd, nw_, (c_) }; const uint32_t nh_ = d.h / 2; tiles_parallel_range(vdn_tiles, &da_, 0, nh_ * (nw_ / 32)); d.h = nh_; d.wd = nw_; d.vw /= 2; vk_pad(d.a, d.h, d.wd, d.vw, (c_)); }
    vd_conv(&d, x, d.c, 32, 96, 3); QE_SWAP
    vd_res(&d, 96, 96); vd_res(&d, 96, 96); QE_DOWN(96)
    vd_res(&d, 96, 192); vd_res(&d, 192, 192); QE_DOWN(192)
    vd_res(&d, 192, 384); vd_res(&d, 384, 384); QE_DOWN(384)
    vd_res(&d, 384, 384); vd_res(&d, 384, 384);
    vd_res(&d, 384, 384); vd_attn(&d, 384); vd_res(&d, 384, 384);
    vd_gn(&d, d.a, d.b, 384, 1e-12f, 1); { uint16_t *s_ = d.a; d.a = d.b; d.b = s_; }
    vd_conv(&d, d.a, d.c, 384, 32, 3); QE_SWAP
    vd_conv(&d, d.a, y, 32, 32, 1);
    eng_end(&eg);
    times[0] = (uint32_t)(HAP_perf_get_time_us() - t0); times[1] = (uint32_t)d.th; times[2] = (uint32_t)d.tp; times[3] = (uint32_t)d.tg;
    return 0;
}
