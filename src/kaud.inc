/* kaud.inc (included by kblock_imp.c) -- YuE2's Oobleck decoder on the DSP: signals [rows x c] in the activation layout, a row a time step. A convolution of kernel 7 is a product whose
 * activation row is the 7 neighbours' channels side by side; a transposed one of stride s and kernel 2 s is s products of x[t] | x[t - 1], their rows interleaved. */
typedef struct { const uint16_t *x; uint32_t c, tiles, n; int off[7]; uint8_t *dst; uint32_t m0; } af_t;
static inline HVX_Vector af_get(const uint16_t *x, uint32_t c, uint32_t tiles, uint32_t b, int r)
{ if (r < 0 || (uint32_t)(r >> 5) >= tiles) return Q6_V_vzero(); return ((const HVX_Vector *)blk((uint16_t *)x, c, (uint32_t)r >> 5, b))[(r & 31) >> 1]; }
static void af_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    af_t *f = (af_t *)a_; const uint32_t cb = f->c / 32;
    for (uint32_t j = t0; j < t1; j++)
    {
        const uint32_t t = f->m0 + j; HVX_Vector *d = (HVX_Vector *)(f->dst + (size_t)j * f->n * cb * BLK);
        if (t + 1 < f->tiles) vk_pf(blk((uint16_t *)f->x, f->c, t + 1, 0), (size_t)cb * BLK);
        for (uint32_t n = 0; n < f->n; n++)
        {
            const int off = f->off[n];
            for (uint32_t b = 0; b < cb; b++)
                for (int k = 0; k < 16; k++, d++)
                {
                    const int r = (int)(t * 32) + 2 * k + off;
                    if (!(off & 1)) *d = af_get(f->x, f->c, f->tiles, b, r);
                    else *d = Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(af_get(f->x, f->c, f->tiles, b, r - 1), 16), Q6_Vw_vasl_VwR(af_get(f->x, f->c, f->tiles, b, r + 1), 16));
                }
        }
    }
}
static void af_fill(void *arg, uint8_t *v_act, uint32_t m0, uint32_t tiles)
{ af_t f = *(af_t *)arg; f.dst = v_act; f.m0 = m0; tiles_parallel_range(af_tiles, &f, 0, tiles); }
/* y = x + sin^2(a x) / b = x + (1 - cos(2 pi f)) / (2 b), f the fractional part of a x / pi; the file holds a / pi and 1 / b */
typedef struct { const uint16_t *x; uint16_t *y; uint32_t c; const HVX_Vector *a, *b; } as_t;
static void as_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    as_t *s = (as_t *)a_; const uint32_t cb = s->c / 32; const HVX_Vector half = splat_hf(0.5f), two = splat_hf(2.f);
    const HVX_Vector c1 = splat_hf(-19.7392088f), c2 = splat_hf(64.9393940f), c3 = splat_hf(-85.4568172f), c4 = splat_hf(60.2446432f), c5 = splat_hf(-26.4262625f), c6 = splat_hf(7.90353260f);
    for (uint32_t t = t0; t < t1; t++) for (uint32_t b = 0; b < cb; b++)
    {
        const HVX_Vector *xi = (const HVX_Vector *)blk((uint16_t *)s->x, s->c, t, b); HVX_Vector *yo = (HVX_Vector *)blk(s->y, s->c, t, b);
        if (!(b & 3) && (t + 1 < t1 || b + 4 < cb)) vk_pf(blk((uint16_t *)s->x, s->c, t, b + 4), 8192);
        for (int k = 0; k < 16; k++)
        {
            HVX_Vector ph = hf(Q6_Vqf16_vmpy_VhfVhf(xi[k], s->a[b]));
            HVX_Vector f = hf(Q6_Vqf16_vsub_VhfVhf(ph, Q6_Vhf_equals_Vh(Q6_Vh_equals_Vhf(ph))));
            f = hf(Q6_Vqf16_vsub_VhfVhf(f, Q6_Vhf_equals_Vh(Q6_Vh_equals_Vhf(hf(Q6_Vqf16_vmpy_VhfVhf(f, two))))));
            const HVX_Vector u = hf(Q6_Vqf16_vmpy_VhfVhf(f, f));
            HVX_Vector p = c6;
            p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(p, u)), c5)); p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(p, u)), c4));
            p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(p, u)), c3)); p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(p, u)), c2));
            p = hf(Q6_Vqf16_vadd_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(p, u)), c1));
            /* cos - 1 = u p: the small end keeps its digits */
            const HVX_Vector d = hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(hf(Q6_Vqf16_vmpy_VhfVhf(u, p)), half)), s->b[b]));
            yo[k] = hf(Q6_Vqf16_vsub_VhfVhf(xi[k], d));
        }
    }
}
/* y row r = the row r + p: row t of the part j, for r + p = t s + j */
typedef struct { const uint16_t *src; uint16_t *y; uint32_t oc, M1, s, p; } ai_t;
static void ai_tiles(void *a_, uint32_t t0, uint32_t t1)
{
    ai_t *a = (ai_t *)a_; const uint32_t cb = a->oc / 32;
    for (uint32_t tl = t0; tl < t1; tl++) for (uint32_t b = 0; b < cb; b++)
    {
        uint16_t *d = blk(a->y, a->oc, tl, b);
        for (uint32_t i = 0; i < 32; i++)
        {
            const uint32_t rp = tl * 32 + i + a->p, j = rp % a->s, t = rp / a->s; const uint16_t *pt = a->src + (size_t)j * a->M1 * a->oc;
            const uint16_t *sr = blk((uint16_t *)pt, a->oc, t >> 5, b) + (size_t)(((t & 31) >> 1) * 64 + (t & 1)); uint16_t *dr = d + (size_t)((i >> 1) * 64 + (i & 1));
            for (int c = 0; c < 32; c++) dr[2 * c] = sr[2 * c];
        }
    }
}
typedef struct { const uint16_t *w; const float *p; uint64_t th, tp, ts, ti, ta; } ad_t;
static inline uint32_t ad_m(uint32_t T) { return (T + 31) & ~31u; }
static void ad_bias(HVX_Vector *bv, const float *bias, uint32_t cp)
{ for (uint32_t b = 0; b < cp / 32; b++) bv[b] = colvec(bias + b * 32); }
/* y = the product of x with n neighbours at offs, the bias on its output blocks */
static void ad_prod(ad_t *d, const uint16_t *x, uint16_t *y, uint32_t rows, uint32_t tilesx, uint32_t ci, uint32_t co, uint32_t n, const int *offs, const float *bias, const uint16_t *w)
{
    static HVX_Vector bv[64]; ad_bias(bv, bias, co);
    const uint32_t K = n * ci; uint32_t kcb = K / 32 < 16 ? K / 32 : 16; while ((K / 32) % kcb) kcb--;
    g_D = ci; g_kc = kcb * 32; af_t f = { x, ci, tilesx, n }; for (uint32_t i = 0; i < n; i++) f.off[i] = offs[i];
    wset_t ws = { w, co, y, OUT_ACT, 0, 0, bv }; const uint64_t c0 = HAP_perf_get_time_us();
    product_multi(0, af_fill, &f, rows, K, &ws, 1, 128, 0, &d->th); g_kc = 0; d->tp += HAP_perf_get_time_us() - c0;
}
/* T0: the latent's frames; x: its 64 channels; scr: three equal signals. y: stage stop's output (stop < 0: the end) */
int kblock_adec(remote_handle64 h_, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen,
                int T0, int stop, uint32_t *times, int timesLen)
{ g_need8 = 1;   /* VTCM laid out for 8 MB */
    static const uint16_t ST[6] = { 6, 5, 4, 4, 2, 2 }, CH[7] = { 2048, 1024, 512, 256, 128, 64, 64 }; static const uint8_t DIL[3] = { 1, 3, 9 };
    const size_t slot = (size_t)scrLen / 3;
    if (timesLen < 8 || (size_t)xLen < (size_t)ad_m(T0) * 64) return AEE_EBADPARM;
    eng_t eg; int err = eng_begin(&eg); if (err) return err;
    g_wfmt = 0; g_tqk = 0; ad_t d = { w, prm }; uint16_t *buf[3] = { scr, scr + slot, scr + 2 * slot }; int cur = 0, skip = -1, k = 0; uint32_t T = (uint32_t)T0, c = 64;
    #define AD_NEXT() (cur != 0 && skip != 0 ? 0 : cur != 1 && skip != 1 ? 1 : 2)
    #define AD_STAGE() do { if (k++ == stop) goto out; } while (0)
    hvx_copy((uint8_t *)buf[0], (const uint8_t *)x, (size_t)ad_m(T) * 64 * 2);
    #define AD_CONV(co_, pdil_, nb_) do { const uint32_t co = (co_), n = (nb_), M = ad_m(T); int off[7]; for (uint32_t i = 0; i < n; i++) off[i] = n == 1 ? 0 : ((int)i - 3) * (pdil_); \
        const int nx = AD_NEXT(); ad_prod(&d, buf[cur], buf[nx], M, M / 32, c, co, n, off, d.p, d.w); d.w = WP(d.w, (size_t)n * c * co); d.p += co; vk_pad(buf[nx], 1, M, T, co); cur = nx; c = co; AD_STAGE(); } while (0)
    #define AD_SNAKE() do { const int nx = AD_NEXT(); static HVX_Vector av[64], iv[64]; ad_bias(av, d.p, c); ad_bias(iv, d.p + c, c); d.p += 2 * c; as_t s = { buf[cur], buf[nx], c, av, iv }; const uint64_t c0 = HAP_perf_get_time_us(); \
        tiles_parallel_range(as_tiles, &s, 0, ad_m(T) / 32); d.ts += HAP_perf_get_time_us() - c0; cur = nx; AD_STAGE(); } while (0)
    AD_CONV(2048, 1, 7);
    for (int i = 0; i < 6; i++)
    {
        AD_SNAKE();
        {   /* the transposed convolution */
            const uint32_t s = ST[i], oc = CH[i + 1], pd = (s + 1) / 2, M1 = ad_m(T + 1), Tn = (T - 1) * s + 2 * s - 2 * pd, Mn = ad_m(Tn); const int off[2] = { 0, -1 }; const int nx = AD_NEXT();
            const float *bias = d.p; d.p += oc;
            for (uint32_t j = 0; j < s; j++) { ad_prod(&d, buf[cur], buf[nx] + (size_t)j * M1 * oc, M1, ad_m(T) / 32, c, oc, 2, off, bias, d.w); d.w = WP(d.w, (size_t)2 * c * oc); }
            ai_t a = { buf[nx], buf[cur], oc, M1, s, pd }; const uint64_t c0 = HAP_perf_get_time_us(); tiles_parallel_range(ai_tiles, &a, 0, Mn / 32); d.ti += HAP_perf_get_time_us() - c0;
            vk_pad(buf[cur], 1, Mn, Tn, oc); T = Tn; c = oc; AD_STAGE();
        }
        for (int r = 0; r < 3; r++)
        {
            skip = cur; AD_SNAKE(); AD_CONV(c, DIL[r], 7); AD_SNAKE(); AD_CONV(c, 1, 1);
            { const uint64_t c0 = HAP_perf_get_time_us(); vk_add(buf[skip], buf[cur], ad_m(T) / 32, c); d.ta += HAP_perf_get_time_us() - c0; skip = -1; AD_STAGE(); }
        }
    }
    AD_SNAKE(); AD_CONV(64, 1, 7);
out:;
    { const size_t n = (size_t)ad_m(T) * c; if ((size_t)yLen >= n) hvx_copy((uint8_t *)y, (const uint8_t *)buf[cur], n * 2); else err = AEE_EBADPARM; }
    times[0] = T; times[1] = c; times[2] = (uint32_t)d.th; times[3] = (uint32_t)d.tp; times[4] = (uint32_t)d.ts; times[5] = (uint32_t)d.ti; times[6] = (uint32_t)d.ta; times[7] = (uint32_t)g_tqk; (void)wLen; (void)prmLen;
    eng_end(&eg); return err;
}
