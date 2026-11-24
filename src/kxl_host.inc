/* kxl_host.inc (included by kengine.c) -- SDXL on the DSP: the UNet's layer runs, the VAE decoder, FLUX.2's VAE encoder, JNI entry points */

static void xl_linear(const float *w, const float *b, int no, int ni, const float *in, float *o)
{ for (int r = 0; r < no; r++) { float a = b[r]; for (int q = 0; q < ni; q++) a += w[(size_t)r * ni + q] * in[q]; o[r] = a; } }
/* both pictures' resnet time Linear at once, the weights read once, rows on 8 threads -- each output summed in xl_linear's order,
 * so the same floats */
typedef struct { const float *w, *b, *i0, *i1; float *o0, *o1; int ni, r0, r1; } xl2_t;
static void *xl2_part(void *a_)
{
    const xl2_t *a = (const xl2_t *)a_;
    for (int r = a->r0; r < a->r1; r++) { float s0 = a->b[r], s1 = a->b[r]; const float *wr = a->w + (size_t)r * a->ni;
        for (int q = 0; q < a->ni; q++) { s0 += wr[q] * a->i0[q]; s1 += wr[q] * a->i1[q]; } a->o0[r] = s0; a->o1[r] = s1; }
    return 0;
}
static void xl_linear2(const float *w, const float *b, int no, int ni, const float *i0, const float *i1, float *o0, float *o1)
{
    enum { NT = 8 }; pthread_t th[NT]; xl2_t ar[NT];
    for (int t = 0; t < NT; t++) { ar[t] = (xl2_t){ w, b, i0, i1, o0, o1, ni, no * t / NT, no * (t + 1) / NT }; pthread_create(&th[t], 0, xl2_part, &ar[t]); }
    for (int t = 0; t < NT; t++) pthread_join(th[t], 0);
}
/* xl_gen's buffers registered with FastRPC once (remote_register_buf_attr, KIPC_ATTR_KEEP_MAP | KIPC_ATTR_TRY_MAP_STATIC), so a call does not
 * map and unmap them each time (the call cost grew with the bytes passed). The driver's cache
 * maintenance stays. Looked up at run time (an older libcdsprpc without it: nothing registered) */
static int (*g_xreg)(void *, int, int, int), (*g_xtofd)(void *); static int g_xreg_init;
static void xreg(void *p, size_t n, int on)
{
    if (!g_xreg_init) { g_xreg_init = 1; { g_xreg = (int (*)(void *, int, int, int))dlsym(RTLD_DEFAULT, "remote_register_buf_attr");
                                                                    g_xtofd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd"); } }
    if (!g_xreg || !g_xtofd || !p) return;
    const int fd = g_xtofd(p); if (fd <= 0) return;
    const int r = g_xreg(p, (int)n, on ? fd : -1, on ? KIPC_ATTR_KEEP_MAP | KIPC_ATTR_TRY_MAP_STATIC : 0);   /* fd -1: unregister */
    if (r && on) LOG("remote_register_buf_attr %zu bytes: 0x%x", n, r);
}
/* The layers go to the DSP in runs (kblock_xrun), each run's weights packed one after the other in one of two slots of XRUNW
 * bytes, read ahead by xrun_reader. The skips, the concats and the swaps happen on the DSP */
#define XRUNW ((size_t)384 << 20)
#define XRUNP ((size_t)4 << 20)   /* a run's parameters, floats */
#define XRP 4   /* threads reading one run */
typedef struct { FILE *fw, *fp, *fl; int (*tab)[3]; const long *tw, *tp, *lw; int steps, nr; const int *k0, *k1; const size_t *woff, *poff, *fow, *fop, *lo; uint16_t *w[2]; float *pr[2];
                 int full[2], err, stop, np; pthread_mutex_t mu; pthread_cond_t cv; } xrr_t;
typedef struct { xrr_t *r; int s, ka, kb, e; } xrp_t;
static int xpread(int fd, void *dst, size_t n, size_t off)
{ size_t got = 0; while (got < n) { const ssize_t g = pread(fd, (uint8_t *)dst + got, n - got, (off_t)(off + got)); if (g <= 0) return 1; got += (size_t)g; } return 0; }
static void *xrun_part(void *a)
{
    xrp_t *p = (xrp_t *)a; xrr_t *r = p->r; const int fdw = fileno(r->fw), fdp = fileno(r->fp);
    for (int k = p->ka; k < p->kb && !p->e; k++)
    {
        if (r->tab[k][0] == 7 || r->tab[k][0] == 8) continue;
        p->e = xpread(fdw, (uint8_t *)r->w[p->s] + r->woff[k], (size_t)r->tw[k], r->fow[k]) || xpread(fdp, r->pr[p->s] + r->poff[k], (size_t)r->tp[k], r->fop[k]);
        if (!p->e && r->fl && r->lw[k]) p->e = xpread(fileno(r->fl), (uint8_t *)r->w[p->s] + r->woff[k] + r->tw[k], (size_t)r->lw[k], r->lo[k]);   /* the layer's runtime LoRA right after its weights */
    }
    return 0;
}
static void *xrun_reader(void *a)
{
    xrr_t *r = (xrr_t *)a; long q = 0;
    for (int st = 0; st < r->steps; st++)
    {
        for (int u = 0; u < r->nr; u++, q++)
        {
            const int s = (int)(q % 2);
            pthread_mutex_lock(&r->mu); while (r->full[s] && !r->stop) pthread_cond_wait(&r->cv, &r->mu); const int stop = r->stop; pthread_mutex_unlock(&r->mu);
            if (stop) return 0;
            int e = 0;
            {   /* the run's layers in np parts of about the same bytes, one on this thread */
                xrp_t pt[8]; pthread_t th[8]; int n = 0, k = r->k0[u]; size_t all = 0, acc = 0; for (int i = r->k0[u]; i < r->k1[u]; i++) all += (size_t)r->tw[i];
                for (int i = 0; i < r->np && k < r->k1[u]; i++)
                {
                    const int ka = k; const size_t upto = all * (size_t)(i + 1) / (size_t)r->np;
                    while (k < r->k1[u] && (i == r->np - 1 || acc < upto || k == ka)) { acc += (size_t)r->tw[k]; k++; }
                    pt[n] = (xrp_t){ r, s, ka, k, 0 }; n++;
                }
                for (int i = 1; i < n; i++) pthread_create(&th[i], 0, xrun_part, &pt[i]);
                if (n) xrun_part(&pt[0]);
                for (int i = 1; i < n; i++) pthread_join(th[i], 0);
                for (int i = 0; i < n; i++) e |= pt[i].e;
            }
            pthread_mutex_lock(&r->mu); r->full[s] = 1; if (e) r->err = 1; pthread_cond_broadcast(&r->cv); pthread_mutex_unlock(&r->mu);
            if (e) return 0;
        }
    }
    return 0;
}
/* xl_gen: SDXL's Euler loop on an open session (the app's, nXlGen): ctxin the cond and uncond texts [2][96 x 2048] (77 used), yin their
 * [2][2816] (pooled + size embedding), z the noise [4 x H x W] in, the latent out. cb (may be 0): after each step, a nonzero return stops (-99).
 * dir: kconv's files (unet.tab / .prm / .w, emb.f32, the time and label Linears). Everything it allocates is freed on the way out. */
static char g_xlora[600];   /* the app's runtime LoRA folder for the next xl_gen (nXlLora), empty: none */
/* the step and the unit of it the UNet is at (step << 20 | units done << 10 | units), for Engine.nXlProgress */
static volatile int g_xlp;
static int xl_gen(remote_handle64 h, const char *dir, const float *ctxin, int C, const float *yin, float *z, int H, int W, int VW, int steps, float cfg, int (*cb)(int, int, void *), void *cba)
{
    char n[1100]; int err = 0;
    static int tab[200][3]; static long tw[200], tp[200]; int nt = 0; snprintf(n, sizeof(n), "%s/unet.tab", dir); FILE *ft = fopen(n, "r"), *fw, *fp; if (!ft) return -2;
    while (nt < 200 && fscanf(ft, "%d %d %d %ld %ld", &tab[nt][0], &tab[nt][1], &tab[nt][2], &tw[nt], &tp[nt]) == 5) nt++; fclose(ft);
    snprintf(n, sizeof(n), "%s/unet.w", dir); fw = fopen(n, "rb"); snprintf(n, sizeof(n), "%s/unet.prm", dir); fp = fopen(n, "rb"); if (!fw || !fp) { if (fw) fclose(fw); if (fp) fclose(fp); return -2; }
    /* a runtime LoRA (g_xlora, the app's nXlLora): <dir>/lora.tab "layer offset bytes" a line, <dir>/lora.w its blobs,
       read after each layer's weights by the run reader and applied by kblock_xop's op | 0x400 */
    static long lw[200]; static size_t lo[200]; memset(lw, 0, sizeof(lw)); FILE *fl = 0;
    { const char *ld = g_xlora[0] ? g_xlora : 0;
      if (ld && *ld) { snprintf(n, sizeof(n), "%s/lora.tab", ld); FILE *lt = fopen(n, "r"); int k; long o, b;
                       if (lt) { while (fscanf(lt, "%d %ld %ld", &k, &o, &b) == 3) if (k >= 0 && k < 200) { lw[k] = b; lo[k] = (size_t)o; } fclose(lt);
                                 snprintf(n, sizeof(n), "%s/lora.w", ld); fl = fopen(n, "rb"); if (!fl) memset(lw, 0, sizeof(lw)); else LOG("sdxl: LoRA from %s", ld); } } }
    /* an activation's room: the picture's latent pixels by the widest layer (2560), 128^2 x 2560 at 1024^2 */
    const size_t BIGA = (size_t)H * W * 2560 * 2 + 1024, emn = 70506240 / 4, ZN = (size_t)4 * H * W;
    /* the cond and uncond pictures go through the UNet together, layer by layer: each layer's weights read once for both, and the next layer's read on a thread while the DSP runs this one */
    /* X: the two pictures' activations one after the other, Y: where the next layer puts them (swapped after each layer); one DSP call a
     * layer for both (kblock_xop op | 0x100) */
    uint16_t *X = ralloc(2 * BIGA), *Y = ralloc(2 * BIGA), *ST0 = ralloc(2 * BIGA), *CTX = ralloc((size_t)2 * C * 2048 * 2), *ctx[2]; ctx[0] = CTX; ctx[1] = CTX + (size_t)C * 2048;   /* C: the text's rows a picture (96, or 32 up from 77 k for k chunks) */
    float *em = malloc(emn * 4), *ep[2]; ep[0] = malloc(ZN * 4); ep[1] = malloc(ZN * 4); float *zc = malloc((size_t)H * W * 32 * 4), *fo = malloc((size_t)H * W * 64 * 4);
    /* the time and label Linears on the CPU */
    float *tw0 = malloc(1280 * 320 * 4), *tb0 = malloc(1280 * 4), *tw2 = malloc(1280 * 1280 * 4), *tb2 = malloc(1280 * 4), *lw0 = malloc(1280 * 2816 * 4), *lb0 = malloc(1280 * 4), *lw2 = malloc(1280 * 1280 * 4), *lb2 = malloc(1280 * 4), lab[2][1280];
    uint64_t t_op[16] = { 0 };   /* the DSP time by op (unet.tab); set here, before any XFAIL can jump to xg_out, whose log reads it */
    int started = 0; pthread_t rth;
    /* the runs: cut where the next layer's weights would pass XRUNW or its parameters XRUNP */
    static int rk0[200], rk1[200]; static size_t rwo[200], rpo[200], rwn[200], rpn[200], rfw[200], rfp[200]; int nr = 0;
    static xrr_t rr; uint16_t *ZIN = 0, *OUTB = 0, *SK = 0; static uint32_t utab[4 + 7 * 200];
    {
        size_t wa = 0, pa = 0; rk0[0] = 0;
        for (int k = 0; k < nt; k++)
        {
            const size_t wn = ((size_t)tw[k] + (size_t)lw[k] + 4095) & ~(size_t)4095, pn = (((size_t)tp[k] / 4 + (tab[k][0] == 1 ? 2 * (size_t)tab[k][2] : 0)) + 31) & ~(size_t)31;
            if (wa + wn > XRUNW || pa + pn > XRUNP) { rk1[nr] = k; rwn[nr] = wa; rpn[nr] = pa; nr++; rk0[nr] = k; wa = pa = 0; }
            rwo[k] = wa; rpo[k] = pa; wa += wn; pa += pn;
            rfw[k] = k ? rfw[k - 1] + (size_t)tw[k - 1] : 0; rfp[k] = k ? rfp[k - 1] + (size_t)tp[k - 1] : 0;   /* where the layer sits in unet.w / unet.prm */
        }
        rk1[nr] = nt; rwn[nr] = wa; rpn[nr] = pa; nr++;
        memset(&rr, 0, sizeof(rr)); rr.fw = fw; rr.fp = fp; rr.tab = tab; rr.tw = tw; rr.tp = tp; rr.steps = steps; rr.nr = nr; rr.k0 = rk0; rr.k1 = rk1; rr.woff = rwo; rr.poff = rpo; rr.fow = rfw; rr.fop = rfp; rr.fl = fl; rr.lw = lw; rr.lo = lo; rr.np = XRP;
        pthread_mutex_init(&rr.mu, 0); pthread_cond_init(&rr.cv, 0);
        for (int i = 0; i < 2; i++) { rr.w[i] = ralloc(XRUNW); rr.pr[i] = (float *)ralloc(XRUNP * 4); }
        ZIN = ralloc((size_t)2 * H * W * 32 * 2); OUTB = ralloc((size_t)2 * H * W * 64 * 2); SK = ralloc(2 * BIGA);
        LOG("sdxl runs: %d calls a step for %d layers", nr, nt);
    }
    #define XFAIL(e_) do { err = (e_); goto xg_out; } while (0)
    { if (!rr.w[0] || !rr.w[1] || !rr.pr[0] || !rr.pr[1] || !ZIN || !OUTB || !SK) XFAIL(-3);
                for (int i = 0; i < 2; i++) { xreg(rr.w[i], XRUNW, 1); xreg(rr.pr[i], XRUNP * 4, 1); } xreg(ZIN, (size_t)2 * H * W * 32 * 2, 1); xreg(OUTB, (size_t)2 * H * W * 64 * 2, 1); xreg(SK, 2 * BIGA, 1); }
    xreg(X, 2 * BIGA, 1); xreg(Y, 2 * BIGA, 1); xreg(ST0, 2 * BIGA, 1); xreg(CTX, (size_t)2 * C * 2048 * 2, 1);
    if (!X || !Y || !ST0 || !CTX || !em || !ep[0] || !ep[1] || !zc || !fo || !tw0 || !tb0 || !tw2 || !tb2 || !lw0 || !lb0 || !lw2 || !lb2) XFAIL(-3);
    if (readf(dir, "emb.f32", em, emn * 4)) XFAIL(-2);
    if (readf(dir, "time_embed.0.w.f32", tw0, 1280 * 320 * 4) || readf(dir, "time_embed.0.b.f32", tb0, 1280 * 4) || readf(dir, "time_embed.2.w.f32", tw2, 1280 * 1280 * 4) || readf(dir, "time_embed.2.b.f32", tb2, 1280 * 4) ||
        readf(dir, "label_emb.0.0.w.f32", lw0, 1280 * 2816 * 4) || readf(dir, "label_emb.0.0.b.f32", lb0, 1280 * 4) || readf(dir, "label_emb.0.2.w.f32", lw2, 1280 * 1280 * 4) || readf(dir, "label_emb.0.2.b.f32", lb2, 1280 * 4)) XFAIL(-2);
    for (int i = 0; i < 2; i++)
    {
        float a[1280]; const float *yv = yin + (size_t)i * 2816;
        to_act(ctxin + (size_t)i * C * 2048, C, 2048, 2048, 2048, (__fp16 *)ctx[i]); xl_linear(lw0, lb0, 1280, 2816, yv, a); for (int q = 0; q < 1280; q++) a[q] = a[q] / (1.f + expf(-a[q])); xl_linear(lw2, lb2, 1280, 1280, a, lab[i]);
    }
    /* the discrete sigmas of SDXL's scaled-linear betas, the steps evenly over the timesteps */
    static float sg[1000]; { double ac = 1; const double b0 = sqrt(0.00085), b1 = sqrt(0.012); for (int i = 0; i < 1000; i++) { const double b = b0 + (b1 - b0) * i / 999; ac *= 1 - b * b; sg[i] = (float)sqrt((1 - ac) / ac); } }
    /* v-prediction (<dir>/pred.txt "v", kconv from the checkpoint's v_pred key): the UNet gives v, eps = z s / (s^2 + 1) + v / sqrt(s^2 + 1);
       "zsnr" (its ztsnr key, NoobAI V-Pred): the sigmas rescaled to zero terminal SNR as ComfyUI does (the last alpha bar 4.897e-8, sigma ~4518) */
    int vpred = 0; { char pn[700]; snprintf(pn, sizeof(pn), "%s/pred.txt", dir); FILE *pf = fopen(pn, "r"); char pt[32] = "";
        if (pf) { if (!fgets(pt, sizeof(pt), pf)) pt[0] = 0; fclose(pf); }
        if (pt[0] == 'v') vpred = 1;
        if (strstr(pt, "zsnr"))
        {
            static double ab[1000]; for (int i = 0; i < 1000; i++) ab[i] = sqrt(1.0 / ((double)sg[i] * sg[i] + 1.0));
            const double a0 = ab[0], aT = ab[999]; for (int i = 0; i < 1000; i++) { ab[i] = (ab[i] - aT) * a0 / (a0 - aT); ab[i] *= ab[i]; }
            ab[999] = 4.8973451890853435e-08; for (int i = 0; i < 1000; i++) sg[i] = (float)sqrt((1.0 - ab[i]) / ab[i]);
        }
        if (vpred) LOG("sdxl: v-prediction%s", strstr(pt, "zsnr") ? ", zero terminal SNR" : ""); }
    float s0 = sg[999]; for (size_t i = 0; i < ZN; i++) z[i] *= s0;
    uint64_t t_dsp = 0, t_all0 = 0, t_rd = 0, t_em = 0, t_call = 0; struct timespec a0, a1, c0_, c1_;
    #define XNOW(v) { clock_gettime(CLOCK_MONOTONIC, &v); }
    #define XDT(p, q) ((uint64_t)((q.tv_sec - p.tv_sec) * 1000000 + (q.tv_nsec - p.tv_nsec) / 1000))
    pthread_create(&rth, 0, xrun_reader, (void *)&rr); started = 1; long q = 0;
    for (int st = 0; st < steps; st++)
    {
        struct timespec s0t, s1t; XNOW(s0t);
        const float tf = 999.f * (1.f - (float)st / steps); const int ti = (int)tf; const float fr = tf - ti, sig = ti >= 999 ? sg[999] : exp((1 - fr) * log(sg[ti]) + fr * log(sg[ti + 1])), tn = 999.f * (1.f - (float)(st + 1) / steps); const int tni = (int)tn;
        const float snx = st + 1 >= steps ? 0.f : exp((1 - (tn - tni)) * log(sg[tni]) + (tn - tni) * log(sg[tni + 1 > 999 ? 999 : tni + 1]));
        float te[320], a[1280], tv[1280]; for (int i = 0; i < 160; i++) { const double f = exp(-log(10000.0) * i / 160) * tf; te[i] = (float)cos(f); te[160 + i] = (float)sin(f); }
        xl_linear(tw0, tb0, 1280, 320, te, a); for (int q = 0; q < 1280; q++) a[q] = a[q] / (1.f + expf(-a[q])); xl_linear(tw2, tb2, 1280, 1280, a, tv);
        const float sc_in = 1.f / sqrtf(sig * sig + 1.f);
        g_xlp = (st << 20) | nr;
        static float sil[2][1280];
        for (int c = 0; c < 2; c++)
        {
            for (int q = 0; q < 1280; q++) { const float e = tv[q] + lab[c][q]; sil[c][q] = e / (1.f + expf(-e)); }
            memset(zc, 0, (size_t)H * W * 32 * 4); for (int ch = 0; ch < 4; ch++) for (size_t i = 0; i < (size_t)H * W; i++) zc[i * 32 + ch] = z[ch * (size_t)H * W + i] * sc_in;
            to_act(zc, H * W, 32, 32, 32, (__fp16 *)(ZIN + (size_t)c * H * W * 32));
        }
        const float *ee = em;
        for (int u = 0; u < nr; u++, q++)
        {
            if (g_abort) XFAIL(-99);   /* Engine.abort: a stop leaves between the runs of layers, not only after a step */
            const int slot = (int)(q % 2);
            XNOW(a0); pthread_mutex_lock(&rr.mu); while (!rr.full[slot] && !rr.err) pthread_cond_wait(&rr.cv, &rr.mu); const int rerr = rr.err && !rr.full[slot]; pthread_mutex_unlock(&rr.mu); XNOW(a1); t_rd += XDT(a0, a1);
            if (rerr) XFAIL(-2);
            float *pr = rr.pr[slot]; uint32_t tm[16] = { 0 }; const int n = rk1[u] - rk0[u];
            utab[0] = (u == 0 ? 1 : 0) | (u == nr - 1 ? 2 : 0) | (st == 0 && u == 0 ? 4 : 0) | 8 | ((uint32_t)(VW < W ? VW : 0) << 16); utab[1] = (uint32_t)n; utab[2] = (uint32_t)H; utab[3] = (uint32_t)W;   /* utab[0]'s flags: 1 first run, 2 last run, 4 a picture's first call, 8 the text's K, V kept */
            XNOW(a0);
            for (int k = rk0[u]; k < rk1[u]; k++)
            {
                const int op = tab[k][0], co = tab[k][2], plen0 = (int)(tp[k] / 4); uint32_t *e = utab + 4 + 7 * (k - rk0[u]);
                e[0] = (uint32_t)op | (lw[k] ? 0x400 : 0); e[1] = (uint32_t)tab[k][1]; e[2] = (uint32_t)co; e[3] = (uint32_t)(rwo[k] / 2); e[4] = (uint32_t)((tw[k] + lw[k]) / 2); e[5] = (uint32_t)rpo[k]; e[6] = (uint32_t)(plen0 + (op == 1 ? 2 * co : 0));
                if (op == 1) { xl_linear2(ee, ee + (size_t)co * 1280, co, 1280, sil[0], sil[1], pr + rpo[k] + plen0, pr + rpo[k] + plen0 + co); ee += (size_t)co * 1280 + co; }
            }
            XNOW(a1); t_em += XDT(a0, a1);
            XNOW(c0_); err = kblock_xrun(h, rr.w[slot], (int)(rwn[u] / 2), pr, (int)rpn[u], utab, 4 + 7 * n, CTX, 2 * C * 2048, ZIN, 2 * H * W * 32, X, (int)BIGA, Y, (int)BIGA, ST0, (int)BIGA, SK, (int)BIGA,
                                         OUTB, 2 * H * W * 64, tm, 16, 0); XNOW(c1_);
            if (err) { LOG("sdxl run %d failed 0x%x (after %u ops)", u, err, tm[1]); XFAIL(err); }
            t_call += XDT(c0_, c1_) - tm[0]; t_dsp += tm[0]; for (int o = 0; o < 10; o++) t_op[o] += tm[2 + o];
            pthread_mutex_lock(&rr.mu); rr.full[slot] = 0; pthread_cond_broadcast(&rr.cv); pthread_mutex_unlock(&rr.mu);
            g_xlp = (st << 20) | ((u + 1) << 10) | nr;
        }
        for (int c = 0; c < 2; c++) { from_act((const __fp16 *)(OUTB + (size_t)c * H * W * 64), H * W, 64, fo); for (int ch = 0; ch < 4; ch++) for (size_t i = 0; i < (size_t)H * W; i++) ep[c][ch * (size_t)H * W + i] = fo[i * 64 + ch]; }
        if (vpred) { const float cs = sig / (sig * sig + 1.f), cv = 1.f / sqrtf(sig * sig + 1.f);
                     for (size_t i = 0; i < ZN; i++) z[i] += (z[i] * cs + (ep[1][i] + cfg * (ep[0][i] - ep[1][i])) * cv) * (snx - sig); }
        else
        for (size_t i = 0; i < ZN; i++) z[i] += (ep[1][i] + cfg * (ep[0][i] - ep[1][i])) * (snx - sig);
        XNOW(s1t); t_all0 += XDT(s0t, s1t);
        {   /* the latent's largest |value| and NaN / inf: the first step showing them is where the numbers broke (the app's log, Logs.stats) */
            float mx = 0; size_t bad = 0; for (size_t i = 0; i < ZN; i++) { const float v = z[i]; if (v != v || v > 3e38f || v < -3e38f) bad++; else if (fabsf(v) > mx) mx = fabsf(v); }
            LOG("sdxl step %d/%d: %.1f s (dsp %.1f, reads %.1f, calls %.1f) max|x| %.3g%s", st + 1, steps, t_all0 / 1e6, t_dsp / 1e6, t_rd / 1e6, t_call / 1e6, mx, bad ? " NaN/inf <- broke here" : "");
        }
        if (cb && cb(st + 1, steps, cba)) XFAIL(-99);
    }
xg_out:
    LOG("sdxl dsp by op (s): conv %.1f resnet %.1f down %.1f up %.1f norm+in %.1f transformer %.1f proj_out %.1f out %.1f", t_op[0] / 1e6, t_op[1] / 1e6, t_op[2] / 1e6,
        t_op[3] / 1e6, t_op[4] / 1e6, t_op[5] / 1e6, t_op[6] / 1e6, t_op[9] / 1e6);
    if (started) { pthread_mutex_lock(&rr.mu); rr.stop = 1; pthread_cond_broadcast(&rr.cv); pthread_mutex_unlock(&rr.mu); pthread_join(rth, 0); }
    { for (int i = 0; i < 2; i++) { xreg(rr.w[i], XRUNW, 0); xreg(rr.pr[i], XRUNP * 4, 0); if (rr.w[i]) rpcmem_free(rr.w[i]); if (rr.pr[i]) rpcmem_free(rr.pr[i]); }
                xreg(ZIN, (size_t)2 * H * W * 32 * 2, 0); xreg(OUTB, (size_t)2 * H * W * 64 * 2, 0); xreg(SK, 2 * BIGA, 0); if (ZIN) rpcmem_free(ZIN); if (OUTB) rpcmem_free(OUTB); if (SK) rpcmem_free(SK); }
    fclose(fw); fclose(fp); if (fl) fclose(fl);
    xreg(X, 2 * BIGA, 0); xreg(Y, 2 * BIGA, 0); xreg(ST0, 2 * BIGA, 0); xreg(CTX, (size_t)2 * C * 2048 * 2, 0);
    if (X) rpcmem_free(X); if (Y) rpcmem_free(Y); if (ST0) rpcmem_free(ST0); if (CTX) rpcmem_free(CTX);
    free(em); free(ep[0]); free(ep[1]); free(zc); free(fo); free(tw0); free(tb0); free(tw2); free(tb2); free(lw0); free(lb0); free(lw2); free(lb2);
    #undef XFAIL
    return err;
}

/* xl_vae: the SDXL VAE decoder (xvae.w / xvae.prm in dir) on an open session: z the latent / 0.13025 [4 x H x W], rgb [3 x 8H x 8W] out in -1..1 */
static int xl_vae(remote_handle64 h, const char *dir, const float *z, int H, int VW, float *rgb)
{
    const int W = (VW + 31) & ~31;   /* a width that is not a multiple of 32: padded, the engine keeps the padding columns 0 (kblock_xvae's stop bits) */
    char n[1100]; long sw = 0, sp = 0; FILE *f;
    snprintf(n, sizeof(n), "%s/xvae.w", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sw = ftell(f); fclose(f); }
    snprintf(n, sizeof(n), "%s/xvae.prm", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sp = ftell(f); fclose(f); }
    if (!sw || !sp) return -2;
    const size_t Mi = (size_t)H * W, Mf = 64 * Mi, part = Mf * 256, scrn = 2 * part + part / 2; int err = 0;
    uint16_t *w = ralloc(sw), *x = ralloc(Mi * 32 * 2), *s = ralloc(scrn * 2), *y = ralloc(Mf * 128 * 2); float *pr = (float *)ralloc(sp), *zc = calloc(Mi * 32, 4), *fa = malloc(Mf * 64 * 4);
    if (!w || !x || !s || !y || !pr || !zc || !fa) err = -3;
    else if (readf(dir, "xvae.w", w, sw) || readf(dir, "xvae.prm", pr, sp)) err = -2;
    else
    {
        for (int c = 0; c < 4; c++) for (int r = 0; r < H; r++) for (int q = 0; q < VW; q++) zc[((size_t)r * W + q) * 32 + c] = z[((size_t)c * H + r) * VW + q];
        to_act(zc, (int)Mi, 32, 32, 32, (__fp16 *)x); uint32_t t[32] = { 0 };
        err = kblock_xvae(h, w, (int)(sw / 2), pr, (int)(sp / 4), x, (int)(Mi * 32), s, (int)scrn, y, (int)(Mf * 128), H, W, 21 | (VW < W ? VW << 8 : 0), t, 32);
        if (!err) { from_act((const __fp16 *)y, (int)Mf, 64, fa); const size_t Mo = (size_t)64 * H * VW;
                    for (int c = 0; c < 3; c++) for (int r = 0; r < 8 * H; r++) for (int q = 0; q < 8 * VW; q++) rgb[c * Mo + (size_t)r * 8 * VW + q] = fa[((size_t)r * 8 * W + q) * 64 + c]; }
    }
    if (w) rpcmem_free(w); if (x) rpcmem_free(x); if (s) rpcmem_free(s); if (y) rpcmem_free(y); if (pr) rpcmem_free(pr); free(zc); free(fa);
    return err;
}

/* xl_fvenc: FLUX.2's VAE encoder (fvenc.w / fvenc.prm in dir) on an open session: rgb [3 x H x W] in -1..1 (W a multiple of 256, H of 16)
   -> mean [32 x H/8 x W/8], the latent's mean before its 2 x 2 patching and batch norm (KleinGen does those) */
static int xl_fvenc(remote_handle64 h, const char *dir, const float *rgb, int H, int W, float *mean)
{
    if (W % 256 || H % 16) return -1;
    char n[1100]; long sw = 0, sp = 0; FILE *f;
    snprintf(n, sizeof(n), "%s/fvenc.w", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sw = ftell(f); fclose(f); }
    snprintf(n, sizeof(n), "%s/fvenc.prm", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sp = ftell(f); fclose(f); }
    if (!sw || !sp) return -2;
    const int lh = H / 8, lw = W / 8; const size_t Mf = (size_t)H * W, Mi = (size_t)lh * lw, part = Mf * 256, scrn = 2 * part + part / 2; int err = 0;
    uint16_t *w = ralloc(sw), *x = ralloc(Mf * 32 * 2), *s = ralloc(scrn * 2), *y = ralloc(Mi * 64 * 2); float *pr = (float *)ralloc(sp), *xc = calloc(Mf * 32, 4), *fa = malloc(Mi * 64 * 4);
    if (!w || !x || !s || !y || !pr || !xc || !fa) err = -3;
    else if (readf(dir, "fvenc.w", w, sw) || readf(dir, "fvenc.prm", pr, sp)) err = -2;
    else
    {
        for (int c = 0; c < 3; c++) for (size_t i = 0; i < Mf; i++) xc[i * 32 + c] = rgb[c * Mf + i];
        to_act(xc, (int)Mf, 32, 32, 32, (__fp16 *)x); uint32_t t[32] = { 0 };
        err = kblock_xvae(h, w, (int)(sw / 2), pr, (int)(sp / 4), x, (int)(Mf * 32), s, (int)scrn, y, (int)(Mi * 64), lh, lw, (1 << 30) | 17, t, 32);
        if (!err) { from_act((const __fp16 *)y, (int)Mi, 64, fa); for (int c = 0; c < 32; c++) for (size_t i = 0; i < Mi; i++) mean[c * Mi + i] = fa[i * 64 + c]; }
        uint32_t all = 0; for (int i = 0; i < 18; i++) all += t[i]; LOG("fvae encoder %dx%d: %u ms on the DSP, err 0x%x", W, H, all / 1000, err);
    }
    if (w) rpcmem_free(w); if (x) rpcmem_free(x); if (s) rpcmem_free(s); if (y) rpcmem_free(y); if (pr) rpcmem_free(pr); free(xc); free(fa);
    return err;
}

/* ------------------------------------------------------------------ the app's SDXL (SdxlGen.kt): a session per call */
static int xl_session(JNIEnv *env, jstring jarch, remote_handle64 *h)
{
    /* the skel next to this library, as nK2Open finds it (without it the DSP could not load it: 0x80000406) */
    char self[512] = ""; Dl_info di; if (dladdr((void *)xl_session, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0); char uri[256];
    snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN); (*env)->ReleaseStringUTFChars(env, jarch, arch);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um)); if (!err) err = kblock_open(uri, h); return err;
}
typedef struct { JNIEnv *env; jobject cb; jmethodID m; } xl_cb_t;
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nXlProgress(JNIEnv *env, jclass cl) { return g_xlp; }
static int xl_cb(int step, int steps, void *a) { xl_cb_t *c = (xl_cb_t *)a; return c->cb && (*c->env)->CallBooleanMethod(c->env, c->cb, c->m, step, steps) ? 1 : 0; }
/* the runtime LoRA folder (lora.tab, lora.w) for the next generations, "" for none */
JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nXlLora(JNIEnv *env, jclass cl, jstring jdir)
{ const char *d = (*env)->GetStringUTFChars(env, jdir, 0); snprintf(g_xlora, sizeof(g_xlora), "%s", d); (*env)->ReleaseStringUTFChars(env, jdir, d); (void)cl; }
/* the Euler loop: ctx [2 x C x 2048] (cond, uncond; C 96 a CLIP chunk, 32 up from 77 k for k chunks), y [2 x 2816], z the noise in and the latent out; cb.step(step, steps): true to stop */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nXlGen(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jfloatArray jctx, jfloatArray jy, jfloatArray jz, jint H, jint W, jint steps, jfloat cfg, jobject cb)
{
    remote_handle64 h = 0; int err = xl_session(env, jarch, &h); if (err) return err;
    const char *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    jfloat *ctx = (*env)->GetFloatArrayElements(env, jctx, 0), *y = (*env)->GetFloatArrayElements(env, jy, 0), *z = (*env)->GetFloatArrayElements(env, jz, 0);
    xl_cb_t c = { env, cb, cb ? (*env)->GetMethodID(env, (*env)->GetObjectClass(env, cb), "step", "(II)Z") : 0 };
    const int C = (int)((*env)->GetArrayLength(env, jctx) / (2 * 2048));   /* the text's rows a picture: 96 for one chunk, more for long prompts */
    /* a width that is not a multiple of 32 latent pixels: the latent padded to the next one, its columns past W zero (the engine masks
       and clears them and pads each level to 32 on its own, kxl.inc), then cut back */
    const int WP = (W + 31) & ~31; float *zp = z;
    if (WP != W) { zp = calloc((size_t)4 * H * WP, 4); if (zp) for (int ch = 0; ch < 4; ch++) for (int r = 0; r < H; r++) memcpy(zp + ((size_t)ch * H + r) * WP, z + ((size_t)ch * H + r) * W, (size_t)W * 4); }
    err = !zp ? -3 : C >= 96 && C % 32 == 0 ? xl_gen(h, dir, ctx, C, y, zp, H, WP, W, steps, cfg, cb ? xl_cb : 0, &c) : -1;
    if (zp != z) { if (zp && !err) for (int ch = 0; ch < 4; ch++) for (int r = 0; r < H; r++) memcpy(z + ((size_t)ch * H + r) * W, zp + ((size_t)ch * H + r) * WP, (size_t)W * 4); free(zp); }
    (*env)->ReleaseFloatArrayElements(env, jctx, ctx, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jy, y, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jz, z, 0);
    (*env)->ReleaseStringUTFChars(env, jdir, dir); kblock_close(h); return err;
}
/* the DSP's own account of itself for a bug report (Logs.report, Engine.hwInfo): kblock_probe 211 into out [16]: VTCM total and available
 * (bytes), the VTCM the engine asks for, HVX threads, HMX lock4, the HMX clock vote's answer, whether the engine's resources were got
 * (0 or the error), the fp16 HMX one-tile self-test's relative error x 1e6 (or -1 when not run), arch; the call's own error in the return */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nHwInfo(JNIEnv *env, jclass cl, jstring jarch, jintArray jout)
{
    remote_handle64 h = 0; int err = xl_session(env, jarch, &h); if (err) return err;
    uint16_t r[1024]; memset(r, 0, sizeof(r)); err = kblock_probe(h, 211, r, 1024); kblock_close(h);
    jint o[16]; memcpy(o, r, sizeof(o)); (*env)->SetIntArrayRegion(env, jout, 0, 16, o); return err;
}
/* FLUX.2's VAE encoder: rgb [3 x H x W] in -1..1 -> mean [32 x H/8 x W/8] (xl_fvenc) */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nFvEnc(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jfloatArray jrgb, jint H, jint W, jfloatArray jmean)
{
    remote_handle64 h = 0; int err = xl_session(env, jarch, &h); if (err) return err;
    const char *dir = (*env)->GetStringUTFChars(env, jdir, 0); jfloat *rgb = (*env)->GetFloatArrayElements(env, jrgb, 0), *m = (*env)->GetFloatArrayElements(env, jmean, 0);
    err = xl_fvenc(h, dir, rgb, H, W, m);
    (*env)->ReleaseFloatArrayElements(env, jrgb, rgb, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jmean, m, 0); (*env)->ReleaseStringUTFChars(env, jdir, dir); kblock_close(h); (void)cl; return err;
}
/* the SDXL VAE decoder: z the latent / 0.13025 [4 x H x W] in, rgb [3 x 8H x 8W] out in -1..1 */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nXlVae(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jfloatArray jz, jint H, jint W, jfloatArray jrgb)
{
    remote_handle64 h = 0; int err = xl_session(env, jarch, &h); if (err) return err;
    const char *dir = (*env)->GetStringUTFChars(env, jdir, 0); jfloat *z = (*env)->GetFloatArrayElements(env, jz, 0), *rgb = (*env)->GetFloatArrayElements(env, jrgb, 0);
    err = xl_vae(h, dir, z, H, W, rgb);
    (*env)->ReleaseFloatArrayElements(env, jz, z, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jrgb, rgb, 0); (*env)->ReleaseStringUTFChars(env, jdir, dir); kblock_close(h); return err;
}
