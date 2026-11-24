/* sdxl_conv.inc (kconv.c) -- an SDXL checkpoint's UNet for kxl.inc's kblock_xop:
 * unet.w (each matrix in the layout, f16), unet.prm (fp32: norms, biases), unet.tab (one op a line: op ci co wbytes pbytes),
 * emb.f32 (the resnets' emb Linears), time_embed.* / label_emb.* .w.f32 / .b.f32. f16 only. */
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif
static FILE *x_w, *x_p, *x_e; static int x_th; static char x_tab[64 * 1024]; static size_t x_tl;

/* the tensor as fp32 with its shape */
static float *x_rd(const char *name, tinfo_t *t) { int64_t r, c; if (find_tensor(name, t)) { fprintf(stderr, "missing %s%s\n", g_pre, name); exit(1); } return read_f32(name, &r, &c); }
static int x_has(const char *name) { tinfo_t t; return !find_tensor(name, &t); }
static void x_f32(FILE *f, const char *name) { tinfo_t t; float *v = x_rd(name, &t); size_t n = 1; for (int d = 0; d < t.nd; d++) n *= t.shape[d]; fwrite(v, 4, n, f); free(v); }
/* W [N x K] to unet.w, the rows zero-padded to 64 */
static void x_mat(float *w, int64_t N, int64_t K, float mul)
{
    const int64_t Np = (N + 63) / 64 * 64; float *p = calloc((size_t)Np * K, 4);
    for (size_t i = 0; i < (size_t)N * K; i++) p[i] = w[i] * mul;
    put(x_w, p, Np, K, x_th); free(p);
}
static void x_lin(const char *name, float mul) { tinfo_t t; float *w = x_rd(name, &t); x_mat(w, t.shape[0], t.shape[1], mul); free(w); }
/* a conv or Linear: [co, c0, kh, kw] as [co x kh kw ci] (input channels last, zero-padded to ci), its bias padded to 64; out co, c0 */
static void x_conv(const char *k, int ci, int *co_, int *c0_)
{
    char n[256]; tinfo_t t; snprintf(n, sizeof(n), "%s.weight", k); float *w = x_rd(n, &t);
    const int co = (int)t.shape[0], c0 = (int)t.shape[1], kk = t.nd == 4 ? (int)(t.shape[2] * t.shape[3]) : 1; if (!ci) ci = c0;
    float *m = calloc((size_t)co * kk * ci, 4);
    for (int o = 0; o < co; o++) for (int c = 0; c < c0; c++) for (int s = 0; s < kk; s++) m[((size_t)o * kk + s) * ci + c] = w[((size_t)o * c0 + c) * kk + s];
    x_mat(m, co, (int64_t)kk * ci, 1.f); free(m); free(w);
    snprintf(n, sizeof(n), "%s.bias", k); float *b = x_rd(n, &t); const int cp = (co + 63) / 64 * 64; float *bp = calloc(cp, 4); memcpy(bp, b, co * 4); fwrite(bp, 4, cp, x_p); free(bp); free(b);
    if (co_) *co_ = co; if (c0_) *c0_ = c0;
}
static void x_gn(const char *k) { char n[256]; snprintf(n, sizeof(n), "%s.weight", k); x_f32(x_p, n); snprintf(n, sizeof(n), "%s.bias", k); x_f32(x_p, n); }
#ifdef _WIN32
#define x_tell _ftelli64
#else
#define x_tell ftello
#endif
static long long x_w0, x_p0;   /* unet.w passes 2 GB: 64-bit positions */
static void x_beg(void) { x_w0 = x_tell(x_w); x_p0 = x_tell(x_p); }
static void x_add(int op, int ci, int co) { x_tl += snprintf(x_tab + x_tl, sizeof(x_tab) - x_tl, "%d %d %d %lld %lld\n", op, ci, co, (long long)(x_tell(x_w) - x_w0), (long long)(x_tell(x_p) - x_p0)); }
static void x_sub(const char *q)
{
    char n[256]; int co, c0, c2;
    snprintf(n, sizeof(n), "%s.in_layers.0.weight", q);
    if (x_has(n))
    {   /* a resnet */
        x_beg(); snprintf(n, sizeof(n), "%s.in_layers.0", q); x_gn(n); snprintf(n, sizeof(n), "%s.in_layers.2", q); x_conv(n, 0, &co, &c0);
        snprintf(n, sizeof(n), "%s.out_layers.0", q); x_gn(n); snprintf(n, sizeof(n), "%s.out_layers.3", q); x_conv(n, 0, &c2, 0);
        snprintf(n, sizeof(n), "%s.skip_connection.weight", q); if (x_has(n)) { snprintf(n, sizeof(n), "%s.skip_connection", q); x_conv(n, 0, 0, 0); }
        snprintf(n, sizeof(n), "%s.emb_layers.1.weight", q); x_f32(x_e, n); snprintf(n, sizeof(n), "%s.emb_layers.1.bias", q); x_f32(x_e, n);
        x_add(1, c0, c2); return;
    }
    snprintf(n, sizeof(n), "%s.proj_in.weight", q);
    if (x_has(n))
    {   /* a spatial transformer: norm, proj_in, its blocks, proj_out */
        x_beg(); snprintf(n, sizeof(n), "%s.norm", q); x_gn(n); snprintf(n, sizeof(n), "%s.proj_in", q); x_conv(n, 0, &co, 0); x_add(4, co, co);
        const float QS = 1.4426950408889634f / 8.f;   /* log2 e / sqrt(64) */
        for (int i = 0;; i++)
        {
            char b[200]; snprintf(b, sizeof(b), "%s.transformer_blocks.%d.", q, i); snprintf(n, sizeof(n), "%snorm1.weight", b); if (!x_has(n)) break;
            x_beg();
            for (int m = 1; m <= 3; m++) { snprintf(n, sizeof(n), "%snorm%d", b, m); x_gn(n); }
            for (int a = 1; a <= 2; a++)
            {
                static const char *mm[4] = { "to_q", "to_k", "to_v", "to_out.0" };
                for (int m = 0; m < 4; m++) { snprintf(n, sizeof(n), "%sattn%d.%s.weight", b, a, mm[m]); x_lin(n, m ? 1.f : QS); }
                snprintf(n, sizeof(n), "%sattn%d.to_out.0.bias", b, a); x_f32(x_p, n);
            }
            snprintf(n, sizeof(n), "%sff.net.0.proj.weight", b);
            {   /* the two halves (up, gate) as two matrices: kxl.inc xl_block points at the gate half with WP(wm1, c F), a matrix of its own */
                tinfo_t t; float *w = x_rd(n, &t); const int64_t F = t.shape[0] / 2, K = t.shape[1];
                x_mat(w, F, K, 1.f); x_mat(w + (size_t)F * K, F, K, 1.f); free(w); } snprintf(n, sizeof(n), "%sff.net.2.weight", b); x_lin(n, 1.f);
            snprintf(n, sizeof(n), "%sff.net.0.proj.bias", b); x_f32(x_p, n); snprintf(n, sizeof(n), "%sff.net.2.bias", b); x_f32(x_p, n);
            x_add(5, co, co);
        }
        x_beg(); snprintf(n, sizeof(n), "%s.proj_out", q); x_conv(n, 0, 0, 0); x_add(6, co, co); return;
    }
    snprintf(n, sizeof(n), "%s.op.weight", q);
    if (x_has(n)) { x_beg(); snprintf(n, sizeof(n), "%s.op", q); x_conv(n, 0, &co, &c0); x_add(2, c0, co); return; }
    snprintf(n, sizeof(n), "%s.conv.weight", q);
    if (x_has(n)) { x_beg(); snprintf(n, sizeof(n), "%s.conv", q); x_conv(n, 0, &co, &c0); x_add(3, c0, co); return; }
    fprintf(stderr, "what is %s\n", q); exit(1);
}
/* the parts k.0 .. k.3 there are, in order */
static void x_block(const char *k)
{
    static const char *probe[4] = { "in_layers.0.weight", "proj_in.weight", "op.weight", "conv.weight" };
    for (int j = 0; j < 4; j++)
    {
        char q[128], t[256]; int any = 0; snprintf(q, sizeof(q), "%s.%d", k, j);
        for (int p = 0; p < 4 && !any; p++) { snprintf(t, sizeof(t), "%s.%s", q, probe[p]); any = x_has(t); }
        if (any) x_sub(q);
    }
}
/* the text encoders: <out>/te as the bundle's <base> (its ONNX graphs; the .data files copied, then every weight clip_map.txt names
 * written over from the checkpoint: its table: the source's rows [r0, r0 + nr), transposed for T). Constants stay the base's. */
static int x_te(const char *out, const char *base)
{
    char p[1024], q[1024]; snprintf(p, sizeof(p), "%s/clip_map.txt", base); FILE *mf = fopen(p, "r"); if (!mf) { perror(p); return 1; }
    snprintf(q, sizeof(q), "%s/te", out);
#ifdef _WIN32
    mkdir(q);
#else
    mkdir(q, 0755);
#endif
    static const char *fs[5] = { "clip_l.onnx", "clip_g.onnx", "clip_l.data", "clip_g.data", "clip_map.txt" };
    for (int i = 0; i < 5; i++)
    {   /* copies; the .data ones as .part until written over */
        snprintf(p, sizeof(p), "%s/%s", base, fs[i]); snprintf(q, sizeof(q), "%s/te/%s%s", out, fs[i], i == 2 || i == 3 ? ".part" : "");
        FILE *a = fopen(p, "rb"), *b = fopen(q, "wb"); if (!a || !b) { perror(a ? q : p); return 1; }
        char *buf = malloc(1 << 22); size_t n; while ((n = fread(buf, 1, 1 << 22, a)) > 0) fwrite(buf, 1, n, b); free(buf); fclose(a); fclose(b);
    }
    const char *pre = g_pre; g_pre = ""; char loc[64], key[256], tr[4]; long long off, nv; int r0, nr, lines = 0;
    FILE *d[2]; for (int i = 0; i < 2; i++) { snprintf(q, sizeof(q), "%s/te/%s.part", out, fs[2 + i]); d[i] = fopen(q, "r+b"); if (!d[i]) { perror(q); return 1; } }
    while (fscanf(mf, "%63s %lld %lld %255s %d %d %3s", loc, &off, &nv, key, &r0, &nr, tr) == 7)
    {
        tinfo_t t; float *a = x_rd(key, &t); const int64_t cols = t.nd > 1 ? t.shape[1] : 1, rows = t.nd ? t.shape[0] : 1;
        if ((int64_t)nr * cols != nv || r0 + nr > rows) { fprintf(stderr, "clip_map: %s %d+%d x %lld is not %lld values\n", key, r0, nr, (long long)cols, nv); return 1; }
        float *o = malloc(nv * 4), *s = a + (size_t)r0 * cols;
        if (tr[0] == 'T') { for (int64_t r = 0; r < nr; r++) for (int64_t c = 0; c < cols; c++) o[c * nr + r] = s[r * cols + c]; }
        else memcpy(o, s, nv * 4);
        FILE *f = d[strstr(loc, "clip_g") ? 1 : 0];
#ifdef _WIN32
        _fseeki64(f, off, SEEK_SET);
#else
        fseeko(f, off, SEEK_SET);
#endif
        fwrite(o, 4, nv, f); free(o); free(a); lines++;
    }
    fclose(mf); fclose(d[0]); fclose(d[1]); g_pre = pre;
    for (int i = 2; i < 4; i++) { snprintf(p, sizeof(p), "%s/te/%s.part", out, fs[i]); snprintf(q, sizeof(q), "%s/te/%s", out, fs[i]); rename(p, q); }
    printf("text encoders: %d weights from the checkpoint\n", lines); return 0;
}
static int sdxl(const char *out, int threads)
{
    if (g_q) { fprintf(stderr, "SDXL: f16 only\n"); return 1; }
    x_th = threads; char p[1024]; struct timeval a, b; gettimeofday(&a, 0);
    snprintf(p, sizeof(p), "%s/unet.w.part", out); x_w = fopen(p, "wb"); snprintf(p, sizeof(p), "%s/unet.prm", out); x_p = fopen(p, "wb"); snprintf(p, sizeof(p), "%s/emb.f32", out); x_e = fopen(p, "wb");
    if (!x_w || !x_p || !x_e) { perror(out); return 1; }
    int ni = 0, no = 0; for (;; ni++) { char n[96]; snprintf(n, sizeof(n), "input_blocks.%d.", ni); char k[128]; int any = 0;
        static const char *s0[6] = { "0.weight", "0.in_layers.0.weight", "0.op.weight", "1.proj_in.weight", "0.conv.weight", "1.norm.weight" };
        for (int s = 0; s < 6 && !any; s++) { snprintf(k, sizeof(k), "%s%s", n, s0[s]); any = x_has(k); } if (!any) break; }
    for (;; no++) { char k[128]; snprintf(k, sizeof(k), "output_blocks.%d.0.in_layers.0.weight", no); if (!x_has(k)) break; }
    /* "SDXL: <n> blocks" then "block <i> done" each (the app's progress, Custom.convert): the input ones, the middle, the output ones */
    int nbk = 0; printf("SDXL: %d blocks, %d input, %d output, out %s\n", ni + no, ni, no, "f16"); fflush(stdout);
#define X_DONE() do { gettimeofday(&b, 0); printf("block %d done, %.1f s\n", nbk++, (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1e6); fflush(stdout); } while (0)
    x_beg(); x_conv("input_blocks.0.0", 32, 0, 0); x_add(0, 32, 320); x_tl += snprintf(x_tab + x_tl, sizeof(x_tab) - x_tl, "7 0 0 0 0\n"); X_DONE();
    for (int i = 1; i < ni; i++) { char k[64]; snprintf(k, sizeof(k), "input_blocks.%d", i); x_block(k); x_tl += snprintf(x_tab + x_tl, sizeof(x_tab) - x_tl, "7 0 0 0 0\n"); X_DONE(); }
    x_block("middle_block");
    for (int i = 0; i < no; i++) { x_tl += snprintf(x_tab + x_tl, sizeof(x_tab) - x_tl, "8 0 0 0 0\n"); char k[64]; snprintf(k, sizeof(k), "output_blocks.%d", i); x_block(k); if (i < no - 1) X_DONE(); }
    { int co; x_beg(); x_gn("out.0"); x_conv("out.2", 0, &co, 0); x_add(9, 320, 64); }
    fclose(x_w); fclose(x_p); fclose(x_e);
    snprintf(p, sizeof(p), "%s/unet.tab", out); FILE *f = fopen(p, "w"); fwrite(x_tab, 1, x_tl, f); fclose(f);
    /* v-prediction (NoobAI V-Pred and the like carry empty "v_pred" / "ztsnr" keys): pred.txt for xl_gen's sampler */
    if (!g_gguf && strstr(g_hdr, "\"v_pred\"")) { snprintf(p, sizeof(p), "%s/pred.txt", out); f = fopen(p, "w"); if (f) { fputs(strstr(g_hdr, "\"ztsnr\"") ? "v zsnr\n" : "v\n", f); fclose(f); } }
    static const char *ex[4] = { "time_embed.0", "time_embed.2", "label_emb.0.0", "label_emb.0.2" };
    for (int i = 0; i < 4; i++) for (int k = 0; k < 2; k++)
    { char n[96], q[1024]; snprintf(n, sizeof(n), "%s.%s", ex[i], k ? "bias" : "weight"); snprintf(q, sizeof(q), "%s/%s.%s.f32", out, ex[i], k ? "b" : "w"); f = fopen(q, "wb"); x_f32(f, n); fclose(f); }
    char q[1024]; snprintf(p, sizeof(p), "%s/unet.w.part", out); snprintf(q, sizeof(q), "%s/unet.w", out); rename(p, q);   /* complete files only */
    /* the bundle's te dir (clip_map.txt): KX_TE (the app's), else the bundle dir argument */
    const char *te = getenv("KX_TE"); if (!te || !*te) te = g_bundle;
    if (*te && x_te(out, te)) return 1;
    X_DONE();   /* the last output block's, after the text encoders */
    gettimeofday(&b, 0); printf("SDXL done, %.1f s\n", (b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1e6);
    return 0;
}
