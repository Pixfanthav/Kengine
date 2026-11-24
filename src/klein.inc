/* klein.inc (included by kengine.c) -- the VAE decoder and encoder, Real-ESRGAN and the large linear products: JNI entry points */
/* JNI (Engine.vdec): z [32 x H x W] -> the picture, ARGB [8 H x 8 W]: kblock_vdec on dir's vdec.w, vdec.prm, its own DSP session for the one call */
#define KTS(a_, b_) (((b_).tv_sec - (a_).tv_sec) * 1e3 + ((b_).tv_nsec - (a_).tv_nsec) / 1e6)
/* nVdec's kept session and weights, at file scope so that nVdecDrop can let them go */
static remote_handle64 kh; static char karch[16], kdir[600]; static uint16_t *kw; static float *kpr; static long ksw, ksp;
/* Engine.vdecDrop, at the end of a run of pictures: the kept session closed and the weights freed. Left open, the DSP process lives as long
 * as the app, and what earlier runs left in its address space adds up: on some devices Anima, then SDXL at two sizes
 * in one app process failed in SDXL's VAE with 0xe (Bad address); the same picture after the app was restarted went through. */
JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nVdecDrop(JNIEnv *env, jclass cl)
{
    if (kw) { rpcmem_free(kw); kw = 0; } if (kpr) { rpcmem_free(kpr); kpr = 0; } ksw = ksp = 0; kdir[0] = 0;
    if (kh) { kblock_close(kh); kh = 0; } karch[0] = 0; (void)env; (void)cl;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nVdec(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jfloatArray jz, jint H, jint W, jint qw, jintArray jpx, jintArray jtimes)
{
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    char self[512] = ""; Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nVdec, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    char uri[256]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    /* the session and the weights kept from one picture to the next (same arch and folder), the scratch
       and output still per call (1 GB at 1024^2: not left pinned between pictures) */
    struct timespec q0, q1, q2, q3, q4; clock_gettime(CLOCK_MONOTONIC, &q0);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 }; remote_handle64 h = 0; int err = 0;
    if (kh && strcmp(karch, arch) == 0) h = kh;
    else
    {
        if (kh) { kblock_close(kh); kh = 0; }
        err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um)); if (!err) err = kblock_open(uri, &h);
        if (!err) { kh = h; snprintf(karch, sizeof(karch), "%s", arch); }
    }
    const int wd = (W + 31) & ~31, PH = 8 * H, PW = 8 * W, PWD = (PW + 31) & ~31; const size_t part = (size_t)PH * PWD * 192, nS = 2 * part + part / 2, nY = (size_t)PH * PWD * 64, nX = (size_t)H * wd * 32;
    char n[1100]; long sw = 0, sp = 0; FILE *f;
    snprintf(n, sizeof(n), "%s/vdec.w", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sw = ftell(f); fclose(f); }
    snprintf(n, sizeof(n), "%s/vdec.prm", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sp = ftell(f); fclose(f); }
    uint16_t *w = 0, *x = 0, *s = 0, *y = 0; float *pr = 0;
    if (!err && (!sw || !sp)) err = -2;
    const int have = kw && strcmp(kdir, dir) == 0 && ksw == sw && ksp == sp;
    if (!err && have) { w = kw; pr = kpr; }
    else if (!err)
    {
        if (kw) { rpcmem_free(kw); rpcmem_free(kpr); kw = 0; kpr = 0; }
        w = ralloc(sw); pr = (float *)ralloc(sp);
        if (w && pr && (readf(dir, "vdec.w", w, sw) || readf(dir, "vdec.prm", pr, sp))) err = -2;
        if (!err && w && pr) { kw = w; kpr = pr; ksw = sw; ksp = sp; snprintf(kdir, sizeof(kdir), "%s", dir); }
    }
    clock_gettime(CLOCK_MONOTONIC, &q1);
    if (!err) { x = ralloc(nX * 2); s = ralloc(nS * 2); y = ralloc(nY * 2); if (!w || !x || !s || !y || !pr) err = -3; }
    clock_gettime(CLOCK_MONOTONIC, &q2);
    if (!err)
    {
        float *zc = calloc(nX, 4); jfloat *z = (*env)->GetFloatArrayElements(env, jz, 0);
        for (int c = 0; c < (qw ? 16 : 32); c++) for (int r = 0; r < H; r++) for (int q = 0; q < W; q++) zc[((size_t)r * wd + q) * 32 + c] = z[((size_t)c * H + r) * W + q];
        (*env)->ReleaseFloatArrayElements(env, jz, z, JNI_ABORT);
        to_act(zc, H * wd, 32, 32, 32, (__fp16 *)x); free(zc);
        uint32_t t[32] = { 0 }; err = kblock_vdec(h, w, (int)(sw / 2), pr, (int)(sp / 4), x, (int)nX, s, (int)nS, y, (int)nY, H, wd, W, 0, (qw ? 1 << 24 : 0) | 21, t, 32);
        (*env)->SetIntArrayRegion(env, jtimes, 0, 32, (const jint *)t);
    }
    clock_gettime(CLOCK_MONOTONIC, &q3);
    if (!err)
    {
        jint *px = (*env)->GetIntArrayElements(env, jpx, 0); const __fp16 *o = (const __fp16 *)y;
        for (int r = 0; r < PH; r++) for (int q = 0; q < PW; q++)
        {
            const size_t i = (size_t)r * PWD + q; const size_t b = (i >> 5) * 2 * 1024 + 32 * (i & 30) + (i & 1); uint32_t v = 0xFF;
            for (int c = 0; c < 3; c++) { float a = (float)o[b + 2 * c]; a = a < -1.f ? -1.f : a > 1.f ? 1.f : a; v = (v << 8) | (uint32_t)((a + 1.f) * 127.5f + 0.5f); }
            px[(size_t)r * PW + q] = (jint)v;
        }
        (*env)->ReleaseIntArrayElements(env, jpx, px, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &q4);
    LOG("vdec host: open + weights %.0f ms (%s), buffers %.0f, decode call %.0f, pixels %.0f", KTS(q0, q1), have ? "kept" : "read", KTS(q1, q2), KTS(q2, q3), KTS(q3, q4));
    if (w && w != kw) rpcmem_free(w); if (pr && pr != kpr) rpcmem_free(pr); if (x) rpcmem_free(x); if (s) rpcmem_free(s); if (y) rpcmem_free(y);
    if (h && h != kh) kblock_close(h);
    if (err && kh) { kblock_close(kh); kh = 0; if (kw) { rpcmem_free(kw); rpcmem_free(kpr); kw = 0; kpr = 0; } }   /* a failed call: start clean next time */
    (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir);
    if (err) LOG("vdec failed 0x%x", err);
    return err;
}

/* JNI (Engine.K2.linear): out [M x N] = x [M x K] W^T on the HMX, W as nMossLinear's; for the large products outside the blocks (x_embedder, proj_out).
 * The weights' layout is made once per (aux, woff, N, K) and kept (8 at most, the oldest let go); x and y buffers grow and stay.
 * klw_flush (nK2Close) lets them all go, so a new model's aux at the same address cannot hit an old entry */
typedef struct { const void *map; jlong woff; int N, K; uint16_t *w; } klw_t;
static klw_t g_klw[8]; static int g_klwn; static uint16_t *g_klx, *g_kly; static size_t g_klxn, g_klyn;
static void klw_flush(void)
{
    for (int i = 0; i < g_klwn; i++) rpcmem_free(g_klw[i].w); g_klwn = 0;
    if (g_klx) rpcmem_free(g_klx); if (g_kly) rpcmem_free(g_kly); g_klx = g_kly = 0; g_klxn = g_klyn = 0;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nK2Linear(JNIEnv *env, jclass cl, jlong jh, jlong jaux, jlong woff, jint N, jint K, jfloatArray jx, jint M, jfloatArray jout)
{
    k2_t *e = (k2_t *)(intptr_t)jh; maux_t *a = (maux_t *)(intptr_t)jaux; if (!e || !a || K % 32 || N % 64) return -1;
    if ((size_t)woff + (size_t)N * K * 2 > a->len) return -2;
    const int M32 = (M + 31) & ~31; const size_t nW = (size_t)N * K, nX = (size_t)M32 * K, nY = (size_t)M32 * N;
    uint16_t *w = 0, s[64]; int err = -3;
    for (int i = 0; i < g_klwn; i++) if (g_klw[i].map == a->map && g_klw[i].woff == woff && g_klw[i].N == N && g_klw[i].K == K) w = g_klw[i].w;
    if (!w && (w = ralloc(nW * 2)))
    {
        const uint16_t *src = (const uint16_t *)((const uint8_t *)a->map + woff);
        for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) w[((size_t)(n / 32) * (K / 32) + k / 32) * 1024 + (size_t)((k & 31) / 2) * 64 + (n & 31) * 2 + (k & 1)] = src[(size_t)n * K + k];
        if (g_klwn == 8) { rpcmem_free(g_klw[0].w); memmove(g_klw, g_klw + 1, 7 * sizeof(klw_t)); g_klwn = 7; }
        g_klw[g_klwn++] = (klw_t){ a->map, woff, N, K, w };
    }
    if (g_klxn < nX) { if (g_klx) rpcmem_free(g_klx); g_klx = ralloc(nX * 2); g_klxn = g_klx ? nX : 0; }
    if (g_klyn < nY) { if (g_kly) rpcmem_free(g_kly); g_kly = ralloc(nY * 2); g_klyn = g_kly ? nY : 0; }
    uint16_t *x = g_klx, *y = g_kly; float *fx = calloc(nX, 4), *fy = malloc(nY * 4);
    if (w && x && y && fx && fy)
    {
        (*env)->GetFloatArrayRegion(env, jx, 0, M * K, fx); to_act(fx, M32, K, K, K, (__fp16 *)x);
        err = kblock_lin(e->h, w, (int)nW, x, (int)nX, s, 0, y, (int)nY, M32, K, N, 0);
        if (!err) { from_act((const __fp16 *)y, M32, N, fy); (*env)->SetFloatArrayRegion(env, jout, 0, M * N, fy); }
    }
    free(fx); free(fy); return err;
}

/* Real-ESRGAN on the DSP: rgb [3 x H x W] 0..1 -> out [3 x 4H x 4W], from dir's esr.w, esr.prm */
static int esr_run(const char *arch, const char *dir, const float *rgb, int H, int W, float *out, uint32_t *t)
{
    char uri[256], n[1100]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 }; remote_handle64 h = 0;
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um)); if (!err) err = kblock_open(uri, &h); if (err) return err;
    const int wd = (W + 31) & ~31; const size_t M = (size_t)H * wd; long sw = 0, sp = 0; FILE *f; int nc = 0;
    snprintf(n, sizeof(n), "%s/esr.w", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sw = ftell(f); fclose(f); }
    snprintf(n, sizeof(n), "%s/esr.prm", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sp = ftell(f); fclose(f); }
    snprintf(n, sizeof(n), "%s/esr.txt", dir); if ((f = fopen(n, "r"))) { if (fscanf(f, "%d", &nc) != 1) nc = 0; fclose(f); }
    uint16_t *w = 0, *x = 0, *s = 0, *y = 0; float *pr = 0;
    if (!sw || !sp || !nc) err = -2;
    if (!err) { w = ralloc(sw); x = ralloc(M * 32 * 2); s = ralloc(M * 64 * 2 * 2); y = ralloc(M * 64 * 2); pr = (float *)ralloc(sp); if (!w || !x || !s || !y || !pr) err = -3; }
    if (!err && (readf(dir, "esr.w", w, sw) || readf(dir, "esr.prm", pr, sp))) err = -2;
    if (!err)
    {
        float *zc = calloc(M * 32, 4); if (!zc) err = -3;
        else { for (int c = 0; c < 3; c++) for (int r = 0; r < H; r++) for (int q = 0; q < W; q++) zc[((size_t)r * wd + q) * 32 + c] = rgb[((size_t)c * H + r) * W + q];
               to_act(zc, (int)M, 32, 32, 32, (__fp16 *)x); free(zc); err = kblock_esr(h, w, (int)(sw / 2), pr, (int)(sp / 4), x, (int)(M * 32), s, (int)(M * 128), y, (int)(M * 64), H, wd, W, nc, t, 4); }
    }
    if (!err)
    {   /* the pixel shuffle: channel 16 c + 4 sy + sx of pixel (r, q) is the picture's (4 r + sy, 4 q + sx), plus the input as the base */
        const __fp16 *o = (const __fp16 *)y; const size_t ow = (size_t)4 * W, oh = (size_t)4 * H;
        for (int r = 0; r < H; r++) for (int q = 0; q < W; q++)
        {
            const size_t i = (size_t)r * wd + q, b = (i >> 5) * 2 * 1024 + 32 * (i & 30) + (i & 1);
            for (int c = 0; c < 3; c++) { const float base = rgb[((size_t)c * H + r) * W + q];
                for (int sy = 0; sy < 4; sy++) for (int sx = 0; sx < 4; sx++)
                {
                    const int ch = 16 * c + 4 * sy + sx;
                    out[((size_t)c * oh + 4 * r + sy) * ow + 4 * q + sx] = (float)o[b + (ch >= 32 ? 1024 + 2 * (ch - 32) : 2 * ch)] + base;
                } }
        }
    }
    if (w) rpcmem_free(w); if (x) rpcmem_free(x); if (s) rpcmem_free(s); if (y) rpcmem_free(y); if (pr) rpcmem_free(pr);
    kblock_close(h); return err;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nEsr(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jfloatArray jrgb, jint H, jint W, jfloatArray jout, jintArray jtimes)
{
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    char self[512] = ""; Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nEsr, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    jfloat *rgb = (*env)->GetFloatArrayElements(env, jrgb, 0); float *o = malloc((size_t)3 * 16 * H * W * 4); uint32_t t[4] = { 0 }; int err = -3;
    if (o) { err = esr_run(arch, dir, rgb, H, W, o, t); if (!err) (*env)->SetFloatArrayRegion(env, jout, 0, 3 * 16 * H * W, o); free(o); }
    (*env)->ReleaseFloatArrayElements(env, jrgb, rgb, JNI_ABORT); (*env)->SetIntArrayRegion(env, jtimes, 0, 4, (const jint *)t);
    (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir); return err;
}

/* the Qwen-Image VAE's encoder on the DSP: img [3 x H x W] -1..1 -> z [16 x H / 8 x W / 8], before the latent's scaling, from dir's venc.w, venc.prm */
static int qenc_run(const char *arch, const char *dir, const float *img, int H, int W, float *z, uint32_t *t)
{
    char uri[256], n[1100]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 }; remote_handle64 h = 0;
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um)); if (!err) err = kblock_open(uri, &h); if (err) return err;
    const int wd = (W + 31) & ~31, zh = H / 8, zw = W / 8, ow = (zw + 31) & ~31; const size_t M = (size_t)H * wd, Mo = (size_t)zh * ow; long sw = 0, sp = 0; FILE *f;
    snprintf(n, sizeof(n), "%s/venc.w", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sw = ftell(f); fclose(f); }
    snprintf(n, sizeof(n), "%s/venc.prm", dir); if ((f = fopen(n, "rb"))) { fseek(f, 0, SEEK_END); sp = ftell(f); fclose(f); }
    uint16_t *w = 0, *x = 0, *s = 0, *y = 0; float *pr = 0;
    if (!sw || !sp) err = -2;
    if (!err) { w = ralloc(sw); x = ralloc(M * 32 * 2); s = ralloc(M * 384 * 2); y = ralloc(Mo * 32 * 2); pr = (float *)ralloc(sp); if (!w || !x || !s || !y || !pr) err = -3; }
    if (!err && (readf(dir, "venc.w", w, sw) || readf(dir, "venc.prm", pr, sp))) err = -2;
    if (!err)
    {
        float *zc = calloc(M * 32, 4); if (!zc) err = -3;
        else { for (int c = 0; c < 3; c++) for (int r = 0; r < H; r++) for (int q = 0; q < W; q++) zc[((size_t)r * wd + q) * 32 + c] = img[((size_t)c * H + r) * W + q];
               to_act(zc, (int)M, 32, 32, 32, (__fp16 *)x); free(zc); err = kblock_qenc(h, w, (int)(sw / 2), pr, (int)(sp / 4), x, (int)(M * 32), s, (int)(M * 384), y, (int)(Mo * 32), H, wd, W, t, 4); }
    }
    if (!err)
    {
        const __fp16 *o = (const __fp16 *)y;
        for (int c = 0; c < 16; c++) for (int r = 0; r < zh; r++) for (int q = 0; q < zw; q++)
        { const size_t i = (size_t)r * ow + q; z[((size_t)c * zh + r) * zw + q] = (float)o[(i >> 5) * 1024 + 32 * (i & 30) + (i & 1) + 2 * c]; }
    }
    if (w) rpcmem_free(w); if (x) rpcmem_free(x); if (s) rpcmem_free(s); if (y) rpcmem_free(y); if (pr) rpcmem_free(pr);
    kblock_close(h); return err;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nQenc(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jfloatArray jimg, jint H, jint W, jfloatArray jz, jintArray jtimes)
{
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    char self[512] = ""; Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nQenc, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    jfloat *img = (*env)->GetFloatArrayElements(env, jimg, 0); const size_t nz = (size_t)16 * (H / 8) * (W / 8); float *z = malloc(nz * 4); uint32_t t[4] = { 0 }; int err = -3;
    if (z) { err = qenc_run(arch, dir, img, H, W, z, t); if (!err) (*env)->SetFloatArrayRegion(env, jz, 0, (jsize)nz, z); free(z); }
    (*env)->ReleaseFloatArrayElements(env, jimg, img, JNI_ABORT); (*env)->SetIntArrayRegion(env, jtimes, 0, 4, (const jint *)t);
    (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir); return err;
}
