/* kengine.c -- the host side of the engine (JNI, class dev.kroma.open.Engine): Anima, Krea 2, MOSS-Image, FLUX.2 Klein and SDXL on the CDSP.
 * Anima's denoising pass: the app gives the patches, the text context, the RoPE tables, the key count and the adaLN modulations; this lays them out
 * for the HMX, runs the input layer, the blocks (each block's weights read from storage while the one before runs) and the output
 * layer, and returns the prediction [tokens x 64].
 * Files in the weights dir (kconv): weights_b<i>.f16, norms_b<i>.f32, embed.f16, final.f16. DSP side: libkengine_skel_<arch>.so in
 * the app's native library dir (one per HVX architecture; the arch picks which). */
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <libgen.h>
#include "kblock.h"

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "kengine", __VA_ARGS__)
enum { D = 2048, CD = 1024, F = 8192, DH = 128 };

typedef struct {
    remote_handle64 h; char dir[512], wdir[512]; int nb, T, C;      /* wdir: the block weights (dir, or a LoRA baked into them) */
    size_t nW, nI, nS; int fmt;                 /* fmt: the block files' format (fmt_of) */
    uint16_t *wb[2], *in, *scr, *out, *pat, *wemb, *wfin, *res; float *prm, *fmod, *norms;
    int next_ready;                             /* block 0's weights already in wb[0] (read after the step before) */
    int nkept, kept_tried; uint16_t *kept[64]; int kept_ok[64];   /* blocks 0..nkept-1 stay in memory once read, as in the Krea 2 engine: re-reading 3.5 GB each step cost 0.4 s */
    volatile int done;                          /* blocks finished this step, for the app's progress bar */
    size_t wroom; uint16_t *lora; size_t lbytes;                  /* LoRAs applied as they run (lrun.inc): per block A, B after the weights */
} eng_t;

static void klw_flush(void);   /* klein.inc: Engine.K2.linear's kept weights */
static volatile int g_abort;   /* Engine.abort: the block loops leave before their next block (-99) */
static void *ralloc(size_t bytes) { void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes); if (!p) LOG("rpcmem_alloc %zu failed", bytes); return p; }
static int readf(const char *dir, const char *name, void *buf, size_t bytes)
{
    char p[640]; snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "rb"); if (!f) { LOG("cannot open %s", p); return -1; }
    size_t n = fread(buf, 1, bytes, f); fclose(f);
    if (n != bytes) { LOG("%s: %zu of %zu bytes", p, n, bytes); return -1; }
    return 0;
}
/* ------------------------------------------------------------------ the block files' format (kconv's format.txt: q8 / q4, none: fp16)
 * A quantized matrix: its values in the fp16 layout's element order (8-bit integers, or per 256 two nibbles a byte: element i low, 128 + i
 * high, + 8), then per 32 x 32 block its 32 column scales (fp16) -- kblock_imp.c reads it the same way. */
static int fmt_of(const char *dir)
{
    char p[640], b[8] = { 0 }; snprintf(p, sizeof(p), "%s/format.txt", dir); FILE *f = fopen(p, "r"); if (!f) return 0;
    size_t n = fread(b, 1, 7, f); fclose(f); (void)n; return !strncmp(b, "q8", 2) ? 1 : !strncmp(b, "q4", 2) ? 2 : 0;
}
static size_t wbytes(size_t n, int fmt) { return fmt == 1 ? n * 17 / 16 : fmt == 2 ? n * 9 / 16 : n * 2; }
/* one quantized matrix of n values -> fp16 in the layout */
static void deq_mat(const uint8_t *src, __fp16 *dst, size_t n, int fmt)
{
    const __fp16 *sc = (const __fp16 *)(src + (fmt == 1 ? n : n / 2));
    for (size_t b = 0; b < n / 1024; b++)
        for (int e = 0; e < 1024; e++)
        {
            int q;
            if (fmt == 1) q = (int8_t)src[b * 1024 + e];
            else { const int k = e / 256, i = e % 256; const uint8_t v = src[b * 512 + k * 128 + (i & 127)]; q = (i < 128 ? v & 15 : v >> 4) - 8; }
            dst[b * 1024 + e] = (__fp16)((float)sc[b * 32 + (e & 63) / 2] * (float)q);
        }
}
/* fp16 in the layout -> the quantized matrix (ggml's Q8_0 / Q4_0 rounding per block column, as kconv does) */
static void quant_mat(const __fp16 *src, uint8_t *dst, size_t n, int fmt)
{
    __fp16 *sc = (__fp16 *)(dst + (fmt == 1 ? n : n / 2)); int8_t q[1024];
    for (size_t b = 0; b < n / 1024; b++)
    {
        const __fp16 *blk = src + b * 1024;
        for (int c = 0; c < 32; c++)
        {
            float amax = 0, mx = 0;
            for (int r = 0; r < 32; r++) { float v = (float)blk[32 * (r & ~1) + 2 * c + (r & 1)]; if (fabsf(v) > amax) { amax = fabsf(v); mx = v; } }
            __fp16 dh = (__fp16)(fmt == 1 ? amax / 127.f : mx / -8.f); float id = (float)dh != 0.f ? 1.f / (float)dh : 0.f; sc[b * 32 + c] = dh;
            for (int r = 0; r < 32; r++)
            {
                const int e = 32 * (r & ~1) + 2 * c + (r & 1); float v = (float)blk[e] * id;
                if (fmt == 1) q[e] = (int8_t)(v > 127.f ? 127 : v < -127.f ? -127 : lrintf(v));
                else { int u = (int)(v + 8.5f); q[e] = (int8_t)((u > 15 ? 15 : u < 0 ? 0 : u) - 8); }
            }
        }
        if (fmt == 1) memcpy(dst + b * 1024, q, 1024);
        else for (int k = 0; k < 4; k++) for (int i = 0; i < 128; i++) dst[b * 512 + k * 128 + i] = (uint8_t)((q[k * 256 + i] + 8) | (q[k * 256 + 128 + i] + 8) << 4);
    }
}
/* a block file in fp16 (dequantized when the format says so); the matrices' sizes in values, in file order */
static int read_block16(const char *dir, const char *name, __fp16 *w, const size_t *mats, int nm, int fmt)
{
    size_t n = 0; for (int m = 0; m < nm; m++) n += mats[m];
    if (!fmt) return readf(dir, name, w, n * 2);
    uint8_t *raw = malloc(wbytes(n, fmt)); if (!raw) return -1;
    int err = readf(dir, name, raw, wbytes(n, fmt));
    if (!err) { size_t vo = 0, bo = 0; for (int m = 0; m < nm; m++) { deq_mat(raw + bo, w + vo, mats[m], fmt); vo += mats[m]; bo += wbytes(mats[m], fmt); } }
    free(raw); return err;
}
/* and written back (to a .part, then renamed) */
static int write_block16(const char *dir, const char *name, const __fp16 *w, const size_t *mats, int nm, int fmt)
{
    size_t n = 0; for (int m = 0; m < nm; m++) n += mats[m];
    char p[640], q[640]; snprintf(p, sizeof(p), "%s/%s.part", dir, name); snprintf(q, sizeof(q), "%s/%s", dir, name);
    FILE *f = fopen(p, "wb"); if (!f) return -2; int err = 0;
    if (!fmt) err = fwrite(w, 2, n, f) != n;
    else
    {
        uint8_t *raw = malloc(wbytes(n, fmt)); size_t vo = 0, bo = 0;
        for (int m = 0; m < nm; m++) { quant_mat(w + vo, raw + bo, mats[m], fmt); vo += mats[m]; bo += wbytes(mats[m], fmt); }
        err = fwrite(raw, 1, bo, f) != bo; free(raw);
    }
    fclose(f); if (err) return -2; rename(p, q); return 0;
}

typedef struct { eng_t *e; int blk; uint16_t *buf; int err; } rd_t;
static void *read_block(void *a)
{
    rd_t *r = (rd_t *)a; char n[64]; snprintf(n, sizeof(n), "weights_b%d.f16", r->blk);
    r->err = readf(r->e->wdir, n, r->buf, wbytes(r->e->nW, r->e->fmt)); return 0;
}

/* the fp32 rows <-> the HMX layout, on ACT_T threads by row ranges (one thread took 0.4 s each way on a MOSS step, and twice that on Krea) */
enum { ACT_T = 4 };
typedef struct { const float *fs; float *fd; const __fp16 *hs; __fp16 *hd; int t0, t1, K, ld, kvalid; } act_job_t;
static void *to_act_job(void *p)
{
    const act_job_t *j = (const act_job_t *)p; const int K = j->K;
    for (int t = j->t0; t < j->t1; t++)
    {
        const int tile = t >> 5, r = t & 31; __fp16 *row = j->hd + (size_t)tile * (K / 32) * 1024 + 32 * (r & ~1) + (r & 1);
        for (int c = 0; c < K; c++) row[(c >> 5) * 1024 + 2 * (c & 31)] = c < j->kvalid ? (__fp16)j->fs[(size_t)t * j->ld + c] : (__fp16)0.f;
    }
    return 0;
}
static void *from_act_job(void *p)
{
    const act_job_t *j = (const act_job_t *)p; const int N = j->K;
    for (int t = j->t0; t < j->t1; t++)
    {
        const int tile = t >> 5, r = t & 31; const __fp16 *row = j->hs + (size_t)tile * (N / 32) * 1024 + 32 * (r & ~1) + (r & 1);
        for (int c = 0; c < N; c++) j->fd[(size_t)t * N + c] = (float)row[(c >> 5) * 1024 + 2 * (c & 31)];
    }
    return 0;
}
static void act_run(void *(*fn)(void *), act_job_t base, int M)
{
    pthread_t th[ACT_T]; act_job_t jb[ACT_T]; int started[ACT_T];
    for (int i = 0; i < ACT_T; i++) { jb[i] = base; jb[i].t0 = (int)((int64_t)M * i / ACT_T); jb[i].t1 = (int)((int64_t)M * (i + 1) / ACT_T); started[i] = i && M >= 64 && !pthread_create(&th[i], 0, fn, &jb[i]); if (i && !started[i]) fn(&jb[i]); }
    fn(&jb[0]);
    for (int i = 1; i < ACT_T; i++) if (started[i]) pthread_join(th[i], 0);
}
/* row-major float [M x K] (row stride ld) -> fp16 activation layout: per 32-row tile its K/32 blocks; (r, c) at 32 (r & ~1) + 2c + (r & 1) */
static void to_act(const float *src, int M, int K, int ld, int kvalid, __fp16 *dst)
{ act_job_t b = { .fs = src, .hd = dst, .K = K, .ld = ld, .kvalid = kvalid }; act_run(to_act_job, b, M); }
static void from_act(const __fp16 *src, int M, int N, float *dst)
{ act_job_t b = { .hs = src, .fd = dst, .K = N }; act_run(from_act_job, b, M); }

/* an engine and all it holds; for nClose and for nOpen's failures (the app retries a refused open up to six times, so a failed open
   must not leave its buffers behind) */
static void eng_free(eng_t *e)
{
    if (!e) return;
    if (e->h) kblock_close(e->h);
    for (int i = 0; i < e->nkept; i++) rpcmem_free(e->kept[i]);
    void *b[] = { e->wb[0], e->wb[1], e->in, e->scr, e->out, e->pat, e->wemb, e->wfin, e->res, e->prm, e->fmod };
    for (unsigned i = 0; i < sizeof(b) / sizeof(b[0]); i++) if (b[i]) rpcmem_free(b[i]);
    free(e->norms); free(e->lora); free(e);
}
JNIEXPORT jlong JNICALL Java_dev_kroma_open_Engine_nOpen(JNIEnv *env, jclass cl, jstring jlib, jstring jarch, jstring jdir, jstring jwdir, jint nb, jint T, jint C)
{
    const char *lib = (*env)->GetStringUTFChars(env, jlib, 0), *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    eng_t *e = calloc(1, sizeof(eng_t)); e->nb = nb; e->T = T; e->C = C; snprintf(e->dir, sizeof(e->dir), "%s", dir);
    { const char *wd = (*env)->GetStringUTFChars(env, jwdir, 0); snprintf(e->wdir, sizeof(e->wdir), "%s", wd[0] ? wd : dir); (*env)->ReleaseStringUTFChars(env, jwdir, wd); }
    /* lib "": the directory this library was loaded from (the app's native library dir, where the skels are too) */
    char self[512] = ""; if (!lib[0]) { Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nOpen, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); } }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", lib[0] ? lib : self);
    setenv("ADSP_LIBRARY_PATH", path, 1);
    char uri[256]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    (*env)->ReleaseStringUTFChars(env, jlib, lib); (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um));
    if (err) { LOG("unsigned module request failed 0x%x", err); free(e); return 0; }
    if ((err = kblock_open(uri, &e->h)) != 0) { LOG("kblock_open %s failed 0x%x", uri, err); free(e); return 0; }
    e->nW = (size_t)D * D * 6 + (size_t)CD * D * 2 + (size_t)D * F * 2; e->nI = (size_t)T * D + (size_t)C * CD + (size_t)T * 128;
    e->nS = (size_t)T * D * 6 + (size_t)T * F + (size_t)C * D * 2; e->fmt = fmt_of(e->dir);
    /* room after the weights for a LoRA run as it goes (lrun.inc), up to 512 rank slots */
    const size_t lroom = (size_t)(5 * (D + D) + (D + D) + 2 * (CD + D) + (D + D) + 2 * (D + F)) * 512 * 2 + (size_t)(9 * D + F) * 4;   /* + a DoRA's gains */
    e->wroom = lroom; e->wb[0] = ralloc(wbytes(e->nW, e->fmt) + lroom); e->wb[1] = ralloc(wbytes(e->nW, e->fmt) + lroom); e->in = ralloc(e->nI * 2); e->scr = ralloc(e->nS * 2); e->out = ralloc((size_t)T * D * 2);
    e->pat = ralloc((size_t)T * 512 * 2); e->wemb = ralloc(512 * (size_t)D * 2); e->wfin = ralloc((size_t)D * 64 * 2); e->res = ralloc((size_t)T * 64 * 2);
    e->prm = ralloc((size_t)nb * (9 * D + 4 * DH) * 4); e->fmod = ralloc(2 * D * 4); e->norms = malloc((size_t)nb * 4 * DH * 4);
    if (!e->wb[0] || !e->wb[1] || !e->in || !e->scr || !e->out || !e->pat || !e->wemb || !e->wfin || !e->res || !e->prm || !e->fmod || !e->norms) { eng_free(e); return 0; }
    /* the output layer from the baked copy when a LoRA changed it (bake_set writes it there), else the base's */
    char fin[640]; snprintf(fin, sizeof(fin), "%s/final.f16", e->wdir); const char *fdir = access(fin, R_OK) == 0 ? e->wdir : e->dir;
    if (readf(e->dir, "embed.f16", e->wemb, 512 * (size_t)D * 2) || readf(fdir, "final.f16", e->wfin, (size_t)D * 64 * 2)) { eng_free(e); return 0; }
    for (int i = 0; i < nb; i++) { char n[64]; snprintf(n, sizeof(n), "norms_b%d.f32", i); if (readf(e->dir, n, e->norms + (size_t)i * 4 * DH, 4 * DH * 4)) { eng_free(e); return 0; } }
    LOG("open: %d blocks, %d tokens, weights %s, skel %s", nb, T, e->fmt == 1 ? "q8" : e->fmt == 2 ? "q4" : "fp16", uri);
    return (jlong)(intptr_t)e;
}

/* diagnosis switch (<weights dir>/kdebug): the stream's largest value, inf / NaN count and mean |x| after the input layer and each
 * block of a step, to the log -- two phones' logs side by side show the first block where they part */
static void dbg_stats(const char *what, int i, const uint16_t *x, size_t n)
{
    float mx = 0; double sa = 0; size_t bad = 0;
    for (size_t k = 0; k < n; k++) { uint16_t h = x[k]; if ((h & 0x7c00) == 0x7c00) { bad++; continue; } float f = (float)*(const __fp16 *)&x[k]; f = f < 0 ? -f : f; if (f > mx) mx = f; sa += f; }
    LOG("dbg %s %d: max %.2f inf/nan %zu mean %.5f", what, i, mx, bad, sa / (double)n);
}
/* one pass: patches [T x 68], ctx [C x 1024], cos / sin [T x 64] each, kvalid real tokens, mods [nb x 9D] (b<i>_mod in order),
 * fmod [2D] -> res [T x 64]. Returns 0 or an error code. */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nStep(JNIEnv *env, jclass cl, jlong jh, jfloatArray jpat, jfloatArray jctx, jfloatArray jcos, jfloatArray jsin,
                                                          jint kvalid, jfloatArray jmods, jfloatArray jfmod, jfloatArray jres)
{
    eng_t *e = (eng_t *)(intptr_t)jh; if (!e) return -1;
    const int T = e->T, C = e->C, nb = e->nb; int err = 0;
    if (!e->kept_tried)
    {   /* what the DSP's 32-bit space and the free memory allow, leaving 2.5 GB */
        e->kept_tried = 1; const size_t bb = wbytes(e->nW, e->fmt) + (e->lora ? e->lbytes : 0); long long avail = 0; char l[128]; FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) { while (fgets(l, sizeof(l), mf)) if (sscanf(l, "MemAvailable: %lld", &avail) == 1) break; fclose(mf); }
        long long room = (3200LL << 20) - (long long)e->nS * 2 - (long long)e->nI * 2 - 2LL * (long long)bb - (long long)e->T * D * 2, mem = avail * 1024 - (2500LL << 20);
        if (mem < room) room = mem;
        int fit = room > 0 ? (int)(room / (long long)bb) : 0; if (fit > e->nb) fit = e->nb; if (fit > 64) fit = 64;
        for (int i = 0; i < fit; i++) { e->kept[i] = ralloc(bb); if (!e->kept[i]) break; e->nkept = i + 1; }
        LOG("engine: %d of %d blocks kept in memory (%lld MB available)", e->nkept, e->nb, avail / 1024);
    }
    #define ABUF(j) ((j) < e->nkept ? e->kept[j] : e->wb[(j) & 1])
    #define AHAVE(j) ((j) < e->nkept && e->kept_ok[j])
    rd_t r0 = { e, 0, ABUF(0), 0 }; pthread_t th0; int pre = !e->next_ready && !AHAVE(0);
    if (pre) pthread_create(&th0, 0, read_block, &r0);                   /* block 0 while the inputs are laid out */
    float *pat = (*env)->GetFloatArrayElements(env, jpat, 0), *ctx = (*env)->GetFloatArrayElements(env, jctx, 0);
    float *cs = (*env)->GetFloatArrayElements(env, jcos, 0), *sn = (*env)->GetFloatArrayElements(env, jsin, 0);
    float *mods = (*env)->GetFloatArrayElements(env, jmods, 0), *fm = (*env)->GetFloatArrayElements(env, jfmod, 0);
    to_act(pat, T, 512, 68, 68, (__fp16 *)e->pat);
    to_act(ctx, C, CD, CD, CD, (__fp16 *)(e->in + (size_t)T * D));
    float *rope = malloc((size_t)T * 128 * 4);
    for (int t = 0; t < T; t++) { memcpy(rope + (size_t)t * 128, cs + (size_t)t * 64, 256); memcpy(rope + (size_t)t * 128 + 64, sn + (size_t)t * 64, 256); }
    to_act(rope, T, 128, 128, 128, (__fp16 *)(e->in + (size_t)T * D + (size_t)C * CD)); free(rope);
    for (int i = 0; i < nb; i++)
    {
        float *p = e->prm + (size_t)i * (9 * D + 4 * DH);
        memcpy(p, mods + (size_t)i * 9 * D, 9 * D * 4); memcpy(p + 9 * D, e->norms + (size_t)i * 4 * DH, 4 * DH * 4);
    }
    memcpy(e->fmod, fm, 2 * D * 4);
    (*env)->ReleaseFloatArrayElements(env, jpat, pat, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jctx, ctx, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jcos, cs, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jsin, sn, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jmods, mods, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jfmod, fm, JNI_ABORT);
    if (pre) { pthread_join(th0, 0); if (r0.err) return -2; if (0 < e->nkept) e->kept_ok[0] = 1; }
    uint32_t times[26] = { 0 }; double a_dsp = 0, a_post = 0;   /* times[12]: the block's waits for post work */
    e->done = 0;
    if ((err = kblock_embed(e->h, e->pat, T * 512, e->wemb, 512 * D, e->out, T * D, T))) { LOG("embed 0x%x", err); return err; }
    char kd[640]; snprintf(kd, sizeof(kd), "%s/kdebug", e->dir); const int dbg = access(kd, F_OK) == 0;
    if (dbg) dbg_stats("embed", -1, e->out, (size_t)T * D);
    for (int i = 0; i < nb && !err; i++)
    {
        if (g_abort) { err = -99; break; }
        /* the next block's weights (after the last: block 0's for the next step) read while this one runs */
        const int jn = (i + 1) % nb; rd_t nx = { e, jn, ABUF(jn), 0 }; pthread_t th;
        int nextbuf_ok = ((i + 1 < nb) || (nb % 2 == 0) || jn < e->nkept) && !AHAVE(jn);   /* block 0 must land in wb[0] */
        if (nextbuf_ok) pthread_create(&th, 0, read_block, &nx);
        /* a LoRA run as it goes: this block's A and B after its weights, the flag, the slots, the strength (folded in B: 1.000) */
        int fl = 1 | e->fmt << 8; size_t wl = wbytes(e->nW, e->fmt);
        if (e->lora) { memcpy((uint8_t *)ABUF(i) + wl, (const uint8_t *)e->lora + (size_t)i * e->lbytes, e->lbytes); wl += e->lbytes;
                       fl |= 0x1000 | (1000 << 16); }
        err = kblock_run(e->h, ABUF(i), (int)(wl / 2), e->in, (int)e->nI, e->prm + (size_t)i * (9 * D + 4 * DH), 9 * D + 4 * DH,
                         e->scr, (int)e->nS, e->out, T * D, T, C, F, kvalid, fl, times, 26);    /* the stream stays in out */
        a_dsp += times[0] / 1e6; a_post += times[12] / 1e6;
        if (nextbuf_ok) { pthread_join(th, 0); if (nx.err) err = -2; else if (jn < e->nkept) e->kept_ok[jn] = 1; }
        e->next_ready = nextbuf_ok && i + 1 == nb && !nx.err;
        if (err) LOG("block %d: 0x%x", i, err);
        if (dbg && !err) dbg_stats("block", i, e->out, (size_t)T * D);
        e->done = i + 1;
    }
    if (!err && (err = kblock_final(e->h, e->out, T * D, e->wfin, D * 64, e->fmod, 2 * D, e->res, T * 64, T))) LOG("final 0x%x", err);
    if (!err) LOG("anima step: DSP %.2f s, post waits %.3f s, hvx %u vtcm %u KB", a_dsp, a_post, times[24], times[25]);
    if (dbg && !err) dbg_stats("final", 0, e->res, (size_t)T * 64);
    if (err) { e->next_ready = 0; return err; }
    float *res = (*env)->GetFloatArrayElements(env, jres, 0); from_act((const __fp16 *)e->res, T, 64, res); (*env)->ReleaseFloatArrayElements(env, jres, res, 0);
    return 0;
}

/* ------------------------------------------------------------------ Krea 2: the 28 blocks between the app's first.onnx and last.onnx
 * Files (kconv): weights_b<i>.f16 (wq, wk, wv, gate, wo, mlp gate, up, down), params_b<i>.f32 (mod [6D] | norms + 1). */
enum { KD = 6144, KKV = 1536, KF = 16384 };
typedef struct {
    remote_handle64 h; char wdir[512]; int nb, L, fmt; size_t nW, nS;
    uint16_t *wb[2], *in, *scr, *out; float *prm, *cst; int next_ready;
    int nres; uint16_t *res[64]; int res_ok[64]; volatile int done;      /* blocks 0..nres-1 kept in memory once read (as many as the app has room for) */
    uint16_t *lora; size_t lbytes; int lr;      /* LoRAs applied as they run (k2_lora_build): per block A, B after the weights */
    int moss, D;                                /* MOSS-Image's blocks (moss.txt beside the weights, kconv): D 3072, no attention gate */
    int kD;                                     /* Klein's width (klein.txt) */
    int nd;                                     /* FLUX.2 Klein (klein.txt, with moss set): its first nd blocks are double ones */
} k2_t;
#define K2BYTES(e, i) ((i) < (e)->nd ? 2 : 1) * wbytes((e)->nW, (e)->fmt)
static uint16_t *k2_lora_build(const char *set, int nb, int *lr_out, size_t *bytes_out, const char *wdir);
static uint16_t *klein_lora_build(const char *set, int nd, int nb, int kD, int *lr_out, size_t *bytes_out);
static int g_k2moss;
typedef struct { k2_t *e; int blk; uint16_t *buf; int err; } k2rd_t;
/* a block's weights read by 4 threads at once (pread on quarters): one thread reading 870 MB left the Krea steps waiting seconds
 * each for the reads, and the flash does better with more requests in flight. The block after it is handed to the kernel to read
 * ahead (POSIX_FADV_WILLNEED) while this one runs */
typedef struct { int fd; uint8_t *buf; off_t off; size_t n; int err; } k2part_t;
static void *k2_part(void *a) { k2part_t *p = (k2part_t *)a; size_t got = 0;
    while (got < p->n) { const ssize_t r = pread(p->fd, p->buf + got, p->n - got, p->off + (off_t)got); if (r <= 0) { p->err = -1; break; } got += (size_t)r; } return 0; }
static void *k2_read(void *a)
{
    k2rd_t *r = (k2rd_t *)a; char n[64], pth[640]; snprintf(n, sizeof(n), "weights_b%d.f16", r->blk); snprintf(pth, sizeof(pth), "%s/%s", r->e->wdir, n);
    const size_t bytes = K2BYTES(r->e, r->blk); r->err = 0;
    const int fd = open(pth, O_RDONLY); if (fd < 0) { LOG("cannot open %s", pth); r->err = -1; return 0; }
    enum { NP = 4 }; k2part_t pt[NP]; pthread_t th[NP]; const size_t q = (bytes / NP + 4095) & ~(size_t)4095;
    for (int k = 0; k < NP; k++) { const size_t o = (size_t)k * q; pt[k] = (k2part_t){ fd, (uint8_t *)r->buf + o, (off_t)o, o >= bytes ? 0 : (bytes - o < q ? bytes - o : q), 0 }; }
    for (int k = 1; k < NP; k++) pthread_create(&th[k], 0, k2_part, &pt[k]);
    k2_part(&pt[0]); for (int k = 1; k < NP; k++) pthread_join(th[k], 0);
    close(fd); for (int k = 0; k < NP; k++) if (pt[k].err) { LOG("%s: read failed", pth); r->err = -1; }
    /* the next block's file into the page cache meanwhile */
    char nn[64], np2[640]; snprintf(nn, sizeof(nn), "weights_b%d.f16", (r->blk + 1) % r->e->nb); snprintf(np2, sizeof(np2), "%s/%s", r->e->wdir, nn);
    const int fn = open(np2, O_RDONLY); if (fn >= 0) { posix_fadvise(fn, 0, 0, POSIX_FADV_WILLNEED); close(fn); }
    return 0;
}

/* a K2 engine and all it holds; for nK2Close and for nK2Open's failures */
static void k2_free(k2_t *e)
{
    if (!e) return;
    if (e->h) kblock_close(e->h);
    void *b[] = { e->wb[0], e->wb[1], e->in, e->scr, e->out, e->prm }; for (unsigned i = 0; i < 6; i++) if (b[i]) rpcmem_free(b[i]);
    for (int i = 0; i < e->nres; i++) rpcmem_free(e->res[i]);
    free(e->cst); free(e->lora); free(e);
}
JNIEXPORT jlong JNICALL Java_dev_kroma_open_Engine_nK2Open(JNIEnv *env, jclass cl, jstring jarch, jstring jdir, jstring jwdir, jint nb, jint L, jint nres, jstring jset)
{
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0), *wd = (*env)->GetStringUTFChars(env, jwdir, 0);
    k2_t *e = calloc(1, sizeof(k2_t)); e->nb = nb; e->L = L; snprintf(e->wdir, sizeof(e->wdir), "%s", wd[0] ? wd : dir);
    { char mp[600]; snprintf(mp, sizeof(mp), "%s/moss.txt", dir); e->moss = access(mp, F_OK) == 0;
      snprintf(mp, sizeof(mp), "%s/klein.txt", dir); FILE *kf = fopen(mp, "r"); if (kf) { e->moss = 1; int ns_ = 0; e->kD = 3072; if (fscanf(kf, "klein %d %d %d", &e->nd, &ns_, &e->kD) < 1) e->nd = 5; fclose(kf); } }
    const int klein = e->nd > 0, PN = klein ? 12 * e->kD + 512 : (e->moss ? 8 * 3072 : 8 * KD) + 256;     /* a block's parameters */
    const int D = klein ? e->kD : e->moss ? 3072 : KD, KV = e->moss ? D : KKV, F = klein ? 3 * D : e->moss ? 8192 : KF; e->D = D;
    /* LoRAs run as they go: built first, so the buffers have room for them */
    { const char *set = (*env)->GetStringUTFChars(env, jset, 0); if (set[0] && e->nd) e->lora = klein_lora_build(set, e->nd, nb, e->kD, &e->lr, &e->lbytes); else if (set[0]) { g_k2moss = e->moss; e->lora = k2_lora_build(set, nb, &e->lr, &e->lbytes, e->wdir); g_k2moss = 0; } (*env)->ReleaseStringUTFChars(env, jset, set); }
    char self[512] = ""; Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nK2Open, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    char uri[256]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um));
    if (!err) err = kblock_open(uri, &e->h);
    e->nW = (size_t)D * D * (e->moss ? 2 : 3) + (size_t)D * KV * 2 + (size_t)D * F * 3; e->nS = (size_t)L * (D * 4 + KV * 2 + F * 2); e->fmt = fmt_of(dir);
    /* the DSP maps every buffer into a 32-bit address space: the two read buffers, the kept blocks and the scratch stay under 3.3GB */
    const size_t bb = (klein ? 2 : 1) * wbytes(e->nW, e->fmt) + e->lbytes;
    if (klein) e->nS = (size_t)L * (D * 6 + F * 2);
    { const long long room = (3300LL << 20) - (long long)e->nS * 2 - 2LL * (long long)bb; const int fit = room > 0 ? (int)(room / (long long)bb) : 0; if (nres > fit) nres = fit; }
    if (!err) { e->wb[0] = ralloc(bb); e->wb[1] = ralloc(bb); e->in = ralloc((size_t)L * (D + 128) * 2);
                if (klein) e->nS = 16;      /* the DSP keeps its own (kblock_k2run) */
                e->scr = ralloc(e->nS * 2);
                e->out = ralloc((size_t)L * D * 2); e->prm = ralloc((size_t)PN * 4); e->cst = malloc((size_t)nb * PN * 4); }
    if (!err && (!e->wb[0] || !e->wb[1] || !e->in || !e->scr || !e->out || !e->prm || !e->cst)) err = -3;
    /* resident blocks: as many buffers as asked and allocatable (a refused one ends the count) */
    if (nres > nb) nres = nb; if (nres > 64) nres = 64;
    for (int i = 0; i < nres && !err; i++) { e->res[i] = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bb); if (!e->res[i]) break; e->nres = i + 1; }
    for (int i = 0; i < nb && !err; i++) { char n[64]; snprintf(n, sizeof(n), "params_b%d.f32", i);
        if (readf(dir, n, e->cst + (size_t)i * PN, klein ? (size_t)(i < e->nd ? 4 : 2) * 128 * 4 : (size_t)(8 * D + 256) * 4)) err = -1; }
    (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir); (*env)->ReleaseStringUTFChars(env, jwdir, wd);
    if (err) { LOG("k2 open failed 0x%x", err); k2_free(e); return 0; }
    LOG("k2 open%s: %d blocks, %d tokens, weights %s, %d kept in memory, LoRA slots %d", e->moss ? " (MOSS)" : "", nb, L, e->fmt == 1 ? "q8" : e->fmt == 2 ? "q4" : "fp16", e->nres, e->lr);
    return (jlong)(intptr_t)e;
}

/* direct FloatBuffers: seq [L x D] in and out (the blocks' result), tvec this step's modulation, rope cos / sin [L x 64]; two masked key ranges */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nK2Step(JNIEnv *env, jclass cl, jlong jh, jobject jseq, jobject jtvec, jobject jcos, jobject jsin,
                                                            jint ka, jint kb, jint ka2, jint kb2)
{
    k2_t *e = (k2_t *)(intptr_t)jh; if (!e) return -1; const int L = e->L, nb = e->nb, D = e->D; int err = 0;
    const int PN = e->nd ? 12 * D + 512 : 8 * D + 256;
    float *seq = (*env)->GetDirectBufferAddress(env, jseq), *tv = (*env)->GetDirectBufferAddress(env, jtvec);
    const float *cs = (*env)->GetDirectBufferAddress(env, jcos), *sn = (*env)->GetDirectBufferAddress(env, jsin);
    if (!seq || !tv || !cs || !sn) return -4;
    /* block j's buffer: its own when resident, else one of the two rotating ones */
    #define KBUF(j) ((j) < e->nres ? e->res[j] : e->wb[((j) - e->nres) & 1])
    #define KHAVE(j) ((j) < e->nres ? e->res_ok[j] : 0)
    k2rd_t r0 = { e, 0, KBUF(0), 0 }; pthread_t th0; int pre = !e->next_ready && !KHAVE(0); if (pre) pthread_create(&th0, 0, k2_read, &r0);
    /* MOSS: the residual stream carried at 1/16 (fp16 between blocks; at 1024^2 it reaches 51k of fp16's 65504 in block 26):
       the sequence scaled down here, each block's two adaLN gates (what its attention and FFN add to the stream) too, and the sequence
       scaled back at the end. The norms see only the stream's direction (eps 1e-6 is nothing at these sizes), so the same model. */
    float rsi = 1.f;
    if (e->moss) rsi = 1.f / 16;   /* Klein too: Klein 9B's stream passed 65504 at block 7 (13000 at block 0), and v81 blew up */
    if (rsi != 1.f) for (size_t k = 0; k < (size_t)L * D; k++) seq[k] *= rsi;
    to_act(seq, L, D, D, D, (__fp16 *)e->out);
    float *rope = malloc((size_t)L * 128 * 4);
    for (int t = 0; t < L; t++) { memcpy(rope + (size_t)t * 128, cs + (size_t)t * 64, 256); memcpy(rope + (size_t)t * 128 + 64, sn + (size_t)t * 64, 256); }
    to_act(rope, L, 128, 128, 128, (__fp16 *)(e->in + (size_t)L * D)); free(rope);
    if (pre) { pthread_join(th0, 0); if (r0.err) return -2; if (0 < e->nres) e->res_ok[0] = 1; }
    uint32_t times[13] = { 0 }; e->done = 0;
    double stg[13] = { 0 };
    double t_call = 0, t_dsp = 0, t_read = 0, t_lora = 0;      /* where a step's time goes: the calls as the host sees them, the DSP's own, reads, LoRA copies */
    for (int i = 0; i < nb && !err; i++)
    {
        if (g_abort) { err = -99; break; }
        const float *c = e->cst + (size_t)i * PN;
        if (!e->nd) { for (int k = 0; k < 6 * D; k++) e->prm[k] = tv[k] + c[k];
                      memcpy(e->prm + 6 * D, c + 6 * D, (size_t)(2 * D + 256) * 4);
                      if (rsi != 1.f) for (int k = 0; k < D; k++) { e->prm[2 * D + k] *= rsi; e->prm[5 * D + k] *= rsi; } }   /* the gates (pre scale, shift, gate, post scale, shift, gate) */
        else if (i < e->nd) { memcpy(e->prm, tv, (size_t)12 * D * 4); memcpy(e->prm + 12 * D, c, 512 * 4);    /* tv: double blocks' text, image [6D each] | single's [3D] */
                              if (rsi != 1.f) for (int g = 0; g < 4; g++) for (int k = 0; k < D; k++) e->prm[(size_t)(g / 2 * 6 + 2 + g % 2 * 3) * D + k] *= rsi; }   /* each stream's two gates (scale, shift, gate x 2) */
        else { memcpy(e->prm, tv + 12 * D, (size_t)3 * D * 4); memset(e->prm + 3 * D, 0, (size_t)3 * D * 4); for (int k = 0; k < 2 * D; k++) e->prm[6 * D + k] = 1.f; memcpy(e->prm + 8 * D, c, 256 * 4);
               if (rsi != 1.f) for (int k = 0; k < D; k++) e->prm[2 * D + k] *= rsi; }   /* the single block's gate */
        /* the next block (after the last: block 0 for the next step) read now unless it is resident and loaded, or its rotating
         * buffer is the one this block runs from */
        const int j = (i + 1) % nb; k2rd_t nx = { e, j, KBUF(j), 0 }; pthread_t th;
        const int rd = !KHAVE(j) && KBUF(j) != KBUF(i); if (rd) pthread_create(&th, 0, k2_read, &nx);
        int fl = 1 | e->fmt << 8 | (e->moss ? 4 : 0) | (e->nd ? (i < e->nd ? 0x800 : 0x400) : 0); size_t wl = K2BYTES(e, i);
        struct timespec q0, q1, q2, q3; clock_gettime(CLOCK_MONOTONIC, &q0);
        if (e->lora) { memcpy((uint8_t *)KBUF(i) + wl, (const uint8_t *)e->lora + (size_t)i * e->lbytes, e->lbytes); wl += e->lbytes;
                       fl |= 0x1000 | (1000 << 16); }
        /* the diagnosis's switches, read at block 0: kdebug (each block's output stats) and kstage "<block> <stage>" (that block's
         * intermediate instead of its output, kblock_k2run's dstage: 1 the attention gate's product (Krea), 2 the attention's output, 3 the out
         * projection's, 4 / 5 q / k after their norm and RoPE, 6 v (Klein: the MLP hidden), 7 the VTCM figures) */
        static int k2dbg = -1, k2stb = -1, k2stg = 0;
        if (i == 0) { char dp[600]; snprintf(dp, sizeof(dp), "%s/kdebug", e->wdir); k2dbg = access(dp, F_OK) == 0; k2stb = -1;
                      snprintf(dp, sizeof(dp), "%s/kstage", e->wdir); FILE *sf = k2dbg ? fopen(dp, "r") : 0;
                      if (sf) { if (fscanf(sf, "%d %d", &k2stb, &k2stg) != 2) k2stb = -1; fclose(sf); } }
        if (k2dbg > 0 && i == k2stb && !(fl & 0x1000)) fl |= (k2stg & 7) << 16;
        clock_gettime(CLOCK_MONOTONIC, &q1);
        err = kblock_k2run(e->h, KBUF(i), (int)(wl / 2), e->in, L * (D + 128), e->prm, PN, e->scr, (int)e->nS, e->out, L * D, L, ka, kb, ka2, kb2, fl, times, 13);
        clock_gettime(CLOCK_MONOTONIC, &q2);
        /* the per-block diagnosis (Advanced.diagLog, or an empty kdebug in the weights' folder): each block's output,
         * max |x| and the non-finite count, to find the block where a chip's arithmetic first breaks */
        if (k2dbg > 0) dbg_stats(k2stb == i ? "k2 stage" : "k2 out", i, e->out, (size_t)L * D);
        if (rd) { pthread_join(th, 0); if (nx.err) err = -2; else if (j < e->nres) e->res_ok[j] = 1; }
        clock_gettime(CLOCK_MONOTONIC, &q3);
        #define SECS(a, b) ((b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9)
        t_lora += SECS(q0, q1); t_call += SECS(q1, q2); t_read += SECS(q2, q3); t_dsp += times[0] / 1e6; for (int q = 1; q < 13; q++) stg[q] += times[q] / 1e6;
        #undef SECS
        e->next_ready = i + 1 == nb && rd && !nx.err;
        if (err) LOG("k2 block %d: 0x%x", i, err);
        e->done = i + 1;
    }
    LOG("k2 step: calls %.2f s (DSP %.2f), waiting for reads %.2f, LoRA copies %.2f", t_call, t_dsp, t_read, t_lora);
    LOG("k2 stages: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f, post waits %.3f", stg[1], stg[2], stg[3], stg[4], stg[5], stg[6], stg[7], stg[8], stg[9], stg[10], stg[11], stg[12]);
    if (!err) from_act((const __fp16 *)e->out, L, D, seq);
    if (rsi != 1.f) for (size_t k = 0; k < (size_t)L * D; k++) seq[k] /= rsi;
    /* an error drops the next step's block 0 read ahead (a step that went through keeps it) */
    if (err) e->next_ready = 0;
    #undef KBUF
    #undef KHAVE
    return err;
}

/* Krea 2 LoRA baked in: <lora>/blocks.<i>.<attn.wq ...>.A.f16 [r x in], .B.f16 [out x r] (LoraImport.convert), times strength.
 * The engine's rows of wq / wk are head-permuted ([even | odd dims]) and mlp.down is stored as four K = 4096 quarters. */
typedef struct { const __fp16 *a, *b; __fp16 *w; int K, N, R, n0, n1, heads, ksplit; float st; } k2bake_t;
static void *k2bake_part(void *p)
{
    k2bake_t *t = (k2bake_t *)p; const int K = t->K, Kq = K / t->ksplit; float *row = malloc((size_t)K * 4), *br = malloc((size_t)t->R * 4);
    for (int n = t->n0; n < t->n1; n++)
    {
        int src = n; if (t->heads) { int h = n / 128, d = n % 128; src = h * 128 + (d < 64 ? 2 * d : 2 * (d - 64) + 1); }   /* the original row */
        int any = 0; for (int r = 0; r < t->R; r++) { br[r] = (float)t->b[(size_t)src * t->R + r] * t->st; any |= br[r] != 0.f; }
        if (!any) continue;
        memset(row, 0, (size_t)K * 4);
        for (int r = 0; r < t->R; r++) { const float bv = br[r]; if (bv == 0.f) continue; const __fp16 *ar = t->a + (size_t)r * K; for (int k = 0; k < K; k++) row[k] += bv * (float)ar[k]; }
        for (int k = 0; k < K; k++)
        {
            const int q = k / Kq, kk = k % Kq;
            __fp16 *e = t->w + (size_t)q * t->N * Kq + (size_t)((n >> 6) * 2 + ((n >> 5) & 1)) * (Kq / 32) * 1024 + (kk >> 5) * 1024 + 32 * ((kk & 31) & ~1) + 2 * (n & 31) + (kk & 1);
            *e = (__fp16)((float)*e + row[k]);
        }
    }
    free(row); free(br); return 0;
}
static __fp16 *read16(const char *dir, const char *name, size_t *count)
{
    char p[700]; snprintf(p, sizeof(p), "%s/%s", dir, name); FILE *f = fopen(p, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); __fp16 *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return 0; } fclose(f); *count = (size_t)n / 2; return b;
}
/* blocks baked so far (nBake / nK2Bake / nBakeSet / nTxtBake), read by the app's progress display from another thread */
static volatile int g_bake_done;
/* the bake itself, without JNI */
static int k2_bake(const char *base, const char *lora, const char *out, int nb, float strength)
{
    static const char *mats[8] = { "attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down" };
    static const int K8[8] = { KD, KD, KD, KD, KD, KD, KD, KF }, N8[8] = { KD, KKV, KKV, KD, KD, KF, KF, KD }, H8[8] = { 48, 12, 0, 0, 0, 0, 0, 0 };
    const size_t nW = (size_t)KD * KD * 3 + (size_t)KD * KKV * 2 + (size_t)KD * KF * 3; __fp16 *w = malloc(nW * 2); int used = 0, err = w ? 0 : -3;
    const int fmt = fmt_of(base);
    size_t msz[11]; for (int m = 0; m < 7; m++) msz[m] = (size_t)K8[m] * N8[m]; for (int q = 0; q < 4; q++) msz[7 + q] = (size_t)KF / 4 * KD;
    if (fmt && !err) { char p[640]; snprintf(p, sizeof(p), "%s/format.txt", out); FILE *f = fopen(p, "w"); if (f) { fputs(fmt == 1 ? "q8\n" : "q4\n", f); fclose(f); } }
    g_bake_done = 0;
    for (int i = 0; i < nb && !err; i++)
    {
        char n[96]; snprintf(n, sizeof(n), "weights_b%d.f16", i);
        if (read_block16(base, n, w, msz, 11, fmt)) { err = -1; break; }
        size_t off = 0;
        for (int m = 0; m < 8; m++)
        {
            const int K = K8[m], N = N8[m]; char na[128], nbn[128]; size_t ca = 0, cb = 0;
            snprintf(na, sizeof(na), "blocks.%d.%s.A.f16", i, mats[m]); snprintf(nbn, sizeof(nbn), "blocks.%d.%s.B.f16", i, mats[m]);
            __fp16 *a = read16(lora, na, &ca), *b = a ? read16(lora, nbn, &cb) : 0;
            const int R = a ? (int)(ca / K) : 0;
            if (a && b && R > 0 && ca == (size_t)R * K && cb == (size_t)N * R)
            {
                pthread_t th[4]; k2bake_t bt[4];
                for (int t = 0; t < 4; t++) { bt[t] = (k2bake_t){ a, b, w + off, K, N, R, N * t / 4, N * (t + 1) / 4, H8[m], m == 7 ? 4 : 1, strength }; pthread_create(&th[t], 0, k2bake_part, &bt[t]); }
                for (int t = 0; t < 4; t++) pthread_join(th[t], 0); used++;
            }
            free(a); free(b); off += (size_t)K * N;
        }
        if (write_block16(out, n, w, msz, 11, fmt)) err = -2; g_bake_done = i + 1;
    }
    free(w);
    return err ? err : used;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nK2Bake(JNIEnv *env, jclass cl, jstring jbase, jstring jlora, jstring jout, jint nb, jfloat strength)
{
    const char *base = (*env)->GetStringUTFChars(env, jbase, 0), *lora = (*env)->GetStringUTFChars(env, jlora, 0), *out = (*env)->GetStringUTFChars(env, jout, 0);
    int r = k2_bake(base, lora, out, nb, strength);
    (*env)->ReleaseStringUTFChars(env, jbase, base); (*env)->ReleaseStringUTFChars(env, jlora, lora); (*env)->ReleaseStringUTFChars(env, jout, out);
    return r;
}

JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nK2Close(JNIEnv *env, jclass cl, jlong jh)
{
    k2_t *e = (k2_t *)(intptr_t)jh; if (!e) return; klw_flush(); k2_free(e);
}

/* ------------------------------------------------------------------ LoRA baked into the block weights
 * The app's active LoRA (LoraImport.activateAnima): b<i>_<key>_A.f32 [64 x in], _B.f32 [out x 64], the strength in B, zero past the
 * LoRA's own rank. W^T [in x out] += (B A)^T straight in the HMX layout, written as <out>/weights_b<i>.f16. */
static const char *LKEY[10] = { "sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o", "mlp1", "mlp2" };
static const int LIN[10] = { D, D, D, D, D, CD, CD, D, D, F }, LOUT[10] = { D, D, D, D, D, D, D, D, F, D };
typedef struct { const float *a, *b; __fp16 *w; int K, N, R, n0, n1; } bake_t;
static void *bake_part(void *p)
{
    bake_t *t = (bake_t *)p; const int K = t->K, kb = K / 32; float *row = malloc((size_t)K * 4);
    for (int n = t->n0; n < t->n1; n++)
    {
        memset(row, 0, (size_t)K * 4);
        for (int r = 0; r < t->R; r++) { const float bv = t->b[(size_t)n * 64 + r]; if (bv == 0.f) continue; const float *ar = t->a + (size_t)r * K; for (int k = 0; k < K; k++) row[k] += bv * ar[k]; }
        const int g = n >> 6, hh = (n >> 5) & 1, c = n & 31; __fp16 *base = t->w + (size_t)(g * 2 + hh) * kb * 1024 + 2 * c;
        for (int k = 0; k < K; k++) { __fp16 *e = base + (k >> 5) * 1024 + 32 * ((k & 31) & ~1) + ((k & 31) & 1); *e = (__fp16)((float)*e + row[k]); }
    }
    free(row); return 0;
}
static float *readall(const char *dir, const char *name, size_t floats)
{
    char p[640]; snprintf(p, sizeof(p), "%s/%s", dir, name); FILE *f = fopen(p, "rb"); if (!f) return 0;
    float *b = malloc(floats * 4); size_t n = fread(b, 4, floats, f); fclose(f); if (n != floats) { free(b); return 0; } return b;
}
/* base: kconv's files; lora: the active LoRA folder; out: where the baked weights_b<i>.f16 go. Returns the number of LoRA pairs used, or < 0. */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nBake(JNIEnv *env, jclass cl, jstring jbase, jstring jlora, jstring jout, jint nb)
{
    const char *base = (*env)->GetStringUTFChars(env, jbase, 0), *lora = (*env)->GetStringUTFChars(env, jlora, 0), *out = (*env)->GetStringUTFChars(env, jout, 0);
    const size_t nW = (size_t)D * D * 6 + (size_t)CD * D * 2 + (size_t)D * F * 2; __fp16 *w = malloc(nW * 2); int used = 0, err = 0;
    const int fmt = fmt_of(base);
    size_t mats[10]; for (int m = 0; m < 10; m++) mats[m] = (size_t)LIN[m] * LOUT[m];
    if (fmt && !err) { char p[640]; snprintf(p, sizeof(p), "%s/format.txt", out); FILE *f = fopen(p, "w"); if (f) { fputs(fmt == 1 ? "q8\n" : "q4\n", f); fclose(f); } }
    g_bake_done = 0;
    for (int i = 0; i < nb && !err; i++)
    {
        char n[96]; snprintf(n, sizeof(n), "weights_b%d.f16", i);
        if (read_block16(base, n, w, mats, 10, fmt)) { err = -1; break; }
        size_t off = 0;
        for (int m = 0; m < 10; m++)
        {
            const int K = LIN[m], N = LOUT[m]; char na[96], nb2[96];
            snprintf(na, sizeof(na), "b%d_%s_A.f32", i, LKEY[m]); snprintf(nb2, sizeof(nb2), "b%d_%s_B.f32", i, LKEY[m]);
            float *a = readall(lora, na, (size_t)64 * K), *b = a ? readall(lora, nb2, (size_t)N * 64) : 0;
            if (a && b)
            {
                int R = 64; while (R > 0) { int z = 1; for (int k = 0; k < K && z; k++) z = a[(size_t)(R - 1) * K + k] == 0.f; if (!z) break; R--; }   /* the LoRA's own rank */
                if (R) { pthread_t th[4]; bake_t bt[4];
                    for (int t = 0; t < 4; t++) { bt[t] = (bake_t){ a, b, w + off, K, N, R, N * t / 4, N * (t + 1) / 4 }; pthread_create(&th[t], 0, bake_part, &bt[t]); }
                    for (int t = 0; t < 4; t++) pthread_join(th[t], 0); used++; }
            }
            free(a); free(b); off += (size_t)K * N;
        }
        if (write_block16(out, n, w, mats, 10, fmt)) err = -2; g_bake_done = i + 1;
    }
    free(w);
    (*env)->ReleaseStringUTFChars(env, jbase, base); (*env)->ReleaseStringUTFChars(env, jlora, lora); (*env)->ReleaseStringUTFChars(env, jout, out);
    return err ? err : used;
}

JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nClose(JNIEnv *env, jclass cl, jlong jh)
{
    eng_free((eng_t *)(intptr_t)jh);
}

JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nAbort(JNIEnv *env, jclass cl, jint on) { g_abort = on; }
/* blocks baked so far, read from another thread */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nBakeProgress(JNIEnv *env, jclass cl) { return g_bake_done; }
/* blocks finished in the step running now (read from another thread while nStep / nK2Step runs) */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nProgress(JNIEnv *env, jclass cl, jlong jh) { eng_t *e = (eng_t *)(intptr_t)jh; return e ? e->done : 0; }
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nK2Progress(JNIEnv *env, jclass cl, jlong jh) { k2_t *e = (k2_t *)(intptr_t)jh; return e ? e->done : 0; }

#include "lbake.inc"
#include "lrun.inc"
#include "moss.inc"
#include "qte.inc"
#include "kte.inc"
#include "klein.inc"
#include "kxl_host.inc"
