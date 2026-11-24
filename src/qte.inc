/* qte.inc (included by kengine.c) -- MOSS-Image's text encoder, the stock Qwen3.5-9B text layers, on the phone: every product on the
 * DSP (kblock_lin, the layer's weights read from storage while the layer before runs), the rest here in fp32 -- the RMSNorms, the
 * Gated DeltaNet's conv1d and recurrence (24 of the 32 layers), the gated attention (8 layers, causal), the residuals. The prompt is
 * short (tens to hundreds of tokens), so this side is a few percent of the work; the products and reading 9GB of weights are the rest.
 * Files (made by Kroma's converter tools, not in this repository): qte_l<i>.w (products in kconv's layout and format), qte_l<i>.f32 (norms, conv, decay), qte_emb.bf16, qte_norm.f32.
 * Output: the last hidden state (after the final norm) [T x 4096], what MOSS-Image's text refiner takes. */
#include <time.h>
/* a matrix after one of n values in format fmt */
static inline const uint16_t *WPH(const uint16_t *base, size_t n, int fmt) { return (const uint16_t *)((const uint8_t *)base + wbytes(n, fmt)); }
enum { QD = 4096, QF = 12288, QL = 32, QV = 248320, QH = 16, QKV = 4, QHD = 256, QROT = 64, QLK = 16, QLV = 32, QLD = 128,
       QIN_L = 2 * QLK * QLD + QLV * QLD + 2 * QLV + QLV * QLD,      /* linear: qkv 8192 | z 4096 | b 32 | a 32 = 12352 */
       QIN_A = 2 * QH * QHD + 2 * QKV * QHD };                       /* attention: q + gate 8192 | k 1024 | v 1024 = 10240 */
#define QKIND_LIN(i) (((i) + 1) % 4 != 0)                           /* full attention every fourth layer */
#define QEPS 1e-6f

typedef struct { int fmt; char dir[512]; size_t wl[QL]; uint16_t *wb[2]; } qte_t;
static size_t qte_wbytes(int i, int fmt)
{
    const size_t in = (size_t)QD * (QKIND_LIN(i) ? QIN_L : QIN_A);
    return wbytes(in, fmt) + wbytes((size_t)QD * QD, fmt) + 3 * wbytes((size_t)QD * QF, fmt);
}
static size_t qte_fbytes(int i) { return (size_t)(2 * QD + (QKIND_LIN(i) ? (2 * QLK * QLD + QLV * QLD) * 4 + 2 * QLV + QLD : 2 * QHD)) * 4; }

typedef struct { const char *dir; int i; uint16_t *buf; size_t bytes; int err; } qrd_t;
static void *qte_read(void *a) { qrd_t *r = (qrd_t *)a; char n[64]; snprintf(n, sizeof(n), "qte_l%d.w", r->i); r->err = readf(r->dir, n, r->buf, r->bytes); return 0; }

static void rms(const float *x, float *o, int n, const float *w)       /* o = x / rms(x) * w */
{
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    const float k = 1.f / sqrtf((float)(s / n) + QEPS); for (int i = 0; i < n; i++) o[i] = x[i] * k * (w ? w[i] : 1.f);
}
static inline float silu(float v) { return v / (1.f + expf(-v)); }
static inline float sigm(float v) { return 1.f / (1.f + expf(-v)); }

/* the Gated DeltaNet's recurrence for value heads h0..h1-1 (fp32, token by token, as torch_recurrent_gated_delta_rule) */
typedef struct { const float *q, *k, *v, *g, *beta; float *o; int T, h0, h1; } gdn_t;
static void *gdn_part(void *a_)
{
    gdn_t *a = (gdn_t *)a_; float *S = malloc((size_t)QLD * QLD * 4);
    for (int h = a->h0; h < a->h1; h++)
    {
        const int hk = h / (QLV / QLK); memset(S, 0, (size_t)QLD * QLD * 4);         /* S [k dim x v dim] */
        for (int t = 0; t < a->T; t++)
        {
            const float *q = a->q + ((size_t)t * QLK + hk) * QLD, *k = a->k + ((size_t)t * QLK + hk) * QLD, *v = a->v + ((size_t)t * QLV + h) * QLD;
            const float dec = expf(a->g[t * QLV + h]), be = a->beta[t * QLV + h];
            float mem[QLD], d[QLD];
            for (int j = 0; j < QLD; j++) mem[j] = 0.f;
            for (int i = 0; i < QLD; i++) { float *row = S + (size_t)i * QLD; const float ki = k[i]; for (int j = 0; j < QLD; j++) { row[j] *= dec; mem[j] += row[j] * ki; } }
            for (int j = 0; j < QLD; j++) d[j] = (v[j] - mem[j]) * be;
            float *o = a->o + ((size_t)t * QLV + h) * QLD; for (int j = 0; j < QLD; j++) o[j] = 0.f;
            for (int i = 0; i < QLD; i++) { float *row = S + (size_t)i * QLD; const float ki = k[i], qi = q[i]; for (int j = 0; j < QLD; j++) { row[j] += ki * d[j]; o[j] += row[j] * qi; } }
        }
    }
    free(S); return 0;
}

/* the whole encoder: ids [T] -> out [T x 4096]. h is an open kblock session; 0 or an error */
static int qte_encode(remote_handle64 h, qte_t *e, const int *ids, int T, float *out, void (*progress)(int))
{
    const int M = (T + 31) / 32 * 32; int err = 0;
    float *x = calloc((size_t)M * QD, 4), *t0 = malloc((size_t)M * QD * 4), *y = malloc((size_t)M * QIN_L * 4);
    uint16_t *act = ralloc((size_t)M * QIN_L * 2), *yo = ralloc((size_t)M * QIN_L * 2), *scr = ralloc((size_t)M * QF * 2);
    if (!x || !t0 || !y || !act || !yo || !scr) { err = -3; goto done; }
    {   /* the embeddings, bf16 rows */
        char p[640]; snprintf(p, sizeof(p), "%s/qte_emb.bf16", e->dir); FILE *f = fopen(p, "rb"); if (!f) { err = -1; goto done; }
        uint16_t row[QD];
        for (int t = 0; t < T; t++)
        {
            if (ids[t] < 0 || ids[t] >= QV || fseeko(f, (off_t)ids[t] * QD * 2, SEEK_SET) || fread(row, 2, QD, f) != QD) { fclose(f); err = -1; goto done; }
            for (int c = 0; c < QD; c++) { uint32_t u = (uint32_t)row[c] << 16; memcpy(&x[(size_t)t * QD + c], &u, 4); }
        }
        fclose(f);
    }
    float *prm = malloc(qte_fbytes(0) > qte_fbytes(3) ? qte_fbytes(0) : qte_fbytes(3));
    qrd_t r0 = { e->dir, 0, e->wb[0], e->wl[0], 0 }; qte_read(&r0); if (r0.err) { err = -2; free(prm); goto done; }
    const int fl = e->fmt << 8;
    for (int i = 0; i < QL && !err; i++)
    {
        if (g_abort) { err = -99; break; }   /* Engine.abort: a stop leaves between layers */
        uint16_t *w = e->wb[i & 1]; pthread_t th; qrd_t nx = { e->dir, i + 1, e->wb[(i + 1) & 1], i + 1 < QL ? e->wl[i + 1] : 0, 0 };
        if (i + 1 < QL) pthread_create(&th, 0, qte_read, &nx);
        { char n[64]; snprintf(n, sizeof(n), "qte_l%d.f32", i); if (readf(e->dir, n, prm, qte_fbytes(i))) { err = -1; goto qte_join; } }
        const float *wn1 = prm, *wn2 = prm + QD, *ex = prm + 2 * QD;
        const int lin = QKIND_LIN(i), NIN = lin ? QIN_L : QIN_A;
        const uint16_t *w_in = w, *w_out = WPH(w_in, (size_t)QD * NIN, e->fmt), *w_mlp = WPH(w_out, (size_t)QD * QD, e->fmt);
        /* the mixer's input products */
        for (int t = 0; t < M; t++) { if (t < T) rms(x + (size_t)t * QD, t0 + (size_t)t * QD, QD, wn1); else memset(t0 + (size_t)t * QD, 0, QD * 4); }
        to_act(t0, M, QD, QD, QD, (__fp16 *)act);
        if ((err = kblock_lin(h, w_in, (int)(wbytes((size_t)QD * NIN, e->fmt) / 2), act, M * QD, scr, M * QF, yo, M * NIN, M, QD, NIN, fl))) goto qte_join;
        from_act((const __fp16 *)yo, M, NIN, y);
        float *mix = t0;                                                         /* [T x 4096] into the out product */
        if (lin)
        {
            const float *cw = ex, *dtb = ex + 8192 * 4, *alog = dtb + QLV, *nw = alog + QLV;
            float *q = malloc((size_t)T * 2048 * 4), *k = malloc((size_t)T * 2048 * 4), *v = malloc((size_t)T * 4096 * 4), *g = malloc((size_t)T * QLV * 4), *be = malloc((size_t)T * QLV * 4), *o = malloc((size_t)T * 4096 * 4);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < 8192; c++)
                {   /* causal conv1d (kernel 4) + silu over qkv's channels */
                    float s = 0.f; for (int j = 0; j < 4; j++) { const int tt = t - 3 + j; if (tt >= 0) s += cw[c * 4 + j] * y[(size_t)tt * NIN + c]; }
                    s = silu(s);
                    if (c < 2048) q[(size_t)t * 2048 + c] = s; else if (c < 4096) k[(size_t)t * 2048 + c - 2048] = s; else v[(size_t)t * 4096 + c - 4096] = s;
                }
            for (int t = 0; t < T; t++)
            {
                for (int hh = 0; hh < QLK; hh++)
                {   /* l2norm of q and k per head; q scaled by 1 / sqrt(128) */
                    float *qh = q + ((size_t)t * QLK + hh) * QLD, *kh = k + ((size_t)t * QLK + hh) * QLD; double sq = 0, sk = 0;
                    for (int j = 0; j < QLD; j++) { sq += qh[j] * qh[j]; sk += kh[j] * kh[j]; }
                    const float iq = 1.f / sqrtf((float)sq + QEPS) / sqrtf((float)QLD), ik = 1.f / sqrtf((float)sk + QEPS);
                    for (int j = 0; j < QLD; j++) { qh[j] *= iq; kh[j] *= ik; }
                }
                for (int hh = 0; hh < QLV; hh++)
                {
                    const float bv = y[(size_t)t * NIN + 12288 + hh], av = y[(size_t)t * NIN + 12320 + hh];
                    be[t * QLV + hh] = sigm(bv);
                    const float sp = av + dtb[hh]; g[t * QLV + hh] = -expf(alog[hh]) * (sp > 20.f ? sp : log1pf(expf(sp)));
                }
            }
            enum { NT = 8 }; pthread_t ts[NT]; gdn_t ga[NT];
            for (int p = 0; p < NT; p++) { ga[p] = (gdn_t){ q, k, v, g, be, o, T, p * QLV / NT, (p + 1) * QLV / NT }; pthread_create(&ts[p], 0, gdn_part, &ga[p]); }
            for (int p = 0; p < NT; p++) pthread_join(ts[p], 0);
            for (int t = 0; t < T; t++)
                for (int hh = 0; hh < QLV; hh++)
                {   /* RMSNorm per head, times silu(z) */
                    const float *oh = o + ((size_t)t * QLV + hh) * QLD, *z = y + (size_t)t * NIN + 8192 + hh * QLD; float *m = mix + (size_t)t * QD + hh * QLD;
                    double s = 0; for (int j = 0; j < QLD; j++) s += oh[j] * oh[j];
                    const float kk = 1.f / sqrtf((float)(s / QLD) + QEPS);
                    for (int j = 0; j < QLD; j++) m[j] = oh[j] * kk * nw[j] * silu(z[j]);
                }
            free(q); free(k); free(v); free(g); free(be); free(o);
        }
        else
        {
            const float *qn = ex, *kn = ex + QHD;
            float *q = malloc((size_t)T * QH * QHD * 4), *gt = malloc((size_t)T * QH * QHD * 4), *k = malloc((size_t)T * QKV * QHD * 4), *v = malloc((size_t)T * QKV * QHD * 4);
            float inv[QROT / 2]; for (int j = 0; j < QROT / 2; j++) inv[j] = 1.f / powf(10000000.f, (float)(2 * j) / QROT);
            for (int t = 0; t < T; t++)
            {
                const float *yr = y + (size_t)t * NIN;
                for (int hh = 0; hh < QH; hh++) { memcpy(q + ((size_t)t * QH + hh) * QHD, yr + hh * 2 * QHD, QHD * 4); memcpy(gt + ((size_t)t * QH + hh) * QHD, yr + hh * 2 * QHD + QHD, QHD * 4); }
                memcpy(k + (size_t)t * QKV * QHD, yr + 2 * QH * QHD, QKV * QHD * 4); memcpy(v + (size_t)t * QKV * QHD, yr + 2 * QH * QHD + QKV * QHD, QKV * QHD * 4);
                for (int hh = 0; hh < QH + QKV; hh++)
                {   /* q / k RMSNorm (1 + w) per head, then RoPE on the first 64 dims (rotate_half over them) */
                    float *a = hh < QH ? q + ((size_t)t * QH + hh) * QHD : k + ((size_t)t * QKV + hh - QH) * QHD; rms(a, a, QHD, hh < QH ? qn : kn);
                    for (int j = 0; j < QROT / 2; j++)
                    {
                        const float ang = (float)t * inv[j], c = cosf(ang), s = sinf(ang), x1 = a[j], x2 = a[j + QROT / 2];
                        a[j] = x1 * c - x2 * s; a[j + QROT / 2] = x2 * c + x1 * s;
                    }
                }
            }
            const float sc = 1.f / sqrtf((float)QHD); float *p = malloc((size_t)T * 4);
            for (int hh = 0; hh < QH; hh++)
                for (int t = 0; t < T; t++)
                {   /* causal attention, 4 query heads to a kv head, times sigmoid(gate) */
                    const float *qh = q + ((size_t)t * QH + hh) * QHD; const int hk = hh / (QH / QKV); float mx = -1e30f, z = 0.f;
                    for (int s = 0; s <= t; s++) { const float *kh = k + ((size_t)s * QKV + hk) * QHD; float d = 0.f; for (int j = 0; j < QHD; j++) d += qh[j] * kh[j]; p[s] = d * sc; if (p[s] > mx) mx = p[s]; }
                    for (int s = 0; s <= t; s++) { p[s] = expf(p[s] - mx); z += p[s]; }
                    float *m = mix + (size_t)t * QD + hh * QHD; const float *gh = gt + ((size_t)t * QH + hh) * QHD;
                    for (int j = 0; j < QHD; j++) { float a = 0.f; for (int s = 0; s <= t; s++) a += p[s] * v[((size_t)s * QKV + hk) * QHD + j]; m[j] = a / z * sigm(gh[j]); }
                }
            free(p); free(q); free(gt); free(k); free(v);
        }
        for (int t = T; t < M; t++) memset(mix + (size_t)t * QD, 0, QD * 4);
        to_act(mix, M, QD, QD, QD, (__fp16 *)act);
        if ((err = kblock_lin(h, w_out, (int)(wbytes((size_t)QD * QD, e->fmt) / 2), act, M * QD, scr, M * QF, yo, M * QD, M, QD, QD, fl))) goto qte_join;
        from_act((const __fp16 *)yo, M, QD, y); for (size_t j = 0; j < (size_t)T * QD; j++) x[j] += y[j];
        /* the MLP */
        for (int t = 0; t < M; t++) { if (t < T) rms(x + (size_t)t * QD, t0 + (size_t)t * QD, QD, wn2); else memset(t0 + (size_t)t * QD, 0, QD * 4); }
        to_act(t0, M, QD, QD, QD, (__fp16 *)act);
        if ((err = kblock_lin(h, w_mlp, (int)(3 * wbytes((size_t)QD * QF, e->fmt) / 2), act, M * QD, scr, M * QF, yo, M * QD, M, QD, QF, fl | 1))) goto qte_join;
        from_act((const __fp16 *)yo, M, QD, y); for (size_t j = 0; j < (size_t)T * QD; j++) x[j] += y[j];
        /* the next layer's read is waited for on an error too, so that no read is left writing into a buffer */
qte_join:
        if (i + 1 < QL) { pthread_join(th, 0); if (nx.err) err = -2; }
        if (progress) progress(i + 1);
    }
    if (err) LOG("qte layer error %d", err);
    if (!err)
    {
        float nw[QD]; if (readf(e->dir, "qte_norm.f32", nw, QD * 4)) err = -1;
        else for (int t = 0; t < T; t++) rms(x + (size_t)t * QD, out + (size_t)t * QD, QD, nw);
    }
    free(prm);
done:
    free(x); free(t0); free(y); if (act) rpcmem_free(act); if (yo) rpcmem_free(yo); if (scr) rpcmem_free(scr);
    return err;
}

/* the weight buffers for a folder (format.txt as kconv's: q8 / q4 / none) */
static qte_t *qte_open(const char *dir)
{
    qte_t *e = calloc(1, sizeof(qte_t)); snprintf(e->dir, sizeof(e->dir), "%s", dir); e->fmt = fmt_of(dir);
    size_t mx = 0; for (int i = 0; i < QL; i++) { e->wl[i] = qte_wbytes(i, e->fmt); if (e->wl[i] > mx) mx = e->wl[i]; }
    e->wb[0] = ralloc(mx); e->wb[1] = ralloc(mx);
    if (!e->wb[0] || !e->wb[1]) { if (e->wb[0]) rpcmem_free(e->wb[0]); if (e->wb[1]) rpcmem_free(e->wb[1]); free(e); return 0; }
    return e;
}
static void qte_close(qte_t *e) { if (!e) return; rpcmem_free(e->wb[0]); rpcmem_free(e->wb[1]); free(e); }

/* JNI (Engine.Qte): a session of its own on the DSP and the two weight buffers; encode; close (before the image blocks open) */
typedef struct { remote_handle64 h; qte_t *e; volatile int done; } qte_jni_t;
static qte_jni_t *g_qte_cur;
static void qte_prog(int i) { if (g_qte_cur) g_qte_cur->done = i; }
JNIEXPORT jlong JNICALL Java_dev_kroma_open_Engine_nQteOpen(JNIEnv *env, jclass cl, jstring jarch, jstring jdir)
{
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    qte_jni_t *q = calloc(1, sizeof(qte_jni_t));
    char self[512] = ""; Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nQteOpen, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    char uri[256]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um));
    if (!err) err = kblock_open(uri, &q->h);
    if (!err && !(q->e = qte_open(dir))) err = -3;
    (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir);
    if (err) { LOG("qte open failed 0x%x", err); if (q->h) kblock_close(q->h); free(q); return 0; }
    return (jlong)(intptr_t)q;
}
/* ids [T] -> out [T x 4096] (the last hidden state); 0 or an error */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nQteEncode(JNIEnv *env, jclass cl, jlong jh, jintArray jids, jfloatArray jout)
{
    qte_jni_t *q = (qte_jni_t *)(intptr_t)jh; if (!q) return -1;
    const int T = (*env)->GetArrayLength(env, jids); if ((*env)->GetArrayLength(env, jout) < T * QD) return -4;
    int *ids = malloc((size_t)T * 4); (*env)->GetIntArrayRegion(env, jids, 0, T, ids);
    float *out = malloc((size_t)T * QD * 4); q->done = 0; g_qte_cur = q;
    const int err = qte_encode(q->h, q->e, ids, T, out, qte_prog); g_qte_cur = 0;
    if (!err) (*env)->SetFloatArrayRegion(env, jout, 0, T * QD, out);
    free(ids); free(out); return err;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nQteProgress(JNIEnv *env, jclass cl, jlong jh) { qte_jni_t *q = (qte_jni_t *)(intptr_t)jh; return q ? q->done : 0; }
JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nQteClose(JNIEnv *env, jclass cl, jlong jh)
{ qte_jni_t *q = (qte_jni_t *)(intptr_t)jh; if (!q) return; qte_close(q->e); kblock_close(q->h); free(q); }
