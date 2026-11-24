/* kte.inc (included by kengine.c after qte.inc) -- Krea 2's text encoder, Qwen3-VL-4B's text layers 0..34, on the DSP the way qte.inc runs
 * MOSS's: the products on the DSP (kblock_lin), the norms, RoPE and the causal attention here in fp32. Krea conditions on the inputs of
 * layers 2, 5, ..., 35 (no final norm): those twelve [T x 2560] are the output, what the app's txt2.onnx fuses.
 * Files (made by Kroma's converter tools, not in this repository): kte_l<i>.w (in = q | k | v, out, MLP), kte_l<i>.f32 (input / post norms [2560], q / k norms [128]), format.txt;
 * the embedding table is the app's krea/embed.f16 (fp16 rows of 2560). */
enum { KTDM = 4096, KTHD = 128, KTNM = 12 };
/* <dir>/kte.txt: D F layers q-heads kv-heads theta outputs t0 t1 ... (without it Krea's). An output t is the input of layer t; t = layers
 * is after the last layer, with kte_norm.f32 (the model's final norm) on it when that file is there */
typedef struct { int fmt; char dir[512]; size_t wl; uint16_t *wb[2]; int D, F, L, H, KV, nt, taps[KTNM]; double theta; } kte_t;
static size_t kte_wbytes(const kte_t *e) { const size_t in = (size_t)(e->H + 2 * e->KV) * KTHD; return wbytes((size_t)e->D * in, e->fmt) + wbytes((size_t)e->H * KTHD * e->D, e->fmt) + 3 * wbytes((size_t)e->D * e->F, e->fmt); }
static void *kte_read(void *a) { qrd_t *r = (qrd_t *)a; char n[64]; snprintf(n, sizeof(n), "kte_l%d.w", r->i); r->err = readf(r->dir, n, r->buf, r->bytes); return 0; }

/* ids [T] -> outputs [nt][T][D]; emb: the fp16 embedding table */
static int kte_encode(remote_handle64 h, kte_t *e, const char *emb, const int *ids, int T, float *taps, void (*progress)(int))
{
    const int KTD = e->D, KTF = e->F, KTL = e->L, KTH = e->H, KTKV = e->KV, KTIN = (KTH + 2 * KTKV) * KTHD, KTNT = e->nt;
    const int M = (T + 31) / 32 * 32; int err = 0, nt = 0;
    float *x = calloc((size_t)M * KTD, 4), *t0 = malloc((size_t)M * KTH * KTHD * 4), *y = malloc((size_t)M * KTIN * 4);
    uint16_t *act = ralloc((size_t)M * KTH * KTHD * 2), *yo = ralloc((size_t)M * KTIN * 2), *scr = ralloc((size_t)M * KTF * 2);
    uint16_t *yg = ralloc((size_t)M * KTF * 2), *yu = ralloc((size_t)M * KTF * 2); float *fg = malloc((size_t)M * KTF * 4), *fu = malloc((size_t)M * KTF * 4), *rs = malloc((size_t)M * 4);
    float prm[2 * KTDM + 2 * KTHD];
    if (!x || !t0 || !y || !act || !yo || !scr || !yg || !yu || !fg || !fu || !rs) { err = -3; goto done; }
    {   FILE *f = fopen(emb, "rb"); if (!f) { err = -1; goto done; }
        __fp16 row[KTDM];
        for (int t = 0; t < T; t++)
        {
            if (ids[t] < 0 || fseeko(f, (off_t)ids[t] * KTD * 2, SEEK_SET) || fread(row, 2, KTD, f) != (size_t)KTD) { fclose(f); err = -1; goto done; }
            for (int c = 0; c < KTD; c++) x[(size_t)t * KTD + c] = (float)row[c];
        }
        fclose(f);
    }
    qrd_t r0 = { e->dir, 0, e->wb[0], e->wl, 0 }; kte_read(&r0); if (r0.err) { err = -2; goto done; }
    const int fl = e->fmt << 8;
    float inv[KTHD / 2]; for (int j = 0; j < KTHD / 2; j++) inv[j] = 1.f / (float)pow(e->theta, (double)(2 * j) / KTHD);
    for (int i = 0; i < KTL && !err; i++)
    {
        if (g_abort) { err = -99; break; }   /* Engine.abort: a stop leaves between layers */
        if (nt < KTNT && e->taps[nt] == i) memcpy(taps + (size_t)nt++ * T * KTD, x, (size_t)T * KTD * 4);      /* the input of layer i */
        uint16_t *w = e->wb[i & 1]; pthread_t th; qrd_t nx = { e->dir, i + 1, e->wb[(i + 1) & 1], i + 1 < KTL ? e->wl : 0, 0 };
        if (i + 1 < KTL) pthread_create(&th, 0, kte_read, &nx);
        { char n[64]; snprintf(n, sizeof(n), "kte_l%d.f32", i); if (readf(e->dir, n, prm, (size_t)(2 * KTD + 2 * KTHD) * 4)) { err = -1; goto kte_join; } }
        const float *wn1 = prm, *wn2 = prm + KTD, *qn = prm + 2 * KTD, *kn = qn + KTHD;
        const uint16_t *w_in = w, *w_out = WPH(w_in, (size_t)KTD * KTIN, e->fmt), *w_mlp = WPH(w_out, (size_t)KTH * KTHD * KTD, e->fmt);
        for (int t = 0; t < M; t++) { if (t < T) rms(x + (size_t)t * KTD, t0 + (size_t)t * KTD, KTD, wn1); else memset(t0 + (size_t)t * KTD, 0, KTD * 4); }
        to_act(t0, M, KTD, KTD, KTD, (__fp16 *)act);
        if ((err = kblock_lin(h, w_in, (int)(wbytes((size_t)KTD * KTIN, e->fmt) / 2), act, M * KTD, scr, M * KTF, yo, M * KTIN, M, KTD, KTIN, fl))) goto kte_join;
        from_act((const __fp16 *)yo, M, KTIN, y);

        float *mix = t0;                                                                 /* [T x 4096] into the out product */
        for (int t = 0; t < T; t++)
            for (int hh = 0; hh < KTH + KTKV; hh++)
            {   /* q / k RMSNorm per head (x w), RoPE over all 128 dims (split half), text positions 0..T-1 */
                float *a = y + (size_t)t * KTIN + hh * KTHD; rms(a, a, KTHD, hh < KTH ? qn : kn);
                for (int j = 0; j < KTHD / 2; j++)
                {
                    const float ang = (float)t * inv[j], c = cosf(ang), s = sinf(ang), x1 = a[j], x2 = a[j + KTHD / 2];
                    a[j] = x1 * c - x2 * s; a[j + KTHD / 2] = x2 * c + x1 * s;
                }
            }
        const float sc = 1.f / sqrtf((float)KTHD); float *p = malloc((size_t)T * 4);
        for (int hh = 0; hh < KTH; hh++)
            for (int t = 0; t < T; t++)
            {   /* causal attention, 4 query heads to a kv head */
                const float *qh = y + (size_t)t * KTIN + hh * KTHD; const int hk = hh / (KTH / KTKV); float mx = -1e30f, z = 0.f;
                for (int s = 0; s <= t; s++) { const float *kh = y + (size_t)s * KTIN + (KTH + hk) * KTHD; float d = 0.f; for (int j = 0; j < KTHD; j++) d += qh[j] * kh[j]; p[s] = d * sc; if (p[s] > mx) mx = p[s]; }
                for (int s = 0; s <= t; s++) { p[s] = expf(p[s] - mx); z += p[s]; }
                float *m = mix + (size_t)t * KTH * KTHD + hh * KTHD;
                for (int j = 0; j < KTHD; j++) { float a = 0.f; for (int s = 0; s <= t; s++) a += p[s] * y[(size_t)s * KTIN + (KTH + KTKV + hk) * KTHD + j]; m[j] = a / z; }
            }
        free(p);
        for (int t = T; t < M; t++) memset(mix + (size_t)t * KTH * KTHD, 0, KTH * KTHD * 4);

        to_act(mix, M, KTH * KTHD, KTH * KTHD, KTH * KTHD, (__fp16 *)act);
        if ((err = kblock_lin(h, w_out, (int)(wbytes((size_t)KTH * KTHD * KTD, e->fmt) / 2), act, M * KTH * KTHD, scr, M * KTF, yo, M * KTD, M, KTH * KTHD, KTD, fl))) goto kte_join;
        from_act((const __fp16 *)yo, M, KTD, y); for (size_t j = 0; j < (size_t)T * KTD; j++) x[j] += y[j];
        for (int t = 0; t < M; t++) { if (t < T) rms(x + (size_t)t * KTD, t0 + (size_t)t * KTD, KTD, wn2); else memset(t0 + (size_t)t * KTD, 0, KTD * 4); }
        to_act(t0, M, KTD, KTD, KTD, (__fp16 *)act);
        /* The MLP in three products with the hidden made here, each row scaled to a largest value of 1 and back after down: the HMX sums a
         * 32-row tile in one range, and a row 70 times the size of its neighbours left them about 1 of absolute error (up to 80 %) */
        const uint16_t *w_up = WPH(w_mlp, (size_t)KTD * KTF, e->fmt), *w_dn = WPH(w_up, (size_t)KTD * KTF, e->fmt); const int wmB = (int)(wbytes((size_t)KTD * KTF, e->fmt) / 2);
        if ((err = kblock_lin(h, w_mlp, wmB, act, M * KTD, scr, M * KTF, yg, M * KTF, M, KTD, KTF, fl))) goto kte_join;
        if ((err = kblock_lin(h, w_up, wmB, act, M * KTD, scr, M * KTF, yu, M * KTF, M, KTD, KTF, fl))) goto kte_join;
        from_act((const __fp16 *)yg, M, KTF, fg); from_act((const __fp16 *)yu, M, KTF, fu);
        for (int t = 0; t < M; t++)
        {
            float *g = fg + (size_t)t * KTF; const float *u = fu + (size_t)t * KTF; float mx = 0.f;
            if (t >= T) { memset(g, 0, KTF * 4); rs[t] = 1.f; continue; }
            for (int c = 0; c < KTF; c++) { const float v = g[c] / (1.f + expf(-g[c])) * u[c]; g[c] = v; const float av = fabsf(v); if (av > mx) mx = av; }
            rs[t] = mx > 0.f ? mx : 1.f; const float inv1 = 1.f / rs[t]; for (int c = 0; c < KTF; c++) g[c] *= inv1;
        }
        to_act(fg, M, KTF, KTF, KTF, (__fp16 *)yg);
        if ((err = kblock_lin(h, w_dn, wmB, yg, M * KTF, scr, M * KTF, yo, M * KTD, M, KTF, KTD, fl))) goto kte_join;
        from_act((const __fp16 *)yo, M, KTD, y); for (int t = 0; t < T; t++) for (int c = 0; c < KTD; c++) y[(size_t)t * KTD + c] *= rs[t]; for (size_t j = 0; j < (size_t)T * KTD; j++) x[j] += y[j];
        /* the next layer's read is waited for on an error too, so that no read is left writing into a buffer */
kte_join:
        if (i + 1 < KTL) { pthread_join(th, 0); if (nx.err) err = -2; }
        if (progress) progress(i + 1);
    }
    if (!err && nt == KTNT - 1)
    {   /* after the last layer (Krea's tap 35), through the final norm when the output has one */
        char np[600]; snprintf(np, sizeof(np), "%s/kte_norm.f32", e->dir); FILE *nf = fopen(np, "rb"); float w[KTDM];
        if (nf && fread(w, 4, KTD, nf) == (size_t)KTD) for (int t = 0; t < T; t++) rms(x + (size_t)t * KTD, x + (size_t)t * KTD, KTD, w);
        if (nf) fclose(nf);
        memcpy(taps + (size_t)nt++ * T * KTD, x, (size_t)T * KTD * 4);
    }
    if (err) LOG("kte layer error %d", err);
done:
    free(fg); free(fu); free(rs); if (yg) rpcmem_free(yg); if (yu) rpcmem_free(yu);
    free(x); free(t0); free(y); if (act) rpcmem_free(act); if (yo) rpcmem_free(yo); if (scr) rpcmem_free(scr);
    return err;
}
static kte_t *kte_open(const char *dir)
{
    kte_t *e = calloc(1, sizeof(kte_t)); snprintf(e->dir, sizeof(e->dir), "%s", dir); e->fmt = fmt_of(dir);
    e->D = 2560; e->F = 9728; e->L = 35; e->H = 32; e->KV = 8; e->theta = 5000000.; e->nt = 12; for (int i = 0; i < 12; i++) e->taps[i] = 2 + 3 * i;
    {   char cp[600]; snprintf(cp, sizeof(cp), "%s/kte.txt", dir); FILE *cf = fopen(cp, "r");
        if (cf) { if (fscanf(cf, "%d %d %d %d %d %lf %d", &e->D, &e->F, &e->L, &e->H, &e->KV, &e->theta, &e->nt) == 7 && e->nt <= KTNM) for (int i = 0; i < e->nt; i++) if (fscanf(cf, "%d", &e->taps[i]) != 1) e->nt = 0; fclose(cf); } }
    e->wl = kte_wbytes(e);
    e->wb[0] = ralloc(e->wl); e->wb[1] = ralloc(e->wl);
    if (!e->wb[0] || !e->wb[1]) { if (e->wb[0]) rpcmem_free(e->wb[0]); if (e->wb[1]) rpcmem_free(e->wb[1]); free(e); return 0; }
    return e;
}
static void kte_close(kte_t *e) { if (!e) return; rpcmem_free(e->wb[0]); rpcmem_free(e->wb[1]); free(e); }

/* JNI (Engine.Kte): open (its own DSP session), encode, close */
typedef struct { remote_handle64 h; kte_t *e; } kte_jni_t;
JNIEXPORT jlong JNICALL Java_dev_kroma_open_Engine_nKteOpen(JNIEnv *env, jclass cl, jstring jarch, jstring jdir)
{
    const char *arch = (*env)->GetStringUTFChars(env, jarch, 0), *dir = (*env)->GetStringUTFChars(env, jdir, 0);
    kte_jni_t *q = calloc(1, sizeof(kte_jni_t));
    char self[512] = ""; Dl_info di; if (dladdr((void *)Java_dev_kroma_open_Engine_nKteOpen, &di) && di.dli_fname) { snprintf(self, sizeof(self), "%s", di.dli_fname); snprintf(self, sizeof(self), "%s", dirname(self)); }
    char path[1024]; snprintf(path, sizeof(path), "%s;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", self); setenv("ADSP_LIBRARY_PATH", path, 1);
    char uri[256]; snprintf(uri, sizeof(uri), "file:///libkengine_skel_%s.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0%s", arch, CDSP_DOMAIN);
    struct remote_rpc_control_unsigned_module um = { CDSP_DOMAIN_ID, 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um, sizeof(um));
    if (!err) err = kblock_open(uri, &q->h);
    if (!err && !(q->e = kte_open(dir))) err = -3;
    (*env)->ReleaseStringUTFChars(env, jarch, arch); (*env)->ReleaseStringUTFChars(env, jdir, dir);
    if (err) { LOG("kte open failed 0x%x", err); if (q->h) kblock_close(q->h); free(q); return 0; }
    return (jlong)(intptr_t)q;
}
/* ids [T] -> taps [12 x T x 2560]; 0 or an error */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nKteEncode(JNIEnv *env, jclass cl, jlong jh, jstring jemb, jintArray jids, jfloatArray jout)
{
    kte_jni_t *q = (kte_jni_t *)(intptr_t)jh; if (!q) return -1;
    const int T = (*env)->GetArrayLength(env, jids); if ((*env)->GetArrayLength(env, jout) < q->e->nt * T * q->e->D) return -4;
    int *ids = malloc((size_t)T * 4); (*env)->GetIntArrayRegion(env, jids, 0, T, ids);
    const char *emb = (*env)->GetStringUTFChars(env, jemb, 0);
    float *out = malloc((size_t)q->e->nt * T * q->e->D * 4);
    const int err = kte_encode(q->h, q->e, emb, ids, T, out, 0);
    (*env)->ReleaseStringUTFChars(env, jemb, emb);
    if (!err) (*env)->SetFloatArrayRegion(env, jout, 0, q->e->nt * T * q->e->D, out);
    free(ids); free(out); return err;
}
JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nKteClose(JNIEnv *env, jclass cl, jlong jh)
{ kte_jni_t *q = (kte_jni_t *)(intptr_t)jh; if (!q) return; kte_close(q->e); kblock_close(q->h); free(q); }
