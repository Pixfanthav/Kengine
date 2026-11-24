
/* ------------------------------------------------------------------ LoRAs baked exactly, one after another (as ComfyUI patches them)
 * A set file lists the LoRA folders, one per line: "<dir>\t<strength>". Each folder (LoraImport) holds per Linear <prefix> any of
 *   <prefix>.A.f16 [r x in], .B.f16 [out x r]   a pair at its own rank, alpha / rank in B
 *   <prefix>.lokr.f32   int32 fo fi o2 i2, then w1 [fo x fi] (alpha / dim in it) and w2 [o2 x i2]: dW = kron(w1, w2)
 *   <prefix>.dora.f32   int32 axis (0: per output row, 1: per input column), int32 n, then the magnitude [n]
 * and the change is W += s dW, or with a magnitude ComfyUI's weight_decompose: Wc = (W + dW) m / |.|, W += s (Wc - W), the norm
 * of each row of the base W on the output axis, of each column of W + dW on the input axis. */
#include <unistd.h>
#include <fcntl.h>
typedef struct { float *a, *b; int R; float *w1, *w2; int fo, fi, o2, i2; float *mag; int maxis; float st; } patch_t;
/* a matrix: kind 0 the engine's fp16 HMX layout (rows head-permuted when heads, K stored as ksplit quarters), 1 float [K][N] */
typedef struct { void *w; int K, N, kind, heads, ksplit; } mat_t;
static inline size_t mat_at(const mat_t *m, int n, int k)
{
    if (m->kind) return (size_t)k * m->N + n;
    const int Kq = m->K / m->ksplit, q = k / Kq, kk = k % Kq;
    return (size_t)q * m->N * Kq + (size_t)((n >> 6) * 2 + ((n >> 5) & 1)) * (Kq / 32) * 1024 + (kk >> 5) * 1024 + 32 * ((kk & 31) & ~1) + 2 * (n & 31) + (kk & 1);
}
static inline float mat_get(const mat_t *m, size_t i) { return m->kind ? ((float *)m->w)[i] : (float)((__fp16 *)m->w)[i]; }
static inline void mat_set(const mat_t *m, size_t i, float v) { if (m->kind) ((float *)m->w)[i] = v; else ((__fp16 *)m->w)[i] = (__fp16)v; }
static inline int mat_src(const mat_t *m, int n) { if (!m->heads) return n; int h = n / 128, d = n % 128; return h * 128 + (d < 64 ? 2 * d : 2 * (d - 64) + 1); }

static void *readraw(const char *dir, const char *name, size_t *bytes)
{
    char p[768]; snprintf(p, sizeof(p), "%s/%s", dir, name); FILE *f = fopen(p, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); void *b = malloc(n > 0 ? (size_t)n : 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return 0; } fclose(f); *bytes = (size_t)n; return b;
}
static void patch_free(patch_t *p) { free(p->a); free(p->b); free(p->w1); free(p->w2); free(p->mag); memset(p, 0, sizeof(*p)); }
/* what one folder has for one Linear [N out x K in]; 0 when nothing */
static int patch_load(const char *dir, const char *prefix, int K, int N, float st, patch_t *p)
{
    memset(p, 0, sizeof(*p)); p->st = st; char n1[400], n2[400]; size_t ba = 0, bb = 0;
    snprintf(n1, sizeof(n1), "%s.A.f16", prefix); snprintf(n2, sizeof(n2), "%s.B.f16", prefix);
    __fp16 *a = readraw(dir, n1, &ba), *b = a ? readraw(dir, n2, &bb) : 0;
    if (a && b && ba > 0 && ba % ((size_t)K * 2) == 0 && bb == (size_t)N * (ba / ((size_t)K * 2)) * 2)
    {
        p->R = (int)(ba / ((size_t)K * 2)); p->a = malloc(ba * 2); p->b = malloc(bb * 2);
        for (size_t i = 0; i < ba / 2; i++) p->a[i] = (float)a[i];
        for (size_t i = 0; i < bb / 2; i++) p->b[i] = (float)b[i];
    }
    free(a); free(b);
    size_t bl = 0; snprintf(n1, sizeof(n1), "%s.lokr.f32", prefix); int32_t *l = readraw(dir, n1, &bl);
    if (l && bl >= 16)
    {
        const int fo = l[0], fi = l[1], o2 = l[2], i2 = l[3];
        if (fo * o2 == N && fi * i2 == K && bl == 16 + ((size_t)fo * fi + (size_t)o2 * i2) * 4)
        {
            p->fo = fo; p->fi = fi; p->o2 = o2; p->i2 = i2; p->w1 = malloc((size_t)fo * fi * 4); p->w2 = malloc((size_t)o2 * i2 * 4);
            memcpy(p->w1, l + 4, (size_t)fo * fi * 4); memcpy(p->w2, l + 4 + fo * fi, (size_t)o2 * i2 * 4);
        }
    }
    free(l);
    size_t bd = 0; snprintf(n1, sizeof(n1), "%s.dora.f32", prefix); int32_t *d = readraw(dir, n1, &bd);
    if (d && bd >= 8 && (p->R || p->w1))
    {
        const int ax = d[0], n = d[1];
        if (((ax == 0 && n == N) || (ax == 1 && n == K)) && bd == 8 + (size_t)n * 4) { p->maxis = ax; p->mag = malloc((size_t)n * 4); memcpy(p->mag, d + 2, (size_t)n * 4); }
    }
    free(d);
    return p->R || p->w1;
}
/* the change to (original) row src: row[K] */
static void patch_row(const patch_t *p, int src, int K, float *row)
{
    memset(row, 0, (size_t)K * 4);
    for (int r = 0; r < p->R; r++) { const float bv = p->b[(size_t)src * p->R + r]; if (bv == 0.f) continue; const float *ar = p->a + (size_t)r * K; for (int k = 0; k < K; k++) row[k] += bv * ar[k]; }
    if (p->w1)
    {
        const int ra = src / p->o2, rb = src % p->o2; const float *w2r = p->w2 + (size_t)rb * p->i2;
        for (int c = 0; c < p->fi; c++) { const float f = p->w1[(size_t)ra * p->fi + c]; if (f == 0.f) continue; float *o = row + (size_t)c * p->i2; for (int dd = 0; dd < p->i2; dd++) o[dd] += f * w2r[dd]; }
    }
}
typedef struct { const patch_t *p; const mat_t *m; int n0, n1, pass; double *col; const float *cn; } pjob_t;
static void *patch_part(void *v)
{
    pjob_t *j = (pjob_t *)v; const patch_t *p = j->p; const mat_t *m = j->m; const int K = m->K;
    float *row = malloc((size_t)K * 4), *wr = malloc((size_t)K * 4);
    for (int n = j->n0; n < j->n1; n++)
    {
        const int src = mat_src(m, n); patch_row(p, src, K, row);
        for (int k = 0; k < K; k++) wr[k] = mat_get(m, mat_at(m, n, k));
        if (j->pass == 0) { for (int k = 0; k < K; k++) { const double c = (double)wr[k] + row[k]; j->col[k] += c * c; } continue; }
        if (!p->mag) { for (int k = 0; k < K; k++) if (row[k] != 0.f) mat_set(m, mat_at(m, n, k), wr[k] + p->st * row[k]); continue; }
        if (p->maxis == 0)
        {
            double s = 0; for (int k = 0; k < K; k++) s += (double)wr[k] * wr[k];
            const float sc = p->mag[src] / (float)(sqrt(s) + 1.1920929e-7);
            for (int k = 0; k < K; k++) mat_set(m, mat_at(m, n, k), wr[k] + p->st * ((wr[k] + row[k]) * sc - wr[k]));
        }
        else for (int k = 0; k < K; k++) mat_set(m, mat_at(m, n, k), wr[k] + p->st * ((wr[k] + row[k]) * p->mag[k] / j->cn[k] - wr[k]));
    }
    free(row); free(wr); return 0;
}
#define BAKE_TH 6
static void patch_apply(const patch_t *p, const mat_t *m)
{
    pthread_t th[BAKE_TH]; pjob_t jb[BAKE_TH]; float *cn = 0; double *col = 0;
    if (p->mag && p->maxis == 1)
    {
        col = calloc((size_t)BAKE_TH * m->K, sizeof(double)); cn = malloc((size_t)m->K * 4);
        for (int t = 0; t < BAKE_TH; t++) { jb[t] = (pjob_t){ p, m, m->N * t / BAKE_TH, m->N * (t + 1) / BAKE_TH, 0, col + (size_t)t * m->K, 0 }; pthread_create(&th[t], 0, patch_part, &jb[t]); }
        for (int t = 0; t < BAKE_TH; t++) pthread_join(th[t], 0);
        for (int k = 0; k < m->K; k++) { double s = 0; for (int t = 0; t < BAKE_TH; t++) s += col[(size_t)t * m->K + k]; cn[k] = (float)(sqrt(s) + 1.1920929e-7); }
    }
    for (int t = 0; t < BAKE_TH; t++) { jb[t] = (pjob_t){ p, m, m->N * t / BAKE_TH, m->N * (t + 1) / BAKE_TH, 1, 0, cn }; pthread_create(&th[t], 0, patch_part, &jb[t]); }
    for (int t = 0; t < BAKE_TH; t++) pthread_join(th[t], 0);
    free(col); free(cn);
}
typedef struct { char dir[512]; float st; } setlib_t;
static int read_set(const char *set, setlib_t *libs, int max)
{
    FILE *f = fopen(set, "r"); if (!f) return -1; char line[700]; int n = 0;
    while (n < max && fgets(line, sizeof(line), f))
    {
        char *tab = strchr(line, '\t'); if (!tab) continue; *tab = 0;
        snprintf(libs[n].dir, sizeof(libs[n].dir), "%s", line); libs[n].st = (float)atof(tab + 1); n++;
    }
    fclose(f); return n;
}
/* all the set's LoRAs into one Linear; the number of folders that had something for it */
static int bake_mat(const setlib_t *libs, int nl, const char *prefix, const mat_t *m)
{
    int used = 0;
    for (int l = 0; l < nl; l++) { patch_t p; if (patch_load(libs[l].dir, prefix, m->K, m->N, libs[l].st, &p)) { patch_apply(&p, m); used++; } patch_free(&p); }
    return used;
}
static int set_has(const setlib_t *libs, int nl, const char *prefix)
{
    static const char *ext[2] = { "A.f16", "lokr.f32" };
    for (int l = 0; l < nl; l++) for (int e = 0; e < 2; e++) { char p[768]; snprintf(p, sizeof(p), "%s/%s.%s", libs[l].dir, prefix, ext[e]); if (access(p, R_OK) == 0) return 1; }
    return 0;
}
/* the blocks with the set baked in: krea 0 Anima (kconv's 10 matrices a block), 1 Krea 2 (8, mlp.down in quarters); a block no
 * LoRA touches is linked to the base file, not copied. Returns the Linears changed, or < 0. */
static int bake_set(const char *base, const char *set, const char *out, int nb, int krea)
{
    setlib_t libs[16]; const int nl = read_set(set, libs, 16); if (nl < 0) return -4;
    static const char *kmats[8] = { "attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down" };
    static const int K8[8] = { KD, KD, KD, KD, KD, KD, KD, KF }, N8[8] = { KD, KKV, KKV, KD, KD, KF, KF, KD }, H8[8] = { 48, 12, 0, 0, 0, 0, 0, 0 };
    const int nm = krea ? 8 : 10, nfile = krea ? 11 : 10; size_t msz[11], nW = 0;
    if (krea) { for (int m = 0; m < 7; m++) msz[m] = (size_t)K8[m] * N8[m]; for (int q = 0; q < 4; q++) msz[7 + q] = (size_t)KF / 4 * KD; }
    else for (int m = 0; m < 10; m++) msz[m] = (size_t)LIN[m] * LOUT[m];
    for (int m = 0; m < nfile; m++) nW += msz[m];
    const int fmt = fmt_of(base); __fp16 *w = malloc(nW * 2); int used = 0, err = w ? 0 : -3;
    if (fmt && !err) { char p[640]; snprintf(p, sizeof(p), "%s/format.txt", out); FILE *f = fopen(p, "w"); if (f) { fputs(fmt == 1 ? "q8\n" : "q4\n", f); fclose(f); } }
    g_bake_done = 0;
    for (int i = 0; i < nb && !err; i++)
    {
        char n[96], pre[160]; snprintf(n, sizeof(n), "weights_b%d.f16", i); int any = 0;
        for (int m = 0; m < nm && !any; m++) { if (krea) snprintf(pre, sizeof(pre), "blocks.%d.%s", i, kmats[m]); else snprintf(pre, sizeof(pre), "b%d_%s", i, LKEY[m]); any = set_has(libs, nl, pre); }
        if (!any)
        {
            char s[640], d[640]; snprintf(s, sizeof(s), "%s/%s", base, n); snprintf(d, sizeof(d), "%s/%s", out, n); unlink(d);
            if (link(s, d) == 0 || symlink(s, d) == 0) { g_bake_done = i + 1; continue; }
        }
        if (read_block16(base, n, w, msz, nfile, fmt)) { err = -1; break; }
        size_t off = 0;
        for (int m = 0; m < nm; m++)
        {
            mat_t mt;
            if (krea) { mt = (mat_t){ w + off, K8[m], N8[m], 0, H8[m], m == 7 ? 4 : 1 }; snprintf(pre, sizeof(pre), "blocks.%d.%s", i, kmats[m]); }
            else { mt = (mat_t){ w + off, LIN[m], LOUT[m], 0, 0, 1 }; snprintf(pre, sizeof(pre), "b%d_%s", i, LKEY[m]); }
            used += bake_mat(libs, nl, pre, &mt) > 0; off += (size_t)mt.K * mt.N;
        }
        if (write_block16(out, n, w, msz, nfile, fmt)) err = -2;
        g_bake_done = i + 1;
    }
    free(w);
    /* Anima's output layer (final.f16, W^T [D x 64] in the same layout) when a LoRA changes final_layer.linear */
    if (!err && !krea && set_has(libs, nl, "final"))
    {
        __fp16 *fw = malloc((size_t)D * 64 * 2);
        if (fw && !readf(base, "final.f16", fw, (size_t)D * 64 * 2))
        {
            mat_t mt = { fw, D, 64, 0, 0, 1 }; used += bake_mat(libs, nl, "final", &mt) > 0;
            char p[640]; snprintf(p, sizeof(p), "%s/final.f16", out); FILE *f = fopen(p, "wb");
            if (!f || fwrite(fw, 2, (size_t)D * 64, f) != (size_t)D * 64) err = -2; if (f) fclose(f);
        }
        free(fw);
    }
    return err ? err : used;
}
/* Krea 2's text path (txt2.onnx, float32 inside the file) with the set baked in: [out] is a copy of the model, rewritten in place
 * at each parameter's offset. keys[i]: a Linear's name (stored [in x out], rows = in, tr 1) or a vector's (an RMSNorm scale stored
 * as 1 + scale: rows = n, tr 0). Returns the parameters changed, or < 0. */
static int txt_bake(const char *out, const char *set, int n, const char *const *keys, const long long *offs, const int *rows, const int *cols, const int *tr)
{
    setlib_t libs[16]; const int nl = read_set(set, libs, 16); if (nl < 0) return -4;
    const int fd = open(out, O_RDWR); if (fd < 0) return -1; int used = 0, err = 0;
    g_bake_done = 0;
    for (int i = 0; i < n && !err; i++)
    {
        const int K = tr[i] ? rows[i] : 1, N = tr[i] ? cols[i] : rows[i];
        if (set_has(libs, nl, keys[i]))
        {
            const size_t bytes = (size_t)K * N * 4; float *w = malloc(bytes);
            if (!w || pread(fd, w, bytes, (off_t)offs[i]) != (ssize_t)bytes) { free(w); err = -2; break; }
            mat_t mt = { w, K, N, 1, 0, 1 };
            if (bake_mat(libs, nl, keys[i], &mt) > 0) { if (pwrite(fd, w, bytes, (off_t)offs[i]) != (ssize_t)bytes) err = -2; else used++; }
            free(w);
        }
        g_bake_done = i + 1;
    }
    close(fd);
    return err ? err : used;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nBakeSet(JNIEnv *env, jclass cl, jstring jbase, jstring jset, jstring jout, jint nb, jint krea)
{
    const char *base = (*env)->GetStringUTFChars(env, jbase, 0), *set = (*env)->GetStringUTFChars(env, jset, 0), *out = (*env)->GetStringUTFChars(env, jout, 0);
    int r = bake_set(base, set, out, nb, krea);
    (*env)->ReleaseStringUTFChars(env, jbase, base); (*env)->ReleaseStringUTFChars(env, jset, set); (*env)->ReleaseStringUTFChars(env, jout, out);
    return r;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nTxtBake(JNIEnv *env, jclass cl, jstring jout, jstring jset, jobjectArray jkeys, jlongArray joffs, jintArray jrows, jintArray jcols, jintArray jtr)
{
    const int n = (*env)->GetArrayLength(env, jkeys);
    const char *out = (*env)->GetStringUTFChars(env, jout, 0), *set = (*env)->GetStringUTFChars(env, jset, 0);
    const char **keys = malloc(sizeof(char *) * n); jstring *js = malloc(sizeof(jstring) * n);
    for (int i = 0; i < n; i++) { js[i] = (jstring)(*env)->GetObjectArrayElement(env, jkeys, i); keys[i] = (*env)->GetStringUTFChars(env, js[i], 0); }
    jlong *offs = (*env)->GetLongArrayElements(env, joffs, 0); jint *rows = (*env)->GetIntArrayElements(env, jrows, 0), *cols = (*env)->GetIntArrayElements(env, jcols, 0), *tr = (*env)->GetIntArrayElements(env, jtr, 0);
    long long *o = malloc(sizeof(long long) * n); for (int i = 0; i < n; i++) o[i] = offs[i];
    int r = txt_bake(out, set, n, keys, o, (const int *)rows, (const int *)cols, (const int *)tr);
    free(o);
    (*env)->ReleaseLongArrayElements(env, joffs, offs, 0); (*env)->ReleaseIntArrayElements(env, jrows, rows, 0); (*env)->ReleaseIntArrayElements(env, jcols, cols, 0); (*env)->ReleaseIntArrayElements(env, jtr, tr, 0);
    for (int i = 0; i < n; i++) { (*env)->ReleaseStringUTFChars(env, js[i], keys[i]); (*env)->DeleteLocalRef(env, js[i]); }
    free(keys); free(js);
    (*env)->ReleaseStringUTFChars(env, jout, out); (*env)->ReleaseStringUTFChars(env, jset, set);
    return r;
}

/* ------------------------------------------------------------------ Krea 2's timestep path with the set in it
 * tpath.bf16 (the app's export of the timestep path): tmlp.0 [6144 x 256] bf16, its bias float32 [6144], tmlp.2 [6144 x 6144], bias, tproj.1 [36864 x 6144],
 * bias. For n sigmas: t = tmlp(emb(1000 sigma)) [n x 6144] (what the last layer takes) and tvec = tproj(t) [n x 36864] (the blocks'
 * modulation), each Linear's rows with the set's changes applied in order as bake_set does (a DoRA magnitude on the input axis
 * is not taken here). */
#include <sys/mman.h>
#include <sys/stat.h>
static inline float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline float gelu_t(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
typedef struct { const uint16_t *w; const float *b; int K, N, nx; const patch_t *p; int np; const float *x; float *y; int n0, n1; } tlin_t;
static void *tlin_part(void *v)
{
    tlin_t *j = (tlin_t *)v; const int K = j->K; float *wr = malloc((size_t)K * 4), *d = malloc((size_t)K * 4);
    for (int n = j->n0; n < j->n1; n++)
    {
        const uint16_t *src = j->w + (size_t)n * K; for (int k = 0; k < K; k++) wr[k] = bf2f(src[k]);
        for (int q = 0; q < j->np; q++)
        {
            const patch_t *p = &j->p[q]; patch_row(p, n, K, d);
            if (p->mag && p->maxis == 0)
            {
                double s = 0; for (int k = 0; k < K; k++) s += (double)wr[k] * wr[k];
                const float sc = p->mag[n] / (float)(sqrt(s) + 1.1920929e-7);
                for (int k = 0; k < K; k++) wr[k] += p->st * ((wr[k] + d[k]) * sc - wr[k]);
            }
            else for (int k = 0; k < K; k++) wr[k] += p->st * d[k];
        }
        for (int s = 0; s < j->nx; s++) { const float *x = j->x + (size_t)s * K; float a = 0; for (int k = 0; k < K; k++) a += wr[k] * x[k]; j->y[(size_t)s * j->N + n] = a + j->b[n]; }
    }
    free(wr); free(d); return 0;
}
/* one Linear of the path for nx inputs; the number of LoRA folders that changed it */
static int tlin(const uint16_t *w, const float *b, int K, int N, const setlib_t *libs, int nl, const char *prefix, const float *x, float *y, int nx)
{
    patch_t ps[16]; int np = 0;
    for (int l = 0; l < nl && np < 16; l++) if (patch_load(libs[l].dir, prefix, K, N, libs[l].st, &ps[np]))
    { if (ps[np].mag && ps[np].maxis == 1) { LOG("time path: %s: a DoRA magnitude on the input axis is left out", prefix); free(ps[np].mag); ps[np].mag = 0; } np++; } else patch_free(&ps[np]);
    pthread_t th[BAKE_TH]; tlin_t jb[BAKE_TH];
    for (int t = 0; t < BAKE_TH; t++) { jb[t] = (tlin_t){ w, b, K, N, nx, ps, np, x, y, N * t / BAKE_TH, N * (t + 1) / BAKE_TH }; pthread_create(&th[t], 0, tlin_part, &jb[t]); }
    for (int t = 0; t < BAKE_TH; t++) pthread_join(th[t], 0);
    for (int q = 0; q < np; q++) patch_free(&ps[q]);
    return np;
}
static int time_bake(const char *tpath, const char *set, const float *sig, int n, float *tout, float *tvout)
{
    enum { TD = 256, FD = 6144, PD = 36864 };
    setlib_t libs[16]; const int nl = read_set(set, libs, 16); if (nl < 0) return -4;
    const size_t need = (size_t)FD * TD * 2 + FD * 4 + (size_t)FD * FD * 2 + FD * 4 + (size_t)PD * FD * 2 + PD * 4;
    const int fd = open(tpath, O_RDONLY); if (fd < 0) return -1;
    struct stat stt; if (fstat(fd, &stt) || (size_t)stt.st_size != need) { close(fd); return -2; }
    const uint8_t *m = mmap(0, need, PROT_READ, MAP_SHARED, fd, 0); close(fd); if (m == MAP_FAILED) return -3;
    const uint16_t *w0 = (const uint16_t *)m; const float *b0 = (const float *)(m + (size_t)FD * TD * 2);
    const uint16_t *w2 = (const uint16_t *)((const uint8_t *)b0 + FD * 4); const float *b2 = (const float *)((const uint8_t *)w2 + (size_t)FD * FD * 2);
    const uint16_t *wp = (const uint16_t *)((const uint8_t *)b2 + FD * 4); const float *bp = (const float *)((const uint8_t *)wp + (size_t)PD * FD * 2);
    float *e = malloc((size_t)n * TD * 4), *h = malloc((size_t)n * FD * 4), *g = malloc((size_t)n * FD * 4);
    for (int s = 0; s < n; s++) for (int i = 0; i < TD / 2; i++)
    {
        const float fr = expf(-logf(10000.f) * (float)i / (TD / 2)), a = 1000.f * sig[s] * fr;
        e[(size_t)s * TD + i] = cosf(a); e[(size_t)s * TD + TD / 2 + i] = sinf(a);
    }
    int used = 0;
    used += tlin(w0, b0, TD, FD, libs, nl, "tmlp.0", e, h, n) > 0;
    for (size_t i = 0; i < (size_t)n * FD; i++) h[i] = gelu_t(h[i]);
    used += tlin(w2, b2, FD, FD, libs, nl, "tmlp.2", h, tout, n) > 0;
    for (size_t i = 0; i < (size_t)n * FD; i++) g[i] = gelu_t(tout[i]);
    used += tlin(wp, bp, FD, PD, libs, nl, "tproj.1", g, tvout, n) > 0;
    free(e); free(h); free(g); munmap((void *)m, need);
    return used;
}
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nTimeBake(JNIEnv *env, jclass cl, jstring jtp, jstring jset, jfloatArray jsig, jfloatArray jt, jfloatArray jtv)
{
    const char *tp = (*env)->GetStringUTFChars(env, jtp, 0), *set = (*env)->GetStringUTFChars(env, jset, 0);
    const int n = (*env)->GetArrayLength(env, jsig);
    jfloat *sig = (*env)->GetFloatArrayElements(env, jsig, 0), *t = (*env)->GetFloatArrayElements(env, jt, 0), *tv = (*env)->GetFloatArrayElements(env, jtv, 0);
    int r = time_bake(tp, set, sig, n, t, tv);
    (*env)->ReleaseFloatArrayElements(env, jsig, sig, JNI_ABORT); (*env)->ReleaseFloatArrayElements(env, jt, t, 0); (*env)->ReleaseFloatArrayElements(env, jtv, tv, 0);
    (*env)->ReleaseStringUTFChars(env, jtp, tp); (*env)->ReleaseStringUTFChars(env, jset, set);
    return r;
}
