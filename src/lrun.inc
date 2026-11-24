/* ------------------------------------------------------------------ Anima: LoRAs applied as they run (kblock_run's flag 0x1000)
 * Instead of a baked copy of the blocks: per block, per Linear, A^T [K x lr] and B^T [lr x N] in the weight layout, the set's LoRAs
 * side by side in rank with each one's strength folded into its B (so a negative strength works too), zero past their total rank.
 * Kept in memory (28 blocks x 6.5MB per 64 slots) and put after the block's weights just before it runs. Returns the slots used, 0 for a
 * set with nothing for the blocks, or < 0: -6 a DoRA on the input axis, -8 more ranks than the DSP takes in a Linear (LR_MAX). */
#define MAXN_L 32768   /* the longest output of a Linear in the LoRAs' blocks */
/* m [r x c] = Q [r x k] R [k x c], k its numerical rank: Gram-Schmidt with column pivoting, twice over for orthogonality. Exact to single precision: the pair the DSP multiplies by is this, nothing is dropped */
static int qr_rank(const float *m, int r, int c, float **qo, float **ro)
{
    const int kmax = r < c ? r : c; double *a = malloc((size_t)r * c * 8), *q = malloc((size_t)r * kmax * 8), *rr = calloc((size_t)kmax * c, 8), *nr = malloc((size_t)c * 8); int *piv = malloc((size_t)c * sizeof(int)); int k = 0; double top = 0;
    for (size_t i = 0; i < (size_t)r * c; i++) a[i] = m[i];
    for (int j = 0; j < c; j++) { piv[j] = j; double s = 0; for (int i = 0; i < r; i++) s += a[(size_t)i * c + j] * a[(size_t)i * c + j]; nr[j] = s; if (s > top) top = s; }
    for (; k < kmax; k++)
    {
        int b = -1; double bv = top * 1e-14; for (int j = k; j < c; j++) if (nr[piv[j]] > bv) { bv = nr[piv[j]]; b = j; }
        if (b < 0) break;
        { int t = piv[k]; piv[k] = piv[b]; piv[b] = t; } const int pc = piv[k]; double *qk = q + (size_t)k * r;
        for (int i = 0; i < r; i++) qk[i] = a[(size_t)i * c + pc];
        for (int pass = 0; pass < 2; pass++) for (int l = 0; l < k; l++) { const double *ql = q + (size_t)l * r; double d = 0; for (int i = 0; i < r; i++) d += ql[i] * qk[i]; for (int i = 0; i < r; i++) qk[i] -= d * ql[i]; }
        double s = 0; for (int i = 0; i < r; i++) s += qk[i] * qk[i]; s = sqrt(s); if (s <= 0) break; for (int i = 0; i < r; i++) qk[i] /= s;
        for (int j = 0; j < c; j++) { double d = 0; for (int i = 0; i < r; i++) d += qk[i] * a[(size_t)i * c + j]; rr[(size_t)k * c + j] = d; for (int i = 0; i < r; i++) a[(size_t)i * c + j] -= d * qk[i]; double t = 0; for (int i = 0; i < r; i++) t += a[(size_t)i * c + j] * a[(size_t)i * c + j]; nr[j] = t; }
    }
    float *Q = malloc((size_t)r * (k ? k : 1) * 4), *R = malloc((size_t)(k ? k : 1) * c * 4);
    for (int i = 0; i < r; i++) for (int l = 0; l < k; l++) Q[(size_t)i * k + l] = (float)q[(size_t)l * r + i];
    for (size_t i = 0; i < (size_t)k * c; i++) R[i] = (float)rr[i];
    free(a); free(q); free(rr); free(nr); free(piv); *qo = Q; *ro = R; return k;
}
/* nLora's QR of each folder's LoKr for the Linear at hand, made in its rank pass and taken again by its blob pass. Per thread: t_lk the Linear's 16 folders, t_lki the folder lrun_load is reading; null elsewhere. */
typedef struct { float *q1, *r1, *q2, *r2; int k1, k2; } lk_t;
static __thread lk_t *t_lk; static __thread int t_lki;
/* a folder's patch for the run: a LoKr's kron(w1, w2) = kron(Q1, Q2) kron(R1, R2) joined to its pair, if any, as one exact pair at the product of the factors' ranks */
static int lrun_load(const char *dir, const char *pre, int K, int N, float st, patch_t *p)
{
    if (!patch_load(dir, pre, K, N, st, p)) return 0;
    if (!p->w1) return 1;
    lk_t *c = t_lk ? &t_lk[t_lki] : 0; float *q1, *r1, *q2, *r2; int k1, k2;
    if (c && c->q1) { q1 = c->q1; r1 = c->r1; q2 = c->q2; r2 = c->r2; k1 = c->k1; k2 = c->k2; }
    else { k1 = qr_rank(p->w1, p->fo, p->fi, &q1, &r1); k2 = qr_rank(p->w2, p->o2, p->i2, &q2, &r2); if (c) *c = (lk_t){ q1, r1, q2, r2, k1, k2 }; }
    const int R0 = p->R, Rk = k1 * k2, R = R0 + Rk;
    float *a = malloc((size_t)R * K * 4), *b = malloc((size_t)N * R * 4);
    if (a && b)
    {
        if (R0) { memcpy(a, p->a, (size_t)R0 * K * 4); for (int n = 0; n < N; n++) memcpy(b + (size_t)n * R, p->b + (size_t)n * R0, (size_t)R0 * 4); }
        for (int u = 0; u < k1; u++) for (int v = 0; v < k2; v++)
        {
            const int r = R0 + u * k2 + v; float *ar = a + (size_t)r * K;
            for (int c = 0; c < p->fi; c++) for (int d = 0; d < p->i2; d++) ar[c * p->i2 + d] = r1[(size_t)u * p->fi + c] * r2[(size_t)v * p->i2 + d];
            for (int ra = 0; ra < p->fo; ra++) for (int rb = 0; rb < p->o2; rb++) b[(size_t)(ra * p->o2 + rb) * R + r] = q1[(size_t)ra * k1 + u] * q2[(size_t)rb * k2 + v];
        }
        free(p->a); free(p->b); p->a = a; p->b = b; p->R = R; a = b = 0;
    }
    free(a); free(b); if (!c) { free(q1); free(r1); free(q2); free(r2); } free(p->w1); free(p->w2); p->w1 = p->w2 = 0;
    return p->R > 0;
}
/* A DoRA as it runs. Baked, one LoRA after another (patch_part): cur' = cur + s ((cur + K) sc - cur), sc = mag / |cur row|, on the output axis. So cur stays g (W + sum of pairs):
 * g' = g (1 - s + s sc), and the new pair's B row takes s sc / g'. The norms need the base weights; a plain pair just takes s / g. bs[l * N + src]: what folder l's B row src is multiplied by, s apart.
 * -6: a magnitude on the input axis (not yet). */
typedef struct { const mat_t *w; const patch_t *ps; const float *pbs; int np, n0, n1; double *nr; } lg_job_t;
static void *lg_part(void *v)
{
    lg_job_t *j = (lg_job_t *)v; const int K = j->w->K, N = j->w->N; float *row = malloc((size_t)K * 4);
    for (int n = j->n0; n < j->n1; n++)
    {
        const int src = mat_src(j->w, n); for (int k = 0; k < K; k++) row[k] = mat_get(j->w, mat_at(j->w, n, k));
        for (int l = 0; l < j->np; l++)
        {
            const patch_t *p = &j->ps[l]; const float c = p->st * j->pbs[(size_t)l * N + src];
            for (int r = 0; r < p->R; r++) { const float bv = p->b[(size_t)src * p->R + r] * c; if (bv == 0.f) continue; const float *a = p->a + (size_t)r * K; for (int k = 0; k < K; k++) row[k] += bv * a[k]; }
        }
        double s = 0; for (int k = 0; k < K; k++) s += (double)row[k] * row[k]; j->nr[src] = sqrt(s);
    }
    free(row); return 0;
}
static int lrun_gains(const setlib_t *libs, int nl, const char *pre, const mat_t *w, float *g, float *bs)
{
    const int K = w->K, N = w->N; int err = 0, np = 0; patch_t ps[16]; float *pbs = malloc((size_t)16 * N * 4); double *nr = malloc((size_t)N * 8); memset(ps, 0, sizeof(ps));
    if (!pbs || !nr) { free(pbs); free(nr); return -3; }
    for (int i = 0; i < N; i++) g[i] = 1.f;
    for (size_t i = 0; i < (size_t)nl * N; i++) bs[i] = 1.f;
    for (int l = 0; l < nl && !err; l++)
    {
        patch_t p; t_lki = l; if (!lrun_load(libs[l].dir, pre, K, N, libs[l].st, &p)) { patch_free(&p); continue; }
        if (p.mag && p.maxis == 1) { err = -6; patch_free(&p); break; }
        if (p.mag)
        {
            pthread_t th[BAKE_TH]; lg_job_t jb[BAKE_TH];
            for (int t = 0; t < BAKE_TH; t++) { jb[t] = (lg_job_t){ w, ps, pbs, np, N * t / BAKE_TH, N * (t + 1) / BAKE_TH, nr }; pthread_create(&th[t], 0, lg_part, &jb[t]); }
            for (int t = 0; t < BAKE_TH; t++) pthread_join(th[t], 0);
            for (int i = 0; i < N; i++)
            {
                const float sc = p.mag[i] / (fabsf(g[i]) * (float)nr[i] + 1.1920929e-7f), s = p.st; float ng = g[i] * (1.f - s + s * sc);
                if (fabsf(ng) < 1e-6f) ng = ng < 0 ? -1e-6f : 1e-6f;
                bs[(size_t)l * N + i] = sc / ng; g[i] = ng;
            }
        }
        else for (int i = 0; i < N; i++) bs[(size_t)l * N + i] = 1.f / g[i];
        for (int i = 0; i < N; i++) pbs[(size_t)np * N + i] = bs[(size_t)l * N + i];
        ps[np++] = p;
    }
    for (int l = 0; l < np; l++) patch_free(&ps[l]);
    free(pbs); free(nr); return err;
}
static void lrun_put_gain(__fp16 *gv, const mat_t *hm, int N, const float *g)
{ for (int n = 0; n < N; n++) { const __fp16 v = g ? (__fp16)g[mat_src(hm, n)] : (__fp16)1.f; gv[(size_t)(n / 32) * 64 + 2 * (n % 32)] = v; gv[(size_t)(n / 32) * 64 + 2 * (n % 32) + 1] = v; } }
#define LR_MAX 4096
static size_t lrun_size(int K, int N, int lr, int gain) { return lr ? 256 + ((size_t)K + N) * lr * 2 + (gain ? (size_t)N * 4 : 0) : 128; }
/* the rank this Linear needs over the set, rounded up to the HMX's groups, and whether a DoRA touches it */
static int lrun_rank(const setlib_t *libs, int nl, const char *pre, int K, int N, int *dora)
{
    int r = 0; *dora = 0;
    for (int l = 0; l < nl; l++)
    {
        char p[768]; snprintf(p, sizeof(p), "%s/%s.dora.f32", libs[l].dir, pre); if (access(p, R_OK) == 0) *dora = 1;
        snprintf(p, sizeof(p), "%s/%s.lokr.f32", libs[l].dir, pre);
        if (access(p, R_OK) == 0) { patch_t q; t_lki = l; if (lrun_load(libs[l].dir, pre, K, N, libs[l].st, &q)) r += q.R; patch_free(&q); continue; }
        snprintf(p, sizeof(p), "%s/%s.A.f16", libs[l].dir, pre); FILE *f = fopen(p, "rb"); if (f) { fseek(f, 0, SEEK_END); r += (int)(ftell(f) / 2 / K); fclose(f); }
    }
    return (r + 63) / 64 * 64;
}
/* One Linear into the blob, as lora_next reads it. Its A^T in `split` parts as its weights are stored; B's columns in the engine's row order (heads); strength and a DoRA's gain folded in. */
static __fp16 *lrun_linear(__fp16 *q, const setlib_t *libs, int nl, const char *pre, int K, int N, int split, int heads, int lr, int gl, const float *lg, const float *lbs)
{
    uint16_t *h = (uint16_t *)q; h[0] = (uint16_t)lr; h[1] = (uint16_t)gl; q += 64; if (!lr) return q;
    __fp16 *la = q, *h2 = la + (size_t)K * lr, *lbm = h2 + 64; ((uint16_t *)h2)[0] = (uint16_t)lr; ((uint16_t *)h2)[1] = (uint16_t)gl; q = lbm + (size_t)lr * N;
    const int Kq = K / split; const mat_t mb = { lbm, lr, N, 0, 0, 1 }, hm = { 0, K, N, 0, heads, 1 }; int off = 0;
    for (int l = 0; l < nl; l++)
    {
        patch_t p; t_lki = l; if (!lrun_load(libs[l].dir, pre, K, N, libs[l].st, &p)) { patch_free(&p); continue; }
        for (int r = 0; r < p.R; r++)
        {
            for (int k = 0; k < K; k++)
            {
                const mat_t ma = { la + (size_t)(k / Kq) * Kq * lr, Kq, lr, 0, 0, 1 };
                mat_set(&ma, mat_at(&ma, off + r, k % Kq), p.a[(size_t)r * K + k]);
            }
            for (int n = 0; n < N; n++)
            {
                const int src = mat_src(&hm, n);
                mat_set(&mb, mat_at(&mb, n, off + r), p.b[(size_t)src * p.R + r] * p.st * (gl ? lbs[(size_t)l * N + src] : 1.f));
            }
        }
        off += p.R; patch_free(&p);
    }
    if (gl) { lrun_put_gain(q, &hm, N, lg); q += (size_t)N * 2; }
    return q;
}
/* the gains for one Linear, the block's base weights read the first time they are needed */
static int lrun_mat_gains(const setlib_t *libs, int nl, const char *pre, const mat_t *mw, const char *dir, const char *bn, __fp16 *wb, const size_t *msz, int nmat, int fmt, int *wread, float *lg, float *lbs)
{
    if (!*wread && read_block16(dir, bn, wb, msz, nmat, fmt)) return -1;
    *wread = 1; return lrun_gains(libs, nl, pre, mw, lg, lbs);
}
/* nLora's two passes over the blocks, a block per job: the ranks (a LoKr's QR), then the blob (the QR again, a DoRA's gains over
 * the block's weights). One thread took 28 s for a LoKr + DoRA at 448 ranks, so the blocks are shared out. */
#define NL_TH 4
typedef struct { eng_t *e; const setlib_t *libs; int nl, t, *rk; char *dk; size_t *bsz, *mats, lb; uint16_t *blob; int ret; lk_t *lk; } nl_job_t;
static void *nl_rank_part(void *v)
{
    nl_job_t *j = (nl_job_t *)v;
    for (int i = j->t; i < j->e->nb; i += NL_TH)
    {
        size_t bs = 0;
        for (int m = 0; m < 10; m++)
        {
            char pre[96]; snprintf(pre, sizeof(pre), "b%d_%s", i, LKEY[m]); int d = 0; t_lk = j->lk + (size_t)(i * 10 + m) * 16; const int r = lrun_rank(j->libs, j->nl, pre, LIN[m], LOUT[m], &d);
            j->rk[i * 10 + m] = r; j->dk[i * 10 + m] = (char)d; bs += lrun_size(LIN[m], LOUT[m], r, d);
        }
        j->bsz[i] = bs;
    }
    t_lk = 0; return 0;
}
static void *nl_build_part(void *v)
{
    nl_job_t *j = (nl_job_t *)v; eng_t *e = j->e; size_t nW = 0; int anyd = 0;
    for (int m = 0; m < 10; m++) nW += j->mats[m];
    for (int i = j->t; i < e->nb; i += NL_TH) for (int m = 0; m < 10; m++) if (j->dk[i * 10 + m]) anyd = 1;
    __fp16 *wb = anyd ? malloc(nW * 2) : 0; float *lg = anyd ? malloc((size_t)MAXN_L * 4) : 0, *lbs = anyd ? malloc((size_t)16 * MAXN_L * 4) : 0;
    if (anyd && (!wb || !lg || !lbs)) j->ret = -3;
    for (int i = j->t; i < e->nb && j->ret > 0; i += NL_TH)
    {
        __fp16 *q = (__fp16 *)((uint8_t *)j->blob + (size_t)i * j->lb); int wread = 0; size_t woff = 0; char bn[64]; snprintf(bn, sizeof(bn), "weights_b%d.f16", i);
        for (int m = 0; m < 10 && j->ret > 0; m++)
        {
            char pre[96]; snprintf(pre, sizeof(pre), "b%d_%s", i, LKEY[m]); const mat_t mw = { wb ? wb + woff : 0, LIN[m], LOUT[m], 0, 0, 1 }; woff += j->mats[m]; const int gl = j->dk[i * 10 + m]; t_lk = j->lk + (size_t)(i * 10 + m) * 16;
            if (gl) { const int ge = lrun_mat_gains(j->libs, j->nl, pre, &mw, e->dir, bn, wb, j->mats, 10, e->fmt, &wread, lg, lbs); if (ge) { j->ret = ge; break; } }
            q = lrun_linear(q, j->libs, j->nl, pre, LIN[m], LOUT[m], 1, 0, j->rk[i * 10 + m], gl, lg, lbs);
        }
    }
    free(wb); free(lg); free(lbs);
    t_lk = 0; return 0;
}
static double nl_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
/* Anima: per block its ten Linears, kept in memory and put after the block's weights just before it runs. Returns the largest rank, 0 for a set with nothing for the blocks,
 * or < 0: -6 a DoRA on the input axis, -8 more ranks than the DSP takes in a Linear, -3 no memory. */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nLora(JNIEnv *env, jclass cl, jlong jh, jstring jset)
{
    eng_t *e = (eng_t *)(intptr_t)jh; if (!e) return -1;
    const char *set = (*env)->GetStringUTFChars(env, jset, 0);
    setlib_t libs[16]; const int nl = read_set(set, libs, 16);
    (*env)->ReleaseStringUTFChars(env, jset, set);
    free(e->lora); e->lora = 0;
    if (nl <= 0) return nl < 0 ? -4 : 0;
    const double t0 = nl_ms();
    int *rk = malloc((size_t)e->nb * 10 * sizeof(int)), maxr = 0; char *dk = calloc((size_t)e->nb, 10); size_t *bsz = calloc((size_t)e->nb, sizeof(size_t)), lb = 0; size_t mats[10], nW = 0;
    lk_t *lk = calloc((size_t)e->nb * 10 * 16, sizeof(lk_t));
    if (!rk || !dk || !bsz || !lk) { free(rk); free(dk); free(bsz); free(lk); return -3; }
    for (int m = 0; m < 10; m++) { mats[m] = (size_t)LIN[m] * LOUT[m]; nW += mats[m]; }
    pthread_t th[NL_TH]; nl_job_t jb[NL_TH];
    for (int t = 0; t < NL_TH; t++) { jb[t] = (nl_job_t){ e, libs, nl, t, rk, dk, bsz, mats, 0, 0, 1, lk }; pthread_create(&th[t], 0, nl_rank_part, &jb[t]); }
    for (int t = 0; t < NL_TH; t++) pthread_join(th[t], 0);
    for (int i = 0; i < e->nb; i++) { if (bsz[i] > lb) lb = bsz[i]; for (int m = 0; m < 10; m++) { if (rk[i * 10 + m] > maxr) maxr = rk[i * 10 + m]; } }
    free(bsz); const double t1 = nl_ms();
    int ret = maxr; uint16_t *blob = 0;
    /* the read buffers were made for 512 ranks everywhere; a full-rank set or a big LoKr may want more */
    if (e->wroom < lb && maxr && maxr <= LR_MAX) { void *n0 = ralloc(wbytes(e->nW, e->fmt) + lb), *n1 = ralloc(wbytes(e->nW, e->fmt) + lb); if (!n0 || !n1) { if (n0) rpcmem_free(n0); if (n1) rpcmem_free(n1); ret = -3; goto out; } rpcmem_free(e->wb[0]); rpcmem_free(e->wb[1]); e->wb[0] = n0; e->wb[1] = n1; e->wroom = lb; }
    if (maxr == 0) goto out;
    if (maxr > LR_MAX) { ret = -8; goto out; }
    blob = calloc((size_t)e->nb, lb);
    if (!blob) { ret = -3; goto out; }
    for (int t = 0; t < NL_TH; t++) { jb[t] = (nl_job_t){ e, libs, nl, t, rk, dk, 0, mats, lb, blob, ret, lk }; pthread_create(&th[t], 0, nl_build_part, &jb[t]); }
    for (int t = 0; t < NL_TH; t++) pthread_join(th[t], 0);
    for (int t = 0; t < NL_TH; t++) if (jb[t].ret <= 0) ret = jb[t].ret;
    if (ret <= 0) { free(blob); blob = 0; }
out:
    for (size_t i = 0; i < (size_t)e->nb * 10 * 16; i++) { free(lk[i].q1); free(lk[i].r1); free(lk[i].q2); free(lk[i].r2); }
    free(lk); free(rk); free(dk);
    if (blob) { e->lora = blob; e->lbytes = lb; LOG("lora as it runs: %d LoRAs, up to %d ranks, %.1f MB a block, ranks %.1f s, blob %.1f s (%d threads)", nl, maxr, lb / 1048576.0, (t1 - t0) / 1e3, (nl_ms() - t1) / 1e3, NL_TH); }
    return ret;
}

/* ------------------------------------------------------------------ Krea 2 and MOSS-Image: the same (kblock_k2run's flag 0x1000)
 * Per block, per Linear (wq, wk, wv, attn gate, wo, mlp gate, up, down): B's columns in the engine's own row order (wq / wk head-permuted), down's A^T in quarters as its weights are stored.
 * Null with *lr_out < 0 saying why (as nLora's), or nothing for the blocks. */
/* g_k2moss (kengine.c): the engine being opened is MOSS-Image's, set by nK2Open around the build */
static uint16_t *k2_lora_build(const char *set, int nb, int *lr_out, size_t *bytes_out, const char *wdir)
{
    static const char *kmats_k[8] = { "attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down" };
    static const int K8k[8] = { KD, KD, KD, KD, KD, KD, KD, KF }, N8k[8] = { KD, KKV, KKV, KD, KD, KF, KF, KD }, H8k[8] = { 48, 12, 0, 0, 0, 0, 0, 0 };
    /* MOSS-Image: its files named as the release's modules under layers.<i> (ffn_up_gate split into ffn_gate and ffn_up), no attention gate, no head permutation, down's A as four K = 2048 quarters */
    static const char *kmats_m[8] = { "proj.q", "proj.k", "proj.v", "", "proj.o", "proj.ffn_gate", "proj.ffn_up", "proj.ffn_down" };
    static const int K8m[8] = { 3072, 3072, 3072, 0, 3072, 3072, 3072, 8192 }, N8m[8] = { 3072, 3072, 3072, 0, 3072, 8192, 8192, 3072 }, H8m[8] = { 0 };
    const char *const *kmats = g_k2moss ? kmats_m : kmats_k; const int *K8 = g_k2moss ? K8m : K8k, *N8 = g_k2moss ? N8m : N8k, *H8 = g_k2moss ? H8m : H8k;
    const char *bp = g_k2moss ? "layers" : "blocks";
    *lr_out = 0; *bytes_out = 0;
    setlib_t libs[16]; const int nl = read_set(set, libs, 16); if (nl <= 0) return 0;
    int *rk = malloc((size_t)nb * 8 * sizeof(int)), maxr = 0, anyd = 0, err = 0; char *dk = calloc((size_t)nb, 8); size_t lb = 0; if (!rk || !dk) { free(rk); free(dk); *lr_out = -3; return 0; }
    for (int i = 0; i < nb; i++)
    {
        size_t bs = 0;
        for (int m = 0; m < 8; m++)
        {
            if (!K8[m]) continue;
            char pre[96]; snprintf(pre, sizeof(pre), "%s.%d.%s", bp, i, kmats[m]); int d = 0; const int r = lrun_rank(libs, nl, pre, K8[m], N8[m], &d);
            rk[i * 8 + m] = r; dk[i * 8 + m] = (char)d; if (r > maxr) maxr = r; if (d) anyd = 1; bs += lrun_size(K8[m], N8[m], r, d);
        }
        if (bs > lb) lb = bs;
    }
    const int wfmt = wdir ? fmt_of(wdir) : 0; uint16_t *blob = 0; __fp16 *wb = 0; float *lg = 0, *lbs = 0; size_t msz[11], nWb = 0; int nmat = 0;
    if (maxr == 0) goto out;
    if (maxr > LR_MAX) { err = -8; goto out; }
    if (anyd && !wdir) { err = -6; goto out; }
    if (anyd) { for (int m = 0; m < 7; m++) if (K8[m]) msz[nmat++] = (size_t)K8[m] * N8[m]; for (int q = 0; q < 4; q++) msz[nmat++] = (size_t)K8[7] / 4 * N8[7]; for (int m = 0; m < nmat; m++) nWb += msz[m]; wb = malloc(nWb * 2); lg = malloc((size_t)MAXN_L * 4); lbs = malloc((size_t)16 * MAXN_L * 4); }
    blob = calloc((size_t)nb, lb);
    if (!blob || (anyd && (!wb || !lg || !lbs))) { err = -3; goto out; }
    g_bake_done = 0;
    for (int i = 0; i < nb && !err; i++)
    {
        __fp16 *q = (__fp16 *)((uint8_t *)blob + (size_t)i * lb); int wread = 0; size_t woff = 0; char bn[64]; snprintf(bn, sizeof(bn), "weights_b%d.f16", i);
        for (int m = 0; m < 8 && !err; m++)
        {
            if (!K8[m]) continue;
            const int K = K8[m], N = N8[m], split = m == 7 ? 4 : 1, gl = dk[i * 8 + m];
            char pre[96]; snprintf(pre, sizeof(pre), "%s.%d.%s", bp, i, kmats[m]);
            const mat_t mw = { wb ? wb + woff : 0, K, N, 0, H8[m], split }; woff += (size_t)K * N;
            if (gl) { const int ge = lrun_mat_gains(libs, nl, pre, &mw, wdir, bn, wb, msz, nmat, wfmt, &wread, lg, lbs); if (ge) { err = ge; break; } }
            q = lrun_linear(q, libs, nl, pre, K, N, split, H8[m], rk[i * 8 + m], gl, lg, lbs);
        }
        g_bake_done = i + 1;
    }
out:
    free(rk); free(dk); free(wb); free(lg); free(lbs);
    if (err) { free(blob); *lr_out = err; return 0; }
    if (!blob) return 0;
    *lr_out = maxr; *bytes_out = lb;
    LOG("k2 lora as it runs: %d LoRAs, up to %d ranks, %.1f MB a block", nl, maxr, lb / 1048576.0);
    return blob;
}
/* FLUX.2 Klein's blocks: per block per stream (double: the text, the image) q, k (B head-permuted), v, out, gate, up, down (A as three K = kD parts), from the files LoraImport.convertKlein writes: kb.d<i>.<stream>.<m> and kb.s<i>.<m>. */
static uint16_t *klein_lora_build(const char *set, int nd, int nb, int kD, int *lr_out, size_t *bytes_out)
{
    static const char mn[7] = { 'q', 'k', 'v', 'o', 'g', 'u', 'd' }; const int K7[7] = { kD, kD, kD, kD, kD, kD, 3 * kD }, N7[7] = { kD, kD, kD, kD, 3 * kD, 3 * kD, kD }, H7[7] = { kD / 128, kD / 128, 0, 0, 0, 0, 0 };
    *lr_out = 0; *bytes_out = 0;
    setlib_t libs[16]; const int nl = read_set(set, libs, 16); if (nl <= 0) return 0;
    int *rk = malloc((size_t)nb * 14 * sizeof(int)), maxr = 0; size_t lb = 0; char pre[96]; if (!rk) { *lr_out = -3; return 0; }
    for (int i = 0; i < nb; i++)
    {
        size_t bs = 0;
        for (int s = 0; s < (i < nd ? 2 : 1); s++) for (int m = 0; m < 7; m++)
        {
            if (i < nd) snprintf(pre, sizeof(pre), "kb.d%d.%d.%c", i, s, mn[m]); else snprintf(pre, sizeof(pre), "kb.s%d.%c", i - nd, mn[m]);
            int d = 0; const int r = lrun_rank(libs, nl, pre, K7[m], N7[m], &d); if (d) { free(rk); *lr_out = -6; return 0; }
            rk[i * 14 + s * 7 + m] = r; if (r > maxr) maxr = r; bs += lrun_size(K7[m], N7[m], r, 0);
        }
        if (bs > lb) lb = bs;
    }
    if (maxr == 0) { free(rk); return 0; }
    if (maxr > LR_MAX) { free(rk); *lr_out = -8; return 0; }
    uint16_t *blob = calloc((size_t)nb, lb); if (!blob) { free(rk); *lr_out = -3; return 0; }
    for (int i = 0; i < nb; i++)
    {
        __fp16 *q = (__fp16 *)((uint8_t *)blob + (size_t)i * lb);
        for (int s = 0; s < (i < nd ? 2 : 1); s++) for (int m = 0; m < 7; m++)
        {
            if (i < nd) snprintf(pre, sizeof(pre), "kb.d%d.%d.%c", i, s, mn[m]); else snprintf(pre, sizeof(pre), "kb.s%d.%c", i - nd, mn[m]);
            q = lrun_linear(q, libs, nl, pre, K7[m], N7[m], m == 6 ? 3 : 1, H7[m], rk[i * 14 + s * 7 + m], 0, 0, 0);
        }
    }
    free(rk); *lr_out = maxr; *bytes_out = lb; LOG("klein lora as it runs: %d LoRAs, up to %d ranks", nl, maxr);
    return blob;
}
/* the rank slots the open K2 engine runs its LoRAs with: 0 none, < 0 why the set could not be run as it goes (nLora's codes) */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nK2LoraSlots(JNIEnv *env, jclass cl, jlong jh) { k2_t *e = (k2_t *)(intptr_t)jh; return e ? e->lr : 0; }

/* direct ByteBuffers over native memory (zeroed), for big tensors the app hands to ONNX Runtime: allocateDirect counts against the
 * Java heap */
JNIEXPORT jobject JNICALL Java_dev_kroma_open_Engine_nDirect(JNIEnv *env, jclass cl, jlong bytes)
{ void *p = calloc(1, (size_t)bytes); return p ? (*env)->NewDirectByteBuffer(env, p, bytes) : 0; }
JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nFreeDirect(JNIEnv *env, jclass cl, jobject buf)
{ void *p = buf ? (*env)->GetDirectBufferAddress(env, buf) : 0; free(p); }
