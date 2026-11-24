/* moss.inc (included by kengine.c) -- MOSS-Image's small parts on the CPU: the text refiner (text_embedder), the input and output
 * layers, the time embedding and modulations. kconv writes them to moss_aux.f16 (fp16 row-major, torch Linear weights [N x K]) with
 * moss_aux.json (name -> shape, byte offset), which the app reads; here only the products the app asks for: out = x W^T (+ b). */
#include <sys/mman.h>
#include <fcntl.h>
#include <arm_neon.h>

typedef struct { const __fp16 *w; const __fp16 *b; const float *x; float *out; int M, K, N, n0, n1; } mlin_t;
/* rows n0..n1-1 of W against every row of x: 4 x 4 tiles of fp32 accumulators, the weights widened from fp16 as they are read */
static void *mlin_part(void *a_)
{
    const mlin_t *a = (const mlin_t *)a_; const int K = a->K, M = a->M, N = a->N;
    int n = a->n0;
    for (; n + 4 <= a->n1; n += 4)
    {
        const __fp16 *w0 = a->w + (size_t)n * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        int m = 0;
        for (; m + 4 <= M; m += 4)
        {
            const float *x0 = a->x + (size_t)m * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
            float32x4_t c[4][4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c[i][j] = vdupq_n_f32(0.f);
            for (int k = 0; k < K; k += 4)
            {
                const float32x4_t W0 = vcvt_f32_f16(vld1_f16(w0 + k)), W1 = vcvt_f32_f16(vld1_f16(w1 + k)), W2 = vcvt_f32_f16(vld1_f16(w2 + k)), W3 = vcvt_f32_f16(vld1_f16(w3 + k));
                const float32x4_t X[4] = { vld1q_f32(x0 + k), vld1q_f32(x1 + k), vld1q_f32(x2 + k), vld1q_f32(x3 + k) };
                for (int i = 0; i < 4; i++) { c[i][0] = vfmaq_f32(c[i][0], X[i], W0); c[i][1] = vfmaq_f32(c[i][1], X[i], W1); c[i][2] = vfmaq_f32(c[i][2], X[i], W2); c[i][3] = vfmaq_f32(c[i][3], X[i], W3); }
            }
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
                a->out[(size_t)(m + i) * N + n + j] = vaddvq_f32(c[i][j]) + (a->b ? (float)a->b[n + j] : 0.f);
        }
        for (; m < M; m++)
            for (int j = 0; j < 4; j++)
            {
                const __fp16 *wr = a->w + (size_t)(n + j) * K; const float *xr = a->x + (size_t)m * K; float32x4_t c = vdupq_n_f32(0.f);
                for (int k = 0; k < K; k += 4) c = vfmaq_f32(c, vld1q_f32(xr + k), vcvt_f32_f16(vld1_f16(wr + k)));
                a->out[(size_t)m * N + n + j] = vaddvq_f32(c) + (a->b ? (float)a->b[n + j] : 0.f);
            }
    }
    for (; n < a->n1; n++)
        for (int m = 0; m < M; m++)
        {
            const __fp16 *wr = a->w + (size_t)n * K; const float *xr = a->x + (size_t)m * K; float s = 0.f;
            for (int k = 0; k < K; k++) s += xr[k] * (float)wr[k];
            a->out[(size_t)m * N + n] = s + (a->b ? (float)a->b[n] : 0.f);
        }
    return 0;
}

typedef struct { void *map; size_t len; } maux_t;
/* moss_aux.f16 mapped (read-only, paged in as used); 0 when it cannot be */
JNIEXPORT jlong JNICALL Java_dev_kroma_open_Engine_nMossAux(JNIEnv *env, jclass cl, jstring jpath)
{
    const char *p = (*env)->GetStringUTFChars(env, jpath, 0); int fd = open(p, O_RDONLY); (*env)->ReleaseStringUTFChars(env, jpath, p);
    if (fd < 0) return 0;
    const off_t len = lseek(fd, 0, SEEK_END); void *m = mmap(0, (size_t)len, PROT_READ, MAP_SHARED, fd, 0); close(fd);
    if (m == MAP_FAILED) return 0;
    maux_t *a = malloc(sizeof(maux_t)); a->map = m; a->len = (size_t)len; return (jlong)(intptr_t)a;
}
JNIEXPORT void JNICALL Java_dev_kroma_open_Engine_nMossAuxClose(JNIEnv *env, jclass cl, jlong jh)
{ maux_t *a = (maux_t *)(intptr_t)jh; if (!a) return; klw_flush(); munmap(a->map, a->len); free(a); }   /* klw_flush: the kept K2.linear weights */

/* out [M x N] = x [M x K] W^T + b: W at byte offset woff of the mapped file ([N x K] fp16), b at boff (-1: none); 8 threads over N */
JNIEXPORT jint JNICALL Java_dev_kroma_open_Engine_nMossLinear(JNIEnv *env, jclass cl, jlong jh, jlong woff, jlong boff, jint N, jint K,
                                                              jfloatArray jx, jint M, jfloatArray jout)
{
    maux_t *a = (maux_t *)(intptr_t)jh; if (!a || K % 4) return -1;
    if ((size_t)woff + (size_t)N * K * 2 > a->len || (boff >= 0 && (size_t)boff + (size_t)N * 2 > a->len)) return -2;
    if ((*env)->GetArrayLength(env, jx) < M * K || (*env)->GetArrayLength(env, jout) < M * N) return -3;
    float *x = (*env)->GetPrimitiveArrayCritical(env, jx, 0), *o = (*env)->GetPrimitiveArrayCritical(env, jout, 0);
    enum { NT = 8 }; pthread_t th[NT]; mlin_t ar[NT];
    const __fp16 *w = (const __fp16 *)((const uint8_t *)a->map + woff), *b = boff >= 0 ? (const __fp16 *)((const uint8_t *)a->map + boff) : 0;
    const int per = ((N + NT - 1) / NT + 3) & ~3;
    int nt = 0;
    for (int t = 0; t < NT; t++)
    {
        const int n0 = t * per, n1 = n0 + per < N ? n0 + per : N; if (n0 >= n1) break;
        ar[t] = (mlin_t){ w, b, x, o, M, K, N, n0, n1 }; pthread_create(&th[t], 0, mlin_part, &ar[t]); nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], 0);
    (*env)->ReleasePrimitiveArrayCritical(env, jout, o, 0); (*env)->ReleasePrimitiveArrayCritical(env, jx, x, JNI_ABORT);
    return 0;
}

