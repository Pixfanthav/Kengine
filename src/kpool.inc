/* kpool.inc (included by kblock_imp.c) -- the engine's own worker threads on QuRT: a fixed set of threads for one call, a ring of jobs, and completion
 * tokens. Written for this engine, so the skel carries no prebuilt library. Jobs are short and the queue is small. */
typedef void (*worker_callback_t)(void *);
typedef struct { worker_callback_t fptr; void *dptr; } worker_pool_job_t;
typedef struct { volatile int n; volatile int done; } worker_synctoken_t;
enum { KP_MAXT = 12, KP_RING = 128, KP_STACK = 64 * 1024 };
typedef struct {
    qurt_thread_t tid[KP_MAXT]; void *stack[KP_MAXT]; int nt; volatile int stop;
    worker_pool_job_t ring[KP_RING]; volatile unsigned head, tail; qurt_mutex_t mu; qurt_sem_t sem;
} kpool_t;
typedef kpool_t *worker_pool_context_t;
static unsigned int num_workers, num_hvx128_contexts;

static void kp_main(void *a)
{
    kpool_t *p = (kpool_t *)a;
    for (;;)
    {
        qurt_sem_down(&p->sem);
        qurt_mutex_lock(&p->mu);
        if (p->head == p->tail) { qurt_mutex_unlock(&p->mu); if (p->stop) break; continue; }
        const worker_pool_job_t j = p->ring[p->head % KP_RING]; p->head++;
        qurt_mutex_unlock(&p->mu);
        j.fptr(j.dptr);
    }
    qurt_thread_exit(0);
}
static int worker_pool_submit(worker_pool_context_t p, worker_pool_job_t job)
{
    qurt_mutex_lock(&p->mu); p->ring[p->tail % KP_RING] = job; p->tail++; qurt_mutex_unlock(&p->mu);
    qurt_sem_up(&p->sem); return 0;
}
static int worker_pool_init(worker_pool_context_t *ctx)
{
    kpool_t *p = (kpool_t *)calloc(1, sizeof(kpool_t)); if (!p) return AEE_ENOMEMORY;
    qurt_mutex_init(&p->mu); qurt_sem_init_val(&p->sem, 0);
    qurt_sysenv_max_hthreads_t mh = { 0 }; qurt_sysenv_get_max_hw_threads(&mh);
    int n = (int)(mh.max_hthreads & 0xff); if (n < 2) n = 2; if (n > KP_MAXT) n = KP_MAXT;
    const int units = qurt_hvx_get_units(); num_hvx128_contexts = (units >> 8) & 0xff; num_workers = n;
    const unsigned short pr = qurt_thread_get_priority(qurt_thread_get_id());
    for (int i = 0; i < n; i++)
    {
        qurt_thread_attr_t at; qurt_thread_attr_init(&at); p->stack[i] = memalign(128, KP_STACK); if (!p->stack[i]) break;
        qurt_thread_attr_set_stack_addr(&at, p->stack[i]); qurt_thread_attr_set_stack_size(&at, KP_STACK);
        qurt_thread_attr_set_priority(&at, pr); qurt_thread_attr_set_name(&at, (char *)"kpool");
        if (qurt_thread_create(&p->tid[i], &at, kp_main, p) != QURT_EOK) { free(p->stack[i]); p->stack[i] = 0; break; }
        p->nt = i + 1;
    }
    *ctx = p; return p->nt ? 0 : AEE_EFAILED;
}
static void worker_pool_deinit(worker_pool_context_t *ctx)
{
    kpool_t *p = *ctx; if (!p) return;
    p->stop = 1; for (int i = 0; i < p->nt; i++) qurt_sem_up(&p->sem);
    for (int i = 0; i < p->nt; i++) { int st; qurt_thread_join(p->tid[i], &st); free(p->stack[i]); }
    qurt_mutex_destroy(&p->mu); qurt_sem_destroy(&p->sem); free(p); *ctx = 0;
}
static inline void worker_pool_synctoken_init(worker_synctoken_t *t, unsigned n) { t->n = (int)n; __atomic_store_n(&t->done, 0, __ATOMIC_RELEASE); }
static inline void worker_pool_synctoken_jobdone(worker_synctoken_t *t) { const int d = __atomic_add_fetch(&t->done, 1, __ATOMIC_ACQ_REL); if (d >= t->n) qurt_futex_wake((void *)&t->done, 1); }
static inline void worker_pool_synctoken_wait(worker_synctoken_t *t) { int d; while ((d = __atomic_load_n(&t->done, __ATOMIC_ACQUIRE)) < t->n) qurt_futex_wait((void *)&t->done, d); }
