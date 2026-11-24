/* kdsp.h: what the engine asks of the DSP's system, by value: the power request, the compute resources, the threads. The functions are the phone's own (resolved when the library loads). */
/* Where the values come from: written by hand, no SDK file copied. The power request (HAP_power_set with
 * DCVS v3, HVX and HMX on, the HMX clock request), the compute resources (VTCM, HMX lock) and the QuRT thread calls are used the same way,
 * with the same values, by llama.cpp's ggml-hexagon backend (ggml/src/ggml-hexagon/htp, MIT License, Copyright (c) 2023-2025 The ggml
 * authors). What has no public source is only the byte size and field offsets of these structures, which the static asserts below state
 * (facts about the interface, not code). */
#ifndef KDSP_H
#define KDSP_H
#pragma clang diagnostic ignored "-Wunused-function"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* the compiler's loop-to-memcpy rewrite is off in the build (it made the HVX copies 10 times slower); the copies written out stay inline when small */
#define memcpy __builtin_memcpy
#define memset __builtin_memset

/* the system's heap */
void *__wrap_malloc(size_t n); void *__wrap_calloc(size_t a, size_t b); void __wrap_free(void *p); void *__wrap_memalign(size_t a, size_t n);
#define malloc __wrap_malloc
#define calloc __wrap_calloc
#define free __wrap_free
#define memalign __wrap_memalign

#define TRUE 1
#define FALSE 0
#define AEE_EFAILED ((int)0x80000401)
#define AEE_ENOMEMORY ((int)0x80000402)
#define AEE_EBADPARM ((int)0x8000040E)
#define AEE_ERESOURCENOTFOUND ((int)0x80000416)

/* the timer: 19.2 MHz */
static inline uint64_t HAP_perf_get_qtimer_count(void) { unsigned long long c; __asm__ volatile(" %0 = c31:30 " : "=r"(c)); return c; }
static inline uint64_t HAP_perf_get_time_us(void) { return HAP_perf_get_qtimer_count() * 10ull / 192ull; }

/* the power request: 120 bytes, the payload at 8 */
enum { HAP_power_set_HVX = 2, HAP_power_set_DCVS_v3 = 12, HAP_power_set_HMX = 13, HAP_power_set_HMX_v2 = 14 };
/* HMX_v2 (v75 on): the HMX on and its own clock voted -- set_power, power_up, set_clock, pick_default, then the corners (target, min,
 * max) and perf_mode (0 = the highest frequency in the corner), as llama.cpp's htp/main.c asks for its burst mode (corners 0xFFFF) */
typedef struct { uint8_t set_power, power_up, set_clock, pick_default; uint32_t target_corner, min_corner, max_corner, perf_mode, freq_mhz, floor_freq_mhz, param[3]; } khmx_v2_t;
enum { HAP_DCVS_EXP_VCORNER_MAX = 0xFFFF };
enum { HAP_DCVS_V2_PERFORMANCE_MODE = 16, HAP_DCVS_VCORNER_MAX = 255 };
typedef struct { uint8_t target_corner, min_corner, max_corner, pad; uint32_t param[3]; } kdcvs_params_t;
typedef struct {
    uint8_t set_dcvs_enable, dcvs_enable, dcvs_option, pad0[5];
    uint8_t set_core_params, pad1[3]; kdcvs_params_t core_params;   /* at 8, 12 */
    uint8_t set_bus_params, pad2[3]; kdcvs_params_t bus_params;     /* at 28, 32 */
    uint8_t pad3[28];
    uint8_t set_sleep_disable, sleep_disable, pad4[2];              /* at 76, 77; 80 bytes */
} kdcvs_v3_t;
typedef struct { uint8_t power_up; } kpower_up_t;
typedef struct { uint32_t type; uint32_t pad; union { kdcvs_v3_t dcvs_v3; kpower_up_t hmx, hvx; khmx_v2_t hmx_v2; uint8_t bytes[112]; }; } HAP_power_request_t;
_Static_assert(sizeof(khmx_v2_t) == 40 && offsetof(khmx_v2_t, target_corner) == 4 && offsetof(khmx_v2_t, perf_mode) == 16, "the HMX v2 request");
_Static_assert(offsetof(kdcvs_v3_t, core_params) == 12 && offsetof(kdcvs_v3_t, set_bus_params) == 28 && offsetof(kdcvs_v3_t, bus_params) == 32 && offsetof(kdcvs_v3_t, set_sleep_disable) == 76 && sizeof(kdcvs_v3_t) == 80, "the DCVS request");
_Static_assert(sizeof(HAP_power_request_t) == 120 && offsetof(HAP_power_request_t, dcvs_v3) == 8, "the power request");
int HAP_power_set(void *context, HAP_power_request_t *request);

/* the compute resources */
#define HAP_COMPUTE_RES_PREEMPTION_CAPABILITY 1
#define HAP_COMPUTE_RES_HMX_LOCK4 16
typedef struct { uint64_t opaque[20]; } compute_res_attr_t;
typedef struct { uint32_t opaque[18]; } compute_res_vtcm_page_t;
#define KWEAK __attribute__((weak))
int KWEAK compute_resource_attr_init(compute_res_attr_t *attr);
int KWEAK compute_resource_attr_init_v2(compute_res_attr_t *attr, unsigned int size, unsigned int version);
int KWEAK compute_resource_attr_set_hmx_param(compute_res_attr_t *attr, unsigned char b_enable);
int KWEAK compute_resource_attr_set_vtcm_param(compute_res_attr_t *attr, unsigned int size, unsigned char b_single_page);
void *KWEAK compute_resource_attr_get_vtcm_ptr(compute_res_attr_t *attr);
int KWEAK compute_resource_query_VTCM(unsigned int app, unsigned int *total, compute_res_vtcm_page_t *total_layout, unsigned int *avail, compute_res_vtcm_page_t *avail_layout);
unsigned int KWEAK compute_resource_acquire(compute_res_attr_t *attr, unsigned int timeout_us);
int KWEAK compute_resource_release(unsigned int ctx);
int KWEAK compute_resource_hmx_lock(unsigned int ctx);
int KWEAK compute_resource_hmx_unlock(unsigned int ctx);
int KWEAK crm_query_capability(int id, unsigned int *data);
int KWEAK crm_hmx_lock4(unsigned int ctx, unsigned int **hmx_busy, unsigned int timeout_us);
int KWEAK crm_hmx_unlock4(unsigned int ctx, unsigned int *hmx_busy);
#define KNOTSUP ((int)0x80000414)
static inline int HAP_compute_res_attr_init(compute_res_attr_t *a)
{
    if (compute_resource_attr_init_v2) { const int r = compute_resource_attr_init_v2(a, sizeof(*a), 768); if (r && compute_resource_attr_init) return compute_resource_attr_init(a); return r; }
    return compute_resource_attr_init ? compute_resource_attr_init(a) : KNOTSUP;
}
static inline int HAP_compute_res_attr_set_vtcm_param(compute_res_attr_t *a, unsigned int n, unsigned char single) { return compute_resource_attr_set_vtcm_param ? compute_resource_attr_set_vtcm_param(a, n, single) : KNOTSUP; }
static inline int HAP_compute_res_attr_set_hmx_param(compute_res_attr_t *a, unsigned char on) { return compute_resource_attr_set_hmx_param ? compute_resource_attr_set_hmx_param(a, on) : KNOTSUP; }
static inline void *HAP_compute_res_attr_get_vtcm_ptr(compute_res_attr_t *a) { return compute_resource_attr_get_vtcm_ptr ? compute_resource_attr_get_vtcm_ptr(a) : 0; }
static inline int HAP_compute_res_query_VTCM(unsigned int app, unsigned int *t, compute_res_vtcm_page_t *tl, unsigned int *v, compute_res_vtcm_page_t *vl) { return compute_resource_query_VTCM ? compute_resource_query_VTCM(app, t, tl, v, vl) : KNOTSUP; }
static inline unsigned int HAP_compute_res_acquire(compute_res_attr_t *a, unsigned int us) { return compute_resource_acquire ? compute_resource_acquire(a, us) : 0; }
static inline int HAP_compute_res_release(unsigned int c) { return compute_resource_release ? compute_resource_release(c) : KNOTSUP; }
static inline int HAP_compute_res_query_capability(int id, unsigned int *d) { return crm_query_capability ? crm_query_capability(id, d) : KNOTSUP; }
static inline int HAP_compute_res_hmx_lock(unsigned int c) { return compute_resource_hmx_lock ? compute_resource_hmx_lock(c) : KNOTSUP; }
static inline int HAP_compute_res_hmx_unlock(unsigned int c) { return compute_resource_hmx_unlock ? compute_resource_hmx_unlock(c) : KNOTSUP; }
static inline int HAP_compute_res_hmx_lock4(unsigned int c, unsigned int **b, unsigned int us) { return crm_hmx_lock4 ? crm_hmx_lock4(c, b, us) : KNOTSUP; }
static inline int HAP_compute_res_hmx_unlock4(unsigned int c, unsigned int *b) { return crm_hmx_unlock4 ? crm_hmx_unlock4(c, b) : KNOTSUP; }

/* the threads */
#define QURT_EOK 0
typedef unsigned int qurt_thread_t;
typedef union { unsigned long long raw[2]; } qurt_mutex_t;
typedef union { unsigned long long raw[2]; } qurt_sem_t;
typedef struct { unsigned int max_hthreads; } qurt_sysenv_max_hthreads_t;
typedef struct {
    char name[16]; unsigned char tcb_partition, stid; unsigned short priority; unsigned char flags, bus_priority; unsigned short timetest_id;
    unsigned int stack_size; void *stack_addr; unsigned short detach_state;
} qurt_thread_attr_t;
_Static_assert(sizeof(qurt_thread_attr_t) == 36 && sizeof(qurt_mutex_t) == 16 && sizeof(qurt_sem_t) == 16, "the threads' structures");
static inline void qurt_thread_attr_init(qurt_thread_attr_t *a) { memset(a, 0, sizeof(*a)); a->priority = 254; a->bus_priority = 255; a->timetest_id = (unsigned short)-2; }
static inline void qurt_thread_attr_set_name(qurt_thread_attr_t *a, const char *n) { strncpy(a->name, n, sizeof(a->name) - 1); a->name[sizeof(a->name) - 1] = 0; }
static inline void qurt_thread_attr_set_priority(qurt_thread_attr_t *a, unsigned short p) { a->priority = p; }
static inline void qurt_thread_attr_set_stack_size(qurt_thread_attr_t *a, unsigned int n) { a->stack_size = n; }
static inline void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t *a, void *p) { a->stack_addr = p; }
int qurt_sysenv_get_max_hw_threads(qurt_sysenv_max_hthreads_t *m);
int qurt_futex_wait(void *lock, int val);
int qurt_futex_wake(void *lock, int n);
void qurt_mutex_init(qurt_mutex_t *m); void qurt_mutex_destroy(qurt_mutex_t *m); void qurt_mutex_lock(qurt_mutex_t *m); void qurt_mutex_unlock(qurt_mutex_t *m);
int qurt_sem_add(qurt_sem_t *s, unsigned int n); int qurt_sem_down(qurt_sem_t *s); void qurt_sem_destroy(qurt_sem_t *s); void qurt_sem_init_val(qurt_sem_t *s, unsigned short v);
static inline int qurt_sem_up(qurt_sem_t *s) { return qurt_sem_add(s, 1); }
int qurt_thread_create(qurt_thread_t *id, qurt_thread_attr_t *a, void (*entry)(void *), void *arg);
qurt_thread_t qurt_thread_get_id(void);
void qurt_thread_exit(int status);
int qurt_thread_join(unsigned int tid, int *status);
int qurt_thread_get_priority(qurt_thread_t id);
int qurt_hvx_get_units(void);
#endif
