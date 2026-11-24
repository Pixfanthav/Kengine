/* kremote.h: what the CPU-DSP calls need, by value: the argument of a call, and the system library's functions (libcdsprpc.so, on the phone).
 * Where the values come from: written by hand, no SDK file copied. remote_arg, the call number
 * (method << 24 | inputs << 16 | outputs << 8), remote_handle64_open / invoke / close, the session control (DSPRPC_CONTROL_UNSIGNED_MODULE),
 * the rpcmem heap and flags, the buffer attributes and the error codes are those of Qualcomm's open FastRPC library, qualcomm/fastrpc
 * (inc/remote.h, inc/rpcmem.h, inc/AEEStdErr.h; BSD-3-Clause, Copyright (c) Qualcomm Innovation Center, Inc.). */
#ifndef KREMOTE_H
#define KREMOTE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef uint64_t remote_handle64;
typedef struct { void *pv; size_t nLen; } kremote_buf;
typedef union { kremote_buf buf; uint32_t h; remote_handle64 h64; } remote_arg;
/* method, input buffers, output buffers */
#define KIPC_SC(m, i, o) (((uint32_t)(m) << 24) | ((uint32_t)(i) << 16) | ((uint32_t)(o) << 8))
#define KIPC_EBAD ((int)0x8000040E)
#define KIPC_EUNSUPPORTED ((int)0x80000414)
#ifndef __hexagon__
#define AEE_EFAILED ((int)0x80000401)
#define AEE_ENOMEMORY ((int)0x80000402)
#define AEE_EBADPARM ((int)0x8000040E)
#endif
#define CDSP_DOMAIN_ID 3
#define CDSP_DOMAIN "&_dom=cdsp"
#define DSPRPC_CONTROL_UNSIGNED_MODULE 2
struct remote_rpc_control_unsigned_module { int domain; int enable; };
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
/* remote_register_buf_attr's attribute bits (qualcomm/fastrpc, inc/remote.h: FASTRPC_ATTR_KEEP_MAP, FASTRPC_ATTR_TRY_MAP_STATIC) */
#define KIPC_ATTR_KEEP_MAP 8
#define KIPC_ATTR_TRY_MAP_STATIC 128
#ifndef __hexagon__
int remote_handle64_open(const char *name, remote_handle64 *ph);
int remote_handle64_invoke(remote_handle64 h, uint32_t scalars, remote_arg *pra);
int remote_handle64_close(remote_handle64 h);
int remote_session_control(uint32_t req, void *data, uint32_t datalen);
void *rpcmem_alloc(int heapid, uint32_t flags, int size);
void rpcmem_free(void *po);
#endif
#ifdef __cplusplus
}
#endif
#endif
