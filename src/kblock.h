/* made by kipc_gen.py from src/kblock.idl */
#ifndef KBLOCK_H
#define KBLOCK_H
#include "kremote.h"
#ifdef __cplusplus
extern "C" {
#endif
#define kblock_URI "file:///libkblock_skel.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0"
int kblock_open(const char *uri, remote_handle64 *h);
int kblock_close(remote_handle64 h);
int kblock_run(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int ctxlen, int ffn, int kvalid, int inplace, uint32_t *times, int timesLen);
int kblock_embed(remote_handle64 k_h, const uint16_t *pat, int patLen, const uint16_t *w, int wLen, uint16_t *xo, int xoLen, int tokens);
int kblock_final(remote_handle64 k_h, const uint16_t *x, int xLen, const uint16_t *w, int wLen, const float *fmod, int fmodLen, uint16_t *res, int resLen, int tokens);
int kblock_k2run(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int ka, int kb, int ka2, int kb2, int inplace, uint32_t *times, int timesLen);
int kblock_probe(remote_handle64 k_h, int mode, uint16_t *res, int resLen);
int kblock_yrun(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int prefix, int ka, int kb, int ka2, int kb2, int inplace, uint32_t *times, int timesLen);
int kblock_vdec(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int vw, int start, int stop, uint32_t *times, int timesLen);
int kblock_adec(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int t0, int stop, uint32_t *times, int timesLen);
int kblock_yrunl(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *rope, int ropeLen, const uint16_t *pre, int preLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, uint16_t *kvo, int kvoLen, uint16_t *pre2, int pre2Len, int tokens, int prefix, int nl, int inplace, uint32_t *times, int timesLen);
int kblock_esr(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int vw, int n, uint32_t *times, int timesLen);
int kblock_qenc(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int vw, uint32_t *times, int timesLen);
int kblock_xop(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, const uint16_t *x2, int x2Len, uint16_t *y, int yLen, int op, int h, int wd, int ci, int co, uint32_t *times, int timesLen);
int kblock_xrun(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint32_t *tab, int tabLen, const uint16_t *ctx, int ctxLen, const uint16_t *zin, int zinLen, uint16_t *xa, int xaLen, uint16_t *xb, int xbLen, uint16_t *st0, int st0Len, uint16_t *sk, int skLen, uint16_t *out, int outLen, uint32_t *times, int timesLen, int fmt);
int kblock_xvae(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int stop, uint32_t *times, int timesLen);
int kblock_lin(remote_handle64 k_h, const uint16_t *w, int wLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int m, int k, int n, int flags);
#ifdef __cplusplus
}
#endif
#endif
