/* made by kipc_gen.py from src/kblock.idl */
#include "kblock.h"
int kblock_open(const char *uri, remote_handle64 *h) { return remote_handle64_open(uri, h); }
int kblock_close(remote_handle64 h) { return remote_handle64_close(h); }
int kblock_run(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int ctxlen, int ffn, int kvalid, int inplace, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wtsLen, (uint32_t)inpLen, (uint32_t)prmLen, (uint32_t)scrLen, (uint32_t)xoLen, (uint32_t)tokens, (uint32_t)ctxlen, (uint32_t)ffn, (uint32_t)kvalid, (uint32_t)inplace, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)wts; k_p[1].buf.nLen = (size_t)wtsLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)inp; k_p[2].buf.nLen = (size_t)inpLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)prm; k_p[3].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)xo; k_p[5].buf.nLen = (size_t)xoLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(2, 4, 3), k_p);
}
int kblock_embed(remote_handle64 k_h, const uint16_t *pat, int patLen, const uint16_t *w, int wLen, uint16_t *xo, int xoLen, int tokens)
{
    uint32_t k_w[] = { (uint32_t)patLen, (uint32_t)wLen, (uint32_t)xoLen, (uint32_t)tokens };
    remote_arg k_p[4];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)pat; k_p[1].buf.nLen = (size_t)patLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)w; k_p[2].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)xo; k_p[3].buf.nLen = (size_t)xoLen * sizeof(uint16_t);
    return remote_handle64_invoke(k_h, KIPC_SC(3, 3, 1), k_p);
}
int kblock_final(remote_handle64 k_h, const uint16_t *x, int xLen, const uint16_t *w, int wLen, const float *fmod, int fmodLen, uint16_t *res, int resLen, int tokens)
{
    uint32_t k_w[] = { (uint32_t)xLen, (uint32_t)wLen, (uint32_t)fmodLen, (uint32_t)resLen, (uint32_t)tokens };
    remote_arg k_p[5];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)x; k_p[1].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)w; k_p[2].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)fmod; k_p[3].buf.nLen = (size_t)fmodLen * sizeof(float);
    k_p[4].buf.pv = (void *)res; k_p[4].buf.nLen = (size_t)resLen * sizeof(uint16_t);
    return remote_handle64_invoke(k_h, KIPC_SC(4, 4, 1), k_p);
}
int kblock_k2run(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int ka, int kb, int ka2, int kb2, int inplace, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wtsLen, (uint32_t)inpLen, (uint32_t)prmLen, (uint32_t)scrLen, (uint32_t)xoLen, (uint32_t)tokens, (uint32_t)ka, (uint32_t)kb, (uint32_t)ka2, (uint32_t)kb2, (uint32_t)inplace, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)wts; k_p[1].buf.nLen = (size_t)wtsLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)inp; k_p[2].buf.nLen = (size_t)inpLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)prm; k_p[3].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)xo; k_p[5].buf.nLen = (size_t)xoLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(5, 4, 3), k_p);
}
int kblock_probe(remote_handle64 k_h, int mode, uint16_t *res, int resLen)
{
    uint32_t k_w[] = { (uint32_t)mode, (uint32_t)resLen };
    remote_arg k_p[2];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)res; k_p[1].buf.nLen = (size_t)resLen * sizeof(uint16_t);
    return remote_handle64_invoke(k_h, KIPC_SC(6, 1, 1), k_p);
}
int kblock_yrun(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *inp, int inpLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, int tokens, int prefix, int ka, int kb, int ka2, int kb2, int inplace, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wtsLen, (uint32_t)inpLen, (uint32_t)prmLen, (uint32_t)scrLen, (uint32_t)xoLen, (uint32_t)tokens, (uint32_t)prefix, (uint32_t)ka, (uint32_t)kb, (uint32_t)ka2, (uint32_t)kb2, (uint32_t)inplace, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)wts; k_p[1].buf.nLen = (size_t)wtsLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)inp; k_p[2].buf.nLen = (size_t)inpLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)prm; k_p[3].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)xo; k_p[5].buf.nLen = (size_t)xoLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(7, 4, 3), k_p);
}
int kblock_vdec(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int vw, int start, int stop, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)xLen, (uint32_t)scrLen, (uint32_t)yLen, (uint32_t)h, (uint32_t)wd, (uint32_t)vw, (uint32_t)start, (uint32_t)stop, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)x; k_p[3].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)y; k_p[5].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(8, 4, 3), k_p);
}
int kblock_adec(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int t0, int stop, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)xLen, (uint32_t)scrLen, (uint32_t)yLen, (uint32_t)t0, (uint32_t)stop, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)x; k_p[3].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)y; k_p[5].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(9, 4, 3), k_p);
}
int kblock_yrunl(remote_handle64 k_h, const uint16_t *wts, int wtsLen, const uint16_t *rope, int ropeLen, const uint16_t *pre, int preLen, const float *prm, int prmLen, uint16_t *scr, int scrLen, uint16_t *xo, int xoLen, uint16_t *kvo, int kvoLen, uint16_t *pre2, int pre2Len, int tokens, int prefix, int nl, int inplace, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wtsLen, (uint32_t)ropeLen, (uint32_t)preLen, (uint32_t)prmLen, (uint32_t)scrLen, (uint32_t)xoLen, (uint32_t)kvoLen, (uint32_t)pre2Len, (uint32_t)tokens, (uint32_t)prefix, (uint32_t)nl, (uint32_t)inplace, (uint32_t)timesLen };
    remote_arg k_p[10];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)wts; k_p[1].buf.nLen = (size_t)wtsLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)rope; k_p[2].buf.nLen = (size_t)ropeLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)pre; k_p[3].buf.nLen = (size_t)preLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)prm; k_p[4].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[5].buf.pv = (void *)scr; k_p[5].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)xo; k_p[6].buf.nLen = (size_t)xoLen * sizeof(uint16_t);
    k_p[7].buf.pv = (void *)kvo; k_p[7].buf.nLen = (size_t)kvoLen * sizeof(uint16_t);
    k_p[8].buf.pv = (void *)pre2; k_p[8].buf.nLen = (size_t)pre2Len * sizeof(uint16_t);
    k_p[9].buf.pv = (void *)times; k_p[9].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(10, 5, 5), k_p);
}
int kblock_esr(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int vw, int n, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)xLen, (uint32_t)scrLen, (uint32_t)yLen, (uint32_t)h, (uint32_t)wd, (uint32_t)vw, (uint32_t)n, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)x; k_p[3].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)y; k_p[5].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(11, 4, 3), k_p);
}
int kblock_qenc(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int vw, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)xLen, (uint32_t)scrLen, (uint32_t)yLen, (uint32_t)h, (uint32_t)wd, (uint32_t)vw, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)x; k_p[3].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)y; k_p[5].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(12, 4, 3), k_p);
}
int kblock_xop(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, const uint16_t *x2, int x2Len, uint16_t *y, int yLen, int op, int h, int wd, int ci, int co, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)xLen, (uint32_t)x2Len, (uint32_t)yLen, (uint32_t)op, (uint32_t)h, (uint32_t)wd, (uint32_t)ci, (uint32_t)co, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)x; k_p[3].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)x2; k_p[4].buf.nLen = (size_t)x2Len * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)y; k_p[5].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(13, 5, 2), k_p);
}
int kblock_xrun(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint32_t *tab, int tabLen, const uint16_t *ctx, int ctxLen, const uint16_t *zin, int zinLen, uint16_t *xa, int xaLen, uint16_t *xb, int xbLen, uint16_t *st0, int st0Len, uint16_t *sk, int skLen, uint16_t *out, int outLen, uint32_t *times, int timesLen, int fmt)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)tabLen, (uint32_t)ctxLen, (uint32_t)zinLen, (uint32_t)xaLen, (uint32_t)xbLen, (uint32_t)st0Len, (uint32_t)skLen, (uint32_t)outLen, (uint32_t)timesLen, (uint32_t)fmt };
    remote_arg k_p[12];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)tab; k_p[3].buf.nLen = (size_t)tabLen * sizeof(uint32_t);
    k_p[4].buf.pv = (void *)ctx; k_p[4].buf.nLen = (size_t)ctxLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)zin; k_p[5].buf.nLen = (size_t)zinLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)xa; k_p[6].buf.nLen = (size_t)xaLen * sizeof(uint16_t);
    k_p[7].buf.pv = (void *)xb; k_p[7].buf.nLen = (size_t)xbLen * sizeof(uint16_t);
    k_p[8].buf.pv = (void *)st0; k_p[8].buf.nLen = (size_t)st0Len * sizeof(uint16_t);
    k_p[9].buf.pv = (void *)sk; k_p[9].buf.nLen = (size_t)skLen * sizeof(uint16_t);
    k_p[10].buf.pv = (void *)out; k_p[10].buf.nLen = (size_t)outLen * sizeof(uint16_t);
    k_p[11].buf.pv = (void *)times; k_p[11].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(14, 6, 6), k_p);
}
int kblock_xvae(remote_handle64 k_h, const uint16_t *w, int wLen, const float *prm, int prmLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int h, int wd, int stop, uint32_t *times, int timesLen)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)prmLen, (uint32_t)xLen, (uint32_t)scrLen, (uint32_t)yLen, (uint32_t)h, (uint32_t)wd, (uint32_t)stop, (uint32_t)timesLen };
    remote_arg k_p[7];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)prm; k_p[2].buf.nLen = (size_t)prmLen * sizeof(float);
    k_p[3].buf.pv = (void *)x; k_p[3].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)scr; k_p[4].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[5].buf.pv = (void *)y; k_p[5].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    k_p[6].buf.pv = (void *)times; k_p[6].buf.nLen = (size_t)timesLen * sizeof(uint32_t);
    return remote_handle64_invoke(k_h, KIPC_SC(15, 4, 3), k_p);
}
int kblock_lin(remote_handle64 k_h, const uint16_t *w, int wLen, const uint16_t *x, int xLen, uint16_t *scr, int scrLen, uint16_t *y, int yLen, int m, int k, int n, int flags)
{
    uint32_t k_w[] = { (uint32_t)wLen, (uint32_t)xLen, (uint32_t)scrLen, (uint32_t)yLen, (uint32_t)m, (uint32_t)k, (uint32_t)n, (uint32_t)flags };
    remote_arg k_p[5];
    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);
    k_p[1].buf.pv = (void *)w; k_p[1].buf.nLen = (size_t)wLen * sizeof(uint16_t);
    k_p[2].buf.pv = (void *)x; k_p[2].buf.nLen = (size_t)xLen * sizeof(uint16_t);
    k_p[3].buf.pv = (void *)scr; k_p[3].buf.nLen = (size_t)scrLen * sizeof(uint16_t);
    k_p[4].buf.pv = (void *)y; k_p[4].buf.nLen = (size_t)yLen * sizeof(uint16_t);
    return remote_handle64_invoke(k_h, KIPC_SC(16, 3, 2), k_p);
}
