/* made by kipc_gen.py from src/kblock.idl */
#include "kblock.h"
#define KEXPORT __attribute__((visibility("default")))
/* the FastRPC loader looks these two up when it opens the library */
KEXPORT uint32_t kblock_skel_handle_invoke_qaic_version = 10051;
KEXPORT char kblock_skel_handle_invoke_uri[] = "file:///libkblock_skel.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0";
static int k_open(uint32_t sc, remote_arg *p)
{
    (void)sc; const char *uri = (const char *)p[1].buf.pv; const uint32_t len = *(const uint32_t *)p[0].buf.pv;
    if (!len || p[1].buf.nLen < len || uri[len - 1]) return KIPC_EBAD;
    return kblock_open(uri, &p[2].h64);
}
KEXPORT int kblock_skel_handle_invoke(remote_handle64 h, uint32_t sc, remote_arg *p)
{
    const uint32_t *w = (const uint32_t *)p[0].buf.pv; (void)w;
    switch ((sc >> 24) & 31)
    {
    case 0: return k_open(sc, p);
    case 1: return kblock_close(p[0].h64);
    case 2:
        if (sc != KIPC_SC(2, 4, 3) || p[0].buf.nLen < 44) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(float) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[10] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_run(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (const float *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (int)w[8], (int)w[9], (uint32_t *)p[6].buf.pv, (int)w[10]);
    case 3:
        if (sc != KIPC_SC(3, 3, 1) || p[0].buf.nLen < 16) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t)) return KIPC_EBAD;
        return kblock_embed(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (uint16_t *)p[3].buf.pv, (int)w[2], (int)w[3]);
    case 4:
        if (sc != KIPC_SC(4, 4, 1) || p[0].buf.nLen < 20) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(float) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t)) return KIPC_EBAD;
        return kblock_final(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (const float *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (int)w[4]);
    case 5:
        if (sc != KIPC_SC(5, 4, 3) || p[0].buf.nLen < 48) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(float) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[11] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_k2run(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (const float *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (int)w[8], (int)w[9], (int)w[10], (uint32_t *)p[6].buf.pv, (int)w[11]);
    case 6:
        if (sc != KIPC_SC(6, 1, 1) || p[0].buf.nLen < 8) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[1] * sizeof(uint16_t)) return KIPC_EBAD;
        return kblock_probe(h, (int)w[0], (uint16_t *)p[1].buf.pv, (int)w[1]);
    case 7:
        if (sc != KIPC_SC(7, 4, 3) || p[0].buf.nLen < 52) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(float) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[12] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_yrun(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (const float *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (int)w[8], (int)w[9], (int)w[10], (int)w[11], (uint32_t *)p[6].buf.pv, (int)w[12]);
    case 8:
        if (sc != KIPC_SC(8, 4, 3) || p[0].buf.nLen < 44) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[10] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_vdec(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (int)w[8], (int)w[9], (uint32_t *)p[6].buf.pv, (int)w[10]);
    case 9:
        if (sc != KIPC_SC(9, 4, 3) || p[0].buf.nLen < 32) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[7] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_adec(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (uint32_t *)p[6].buf.pv, (int)w[7]);
    case 10:
        if (sc != KIPC_SC(10, 5, 5) || p[0].buf.nLen < 52) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(float) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[5] * sizeof(uint16_t) || p[7].buf.nLen < (size_t)w[6] * sizeof(uint16_t) || p[8].buf.nLen < (size_t)w[7] * sizeof(uint16_t) || p[9].buf.nLen < (size_t)w[12] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_yrunl(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (const float *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (uint16_t *)p[6].buf.pv, (int)w[5], (uint16_t *)p[7].buf.pv, (int)w[6], (uint16_t *)p[8].buf.pv, (int)w[7], (int)w[8], (int)w[9], (int)w[10], (int)w[11], (uint32_t *)p[9].buf.pv, (int)w[12]);
    case 11:
        if (sc != KIPC_SC(11, 4, 3) || p[0].buf.nLen < 40) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[9] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_esr(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (int)w[8], (uint32_t *)p[6].buf.pv, (int)w[9]);
    case 12:
        if (sc != KIPC_SC(12, 4, 3) || p[0].buf.nLen < 36) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[8] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_qenc(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (uint32_t *)p[6].buf.pv, (int)w[8]);
    case 13:
        if (sc != KIPC_SC(13, 5, 2) || p[0].buf.nLen < 44) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[10] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_xop(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (const uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (int)w[8], (int)w[9], (uint32_t *)p[6].buf.pv, (int)w[10]);
    case 14:
        if (sc != KIPC_SC(14, 6, 6) || p[0].buf.nLen < 48) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint32_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[5] * sizeof(uint16_t) || p[7].buf.nLen < (size_t)w[6] * sizeof(uint16_t) || p[8].buf.nLen < (size_t)w[7] * sizeof(uint16_t) || p[9].buf.nLen < (size_t)w[8] * sizeof(uint16_t) || p[10].buf.nLen < (size_t)w[9] * sizeof(uint16_t) || p[11].buf.nLen < (size_t)w[10] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_xrun(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint32_t *)p[3].buf.pv, (int)w[2], (const uint16_t *)p[4].buf.pv, (int)w[3], (const uint16_t *)p[5].buf.pv, (int)w[4], (uint16_t *)p[6].buf.pv, (int)w[5], (uint16_t *)p[7].buf.pv, (int)w[6], (uint16_t *)p[8].buf.pv, (int)w[7], (uint16_t *)p[9].buf.pv, (int)w[8], (uint16_t *)p[10].buf.pv, (int)w[9], (uint32_t *)p[11].buf.pv, (int)w[10], (int)w[11]);
    case 15:
        if (sc != KIPC_SC(15, 4, 3) || p[0].buf.nLen < 36) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(float) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t) || p[5].buf.nLen < (size_t)w[4] * sizeof(uint16_t) || p[6].buf.nLen < (size_t)w[8] * sizeof(uint32_t)) return KIPC_EBAD;
        return kblock_xvae(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const float *)p[2].buf.pv, (int)w[1], (const uint16_t *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (uint16_t *)p[5].buf.pv, (int)w[4], (int)w[5], (int)w[6], (int)w[7], (uint32_t *)p[6].buf.pv, (int)w[8]);
    case 16:
        if (sc != KIPC_SC(16, 3, 2) || p[0].buf.nLen < 32) return KIPC_EBAD;
        if (p[1].buf.nLen < (size_t)w[0] * sizeof(uint16_t) || p[2].buf.nLen < (size_t)w[1] * sizeof(uint16_t) || p[3].buf.nLen < (size_t)w[2] * sizeof(uint16_t) || p[4].buf.nLen < (size_t)w[3] * sizeof(uint16_t)) return KIPC_EBAD;
        return kblock_lin(h, (const uint16_t *)p[1].buf.pv, (int)w[0], (const uint16_t *)p[2].buf.pv, (int)w[1], (uint16_t *)p[3].buf.pv, (int)w[2], (uint16_t *)p[4].buf.pv, (int)w[3], (int)w[4], (int)w[5], (int)w[6], (int)w[7]);
    }
    return KIPC_EUNSUPPORTED;
}
