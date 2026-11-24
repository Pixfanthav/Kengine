# kipc_gen.py: the CPU-DSP calls from src/kblock.idl: src/kblock.h, src/kipc_stub.c, src/kipc_skel.c.
# A call: argument 0 holds the buffers' lengths and the integers, as 32-bit words; then the input buffers, then the output buffers.
import re, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); S = H + "/src/"
t = re.sub(r"//.*", "", open(S + "kblock.idl", encoding="utf-8").read())
body = t[t.index("{") + 1:t.rindex("}")]
ty = {"uint16": "uint16_t", "float": "float", "uint32": "uint32_t"}
fns = []
for name, args in re.findall(r"long\s+(\w+)\s*\(([^)]*)\)\s*;", body):
    a = []
    for p in args.split(","):
        p = p.split()
        if p[1].startswith("sequence"): a.append((p[0], ty[re.search(r"<(\w+)>", p[1]).group(1)], p[2]))
        else: a.append(("int", "int", p[2]))
    fns.append((name, a))
def proto(n, a):
    s = ["remote_handle64 k_h"]
    for k, c, v in a:
        if k == "int": s.append("int " + v)
        else: s.append(("const " if k == "in" else "") + c + " *" + v + ", int " + v + "Len")
    return "int kblock_%s(%s)" % (n, ", ".join(s))
with open(S + "kblock.h", "w", newline="\n") as f:
    f.write("/* made by kipc_gen.py from src/kblock.idl */\n#ifndef KBLOCK_H\n#define KBLOCK_H\n#include \"kremote.h\"\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n")
    f.write('#define kblock_URI "file:///libkblock_skel.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0"\n')
    f.write("int kblock_open(const char *uri, remote_handle64 *h);\nint kblock_close(remote_handle64 h);\n")
    for n, a in fns: f.write(proto(n, a) + ";\n")
    f.write("#ifdef __cplusplus\n}\n#endif\n#endif\n")
with open(S + "kipc_stub.c", "w", newline="\n") as f:
    f.write('/* made by kipc_gen.py from src/kblock.idl */\n#include "kblock.h"\n')
    f.write("int kblock_open(const char *uri, remote_handle64 *h) { return remote_handle64_open(uri, h); }\nint kblock_close(remote_handle64 h) { return remote_handle64_close(h); }\n")
    for m, (n, a) in enumerate(fns, 2):
        ins = [v for k, c, v in a if k == "in"]; outs = [v for k, c, v in a if k == "rout"]; words = []
        for k, c, v in a: words.append("(uint32_t)" + v + ("" if k == "int" else "Len"))
        ca = dict((v, c) for k, c, v in a)
        f.write(proto(n, a) + "\n{\n    uint32_t k_w[] = { %s };\n    remote_arg k_p[%d];\n    k_p[0].buf.pv = k_w; k_p[0].buf.nLen = sizeof(k_w);\n" % (", ".join(words), 1 + len(ins) + len(outs)))
        for i, v in enumerate(ins + outs, 1): f.write("    k_p[%d].buf.pv = (void *)%s; k_p[%d].buf.nLen = (size_t)%sLen * sizeof(%s);\n" % (i, v, i, v, ca[v]))
        f.write("    return remote_handle64_invoke(k_h, KIPC_SC(%d, %d, %d), k_p);\n}\n" % (m, 1 + len(ins), len(outs)))
with open(S + "kipc_skel.c", "w", newline="\n") as f:
    f.write('/* made by kipc_gen.py from src/kblock.idl */\n#include "kblock.h"\n')
    f.write('#define KEXPORT __attribute__((visibility("default")))\n')
    f.write('/* the FastRPC loader looks these two up when it opens the library */\nKEXPORT uint32_t kblock_skel_handle_invoke_qaic_version = 10051;\nKEXPORT char kblock_skel_handle_invoke_uri[] = "file:///libkblock_skel.so?kblock_skel_handle_invoke&_modver=1.0&_idlver=1.0.0";\n')
    f.write("static int k_open(uint32_t sc, remote_arg *p)\n{\n    (void)sc; const char *uri = (const char *)p[1].buf.pv; const uint32_t len = *(const uint32_t *)p[0].buf.pv;\n    if (!len || p[1].buf.nLen < len || uri[len - 1]) return KIPC_EBAD;\n    return kblock_open(uri, &p[2].h64);\n}\n")
    f.write("KEXPORT int kblock_skel_handle_invoke(remote_handle64 h, uint32_t sc, remote_arg *p)\n{\n    const uint32_t *w = (const uint32_t *)p[0].buf.pv; (void)w;\n    switch ((sc >> 24) & 31)\n    {\n    case 0: return k_open(sc, p);\n    case 1: return kblock_close(p[0].h64);\n")
    for m, (n, a) in enumerate(fns, 2):
        ins = [v for k, c, v in a if k == "in"]; outs = [v for k, c, v in a if k == "rout"]; nin, nout = 1 + len(ins), len(outs)
        pos = {v: i for i, v in enumerate(ins + outs, 1)}; ca = []; words = len(a)
        for i, (k, c, v) in enumerate(a):
            if k == "int": ca.append("(int)w[%d]" % i)
            else: ca.append("(%s%s *)p[%d].buf.pv, (int)w[%d]" % ("const " if k == "in" else "", c, pos[v], i))
        chk = " || ".join("p[%d].buf.nLen < (size_t)w[%d] * sizeof(%s)" % (pos[v], i, c) for i, (k, c, v) in enumerate(a) if k != "int")
        f.write("    case %d:\n        if (sc != KIPC_SC(%d, %d, %d) || p[0].buf.nLen < %d) return KIPC_EBAD;\n" % (m, m, nin, nout, words * 4))
        if chk: f.write("        if (%s) return KIPC_EBAD;\n" % chk)
        f.write("        return kblock_%s(h, %s);\n" % (n, ", ".join(ca)))
    f.write("    }\n    return KIPC_EUNSUPPORTED;\n}\n")
print(len(fns), "calls")
