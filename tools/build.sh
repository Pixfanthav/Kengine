#!/bin/bash
# build.sh: Kengine's Android libraries from this folder with the Android NDK and Zig alone -> out/
#   out/libkengine.so            the host library (JNI, arm64-v8a)
#   out/libkengine_skel_vNN.so   the DSP library for each Hexagon architecture (v73, v75, v79, v81)
#   out/libkblock.so             the calls' CPU side alone, for the YuE2 synthesizer
#   out/libkconv.so              the on-phone converter (a program, named .so so that Android installs it)
# Set NDK (the NDK's folder) and ZIG (the zig executable) first, e.g.
#   NDK=$HOME/Android/Sdk/ndk/28.2.13676358 ZIG=zig bash tools/build.sh
set -e
R=$(cd "$(dirname "$0")/.." && pwd); S=$R/src; O=$R/out; mkdir -p $O/obj
: "${NDK:?set NDK to the Android NDK folder}"; ZIG=${ZIG:-zig}; PY=${PY:-python3}
HOST=$(ls "$NDK/toolchains/llvm/prebuilt" | head -1); B=$NDK/toolchains/llvm/prebuilt/$HOST/bin; CC=$B/clang; [ -x "$CC" ] || CC=$B/clang.exe
T="--target=aarch64-none-linux-android26"

echo "== the CPU-DSP calls (src/kblock.h, kipc_stub.c, kipc_skel.c from src/kblock.idl)"
$PY "$R/tools/kipc_gen.py"

for A in ${ARCHS:-v73 v75 v79 v81}; do
  echo "== DSP library $A"
  C=$A; X=""; [ $A = v81 ] && { C=v79; X="-U__HVX_ARCH__ -D__HVX_ARCH__=81 -U__HEXAGON_ARCH__ -D__HEXAGON_ARCH__=81"; }   # Zig's newest Hexagon CPU is v79
  F="--target=hexagon-linux-musl -mcpu=hexagon$C+hvx$C+hvx_length128b+hvx_qfloat -O2 -G0 -fdata-sections -fno-zero-initialized-in-bss -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-memmove -fomit-frame-pointer -fno-unwind-tables -fno-asynchronous-unwind-tables -g0 -fPIC -fno-sanitize=all -fno-stack-protector -I$S $X"
  $ZIG cc $F -c $S/kblock_imp.c -o $O/obj/imp_$A.o
  $ZIG cc $F -c $S/kipc_skel.c -o $O/obj/ipc_$A.o
  $ZIG cc --target=hexagon-linux-musl -mcpu=hexagon$C -c $S/hmx_words.S -o $O/obj/hmx_$A.o
  $ZIG ld.lld -shared -s -soname libkblock_skel.so --allow-shlib-undefined --no-rosegment -z separate-loadable-segments -z norelro -z max-page-size=4096 -z common-page-size=4096 --no-eh-frame-hdr     $O/obj/imp_$A.o $O/obj/ipc_$A.o $O/obj/hmx_$A.o -o $O/libkengine_skel_$A.so
done

echo "== host library"
# libcdsprpc.so here only gives the linker the names: the phone's own library is the one that loads
echo 'void remote_handle64_open(void) {} void remote_handle64_invoke(void) {} void remote_handle64_close(void) {} void remote_session_control(void) {} void rpcmem_alloc(void) {} void rpcmem_free(void) {}' > $O/obj/stub.c
$CC $T -shared -nostdlib -Wl,-soname,libcdsprpc.so $O/obj/stub.c -o $O/obj/libcdsprpc.so
HF="-DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -D_FORTIFY_SOURCE=2 -O3 -DNDEBUG -fPIC -I$S -ffile-prefix-map=$R=. -w"
$CC $T $HF -c $S/kengine.c -o $O/obj/kengine.o
$CC $T $HF -c $S/kipc_stub.c -o $O/obj/kipc_stub.o
$CC $T -shared -nostdlib++ -s -Wl,-soname,libkengine.so -Wl,--no-rosegment -Wl,--no-undefined -Wl,-z,max-page-size=16384 -Wl,--gc-sections -Wl,--build-id=sha1   $O/obj/kengine.o $O/obj/kipc_stub.o -L$O/obj -lcdsprpc -llog -latomic -lm -ldl -o $O/libkengine.so

# libkblock.so: the CPU side of the calls alone, for the YuE2 synthesizer (yue2.cpp with Kroma's patch), which calls the DSP itself
$CC $T -shared -nostdlib++ -s -Wl,-soname,libkblock.so -Wl,--no-undefined -Wl,-z,max-page-size=16384 $O/obj/kipc_stub.o -L$O/obj -lcdsprpc -o $O/libkblock.so

echo "== converter"
$CC $T -O2 -fPIE -pie -s -w -ffile-prefix-map=$R=. $R/kconv/kconv.c -lm -o $O/libkconv.so

ls -la $O/*.so
