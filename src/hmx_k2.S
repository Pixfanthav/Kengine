/* hmx_k2.S -- the engine's fp16 HMX routines. The HMX instructions are of the forms that llama.cpp's ggml-hexagon backend uses
 * (ggml/src/ggml-hexagon/htp/hmx-utils.h, MIT License); hmx_words.S holds their encodings.
 *
 * hmx_mpy_chunks: one [32 x K] x [K x 32] fp16 HMX product with K split into chunks that are accumulated before the result is
 * converted, so that K is not limited by what a single activation/weight load can take (under 1024).
 *
 *   void hmx_mpy_chunks(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t out_ctrl,
 *                       uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks)
 *
 *   act, wgt:   the first chunk of the activation tile and of the weight sub-block in VTCM, as 32 x 32 fp16 blocks of 2 KB
 *               consecutive along K, so chunk c starts c * chunk_bytes further on
 *   act_ctrl, wgt_ctrl: the load range for one chunk ((blocks << 11) | 0x7C for the activation, (blocks << 11) - 1 for the weights)
 *   bias:       the v73+ bias/scale block (identity: scale 1.0, biases 0)
 *   out:        the [32 x 32] result, 2 KB, in the same interleaved layout
 */
.p2align 4,,15
.global hmx_mpy_chunks
.type hmx_mpy_chunks, @function
hmx_mpy_chunks:
{
    R6 = memw(R29+#0)                 /* bias */
    R7 = memw(R29+#4)                 /* chunk_bytes */
}
{
    R8 = memw(R29+#8)                 /* nchunks */
}
{
    loop0(.Lchunk, R8)
}
.Lchunk:
{
    activation.hf = mxmem(R0, R1):deep
    weight.hf = mxmem(R2, R3)
}
{
    R0 = add(R0, R7)
    R2 = add(R2, R7)
}:endloop0
{
    R9 = #2
}
{
    bias = mxmem2(R6)
}
{
    mxmem(R4, R5):after.hf = acc
}
{
    jumpr R31
}
.size hmx_mpy_chunks, .-hmx_mpy_chunks

/* hmx_mpy_row -- count products of one activation tile (one chunk) with weight blocks wgt + i * wgt_step, each converted and
 * stored at out + i * 2 KB: the q k^T scores of a query tile in one call.
 *
 *   void hmx_mpy_row(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t wgt_step,
 *                    uint8_t *bias, uint32_t count)
 */
.p2align 4,,15
.global hmx_mpy_row
.type hmx_mpy_row, @function
hmx_mpy_row:
{
    R6 = memw(R29+#0)                 /* bias */
    R8 = memw(R29+#4)                 /* count */
    R9 = #2
    R10 = #0
}
{
    bias = mxmem2(R6)
}
{
    loop0(.Lrow, R8)
}
.Lrow:
{
    activation.hf = mxmem(R0, R1):deep
    weight.hf = mxmem(R2, R3)
}
{
    mxmem(R4, R10):after.hf = acc
    R2 = add(R2, R5)
    R4 = add(R4, #2048)
}:endloop0
{
    jumpr R31
}
.size hmx_mpy_row, .-hmx_mpy_row

/* hmx_mpy_chunks_lora -- hmx_mpy_chunks with a LoRA's share accumulated before the one conversion: after W's chunks, (act la) lb's
 * lchunks single blocks (the [32 x LR] act la tile against lb's [LR x 32] half), so the share needs no output of its own, no second
 * conversion and no add afterwards (product_multi did those per group and tile, which cost 33 ms of a MOSS block with the DMD LoRA)
 *
 *   void hmx_mpy_chunks_lora(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t out_ctrl,
 *                            uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks, uint8_t *lact, uint8_t *lwgt, uint32_t lchunks)
 */
.p2align 4,,15
.global hmx_mpy_chunks_lora
.type hmx_mpy_chunks_lora, @function
hmx_mpy_chunks_lora:
{
    R6 = memw(R29+#0)                 /* bias */
    R7 = memw(R29+#4)                 /* chunk_bytes */
}
{
    R8 = memw(R29+#8)                 /* nchunks */
    R10 = memw(R29+#12)               /* lact */
}
{
    R11 = memw(R29+#16)               /* lwgt */
    R14 = memw(R29+#20)               /* lchunks */
}
{
    loop0(.Llchunk, R8)
    R12 = #0x087C                     /* one block a load: (1 << 11) | 0x7C */
    R13 = #0x07FF                     /* (1 << 11) - 1 */
}
.Llchunk:
{
    activation.hf = mxmem(R0, R1):deep
    weight.hf = mxmem(R2, R3)
}
{
    R0 = add(R0, R7)
    R2 = add(R2, R7)
}:endloop0
{
    loop0(.Llora, R14)
}
.Llora:
{
    activation.hf = mxmem(R10, R12):deep
    weight.hf = mxmem(R11, R13)
}
{
    R10 = add(R10, #2048)
    R11 = add(R11, #2048)
}:endloop0
{
    R9 = #2
}
{
    bias = mxmem2(R6)
}
{
    mxmem(R4, R5):after.hf = acc
}
{
    jumpr R31
}
.size hmx_mpy_chunks_lora, .-hmx_mpy_chunks_lora

/* hmx_mpy_group -- hmx_mpy_chunks for every tile of a batch and both 32-column halves of a weight group in one
 * call, the bias loaded once: tile j's activation at act + j * tile_bytes, its results at out + j * 4 KB (half h at + h * 2 KB), the
 * second half's weights at wgt + half. product_multi made 2 calls a tile, each with its stack arguments, bias load and return.
 *
 *   void hmx_mpy_group(uint8_t *act, uint32_t act_ctrl, uint8_t *wgt, uint32_t wgt_ctrl, uint8_t *out, uint32_t tile_bytes,
 *                      uint8_t *bias, uint32_t chunk_bytes, uint32_t nchunks, uint32_t tiles, uint32_t half)
 */
.p2align 4,,15
.global hmx_mpy_group
.type hmx_mpy_group, @function
hmx_mpy_group:
{
    R6 = memw(R29+#0)                 /* bias */
    R7 = memw(R29+#4)                 /* chunk_bytes */
}
{
    R8 = memw(R29+#8)                 /* nchunks */
    R9 = memw(R29+#12)                /* tiles */
}
{
    R10 = memw(R29+#16)               /* half */
    R11 = #0
}
{
    bias = mxmem2(R6)
}
{
    loop1(.Lgtile, R9)
}
.Lgtile:
{
    R12 = R0
    R13 = R2
    loop0(.Lgc0, R8)
}
.Lgc0:
{
    activation.hf = mxmem(R12, R1):deep
    weight.hf = mxmem(R13, R3)
}
{
    R12 = add(R12, R7)
    R13 = add(R13, R7)
}:endloop0
{
    mxmem(R4, R11):after.hf = acc
    R12 = R0
    R13 = add(R2, R10)
}
{
    loop0(.Lgc1, R8)
}
.Lgc1:
{
    activation.hf = mxmem(R12, R1):deep
    weight.hf = mxmem(R13, R3)
}
{
    R12 = add(R12, R7)
    R13 = add(R13, R7)
}:endloop0
{
    R14 = add(R4, #2048)
}
{
    mxmem(R14, R11):after.hf = acc
    R0 = add(R0, R5)
    R4 = add(R4, #4096)
}:endloop1
{
    jumpr R31
}
.size hmx_mpy_group, .-hmx_mpy_group

/* hmx_mpy_chunks_lorad -- hmx_mpy_chunks_lora with the LoRA's lchunks blocks in one deep load each side instead of one block a
 * load (a 1-block load costs 1.6x a 4-block one). lchunks < 32. Not on v81: a deep load of LR / 32 blocks came out wrong there
 * -- the caller keeps hmx_mpy_chunks_lora there. Same arguments as hmx_mpy_chunks_lora. */
.p2align 4,,15
.global hmx_mpy_chunks_lorad
.type hmx_mpy_chunks_lorad, @function
hmx_mpy_chunks_lorad:
{
    R6 = memw(R29+#0)                 /* bias */
    R7 = memw(R29+#4)                 /* chunk_bytes */
}
{
    R8 = memw(R29+#8)                 /* nchunks */
    R10 = memw(R29+#12)               /* lact */
}
{
    R11 = memw(R29+#16)               /* lwgt */
    R14 = memw(R29+#20)               /* lchunks */
}
{
    loop0(.Ldchunk, R8)
    R12 = asl(R14, #11)
}
.Ldchunk:
{
    activation.hf = mxmem(R0, R1):deep
    weight.hf = mxmem(R2, R3)
}
{
    R0 = add(R0, R7)
    R2 = add(R2, R7)
}:endloop0
{
    R13 = add(R12, #-1)
    R12 = or(R12, #0x7C)
}
{
    activation.hf = mxmem(R10, R12):deep
    weight.hf = mxmem(R11, R13)
}
{
    bias = mxmem2(R6)
}
{
    mxmem(R4, R5):after.hf = acc
}
{
    jumpr R31
}
.size hmx_mpy_chunks_lorad, .-hmx_mpy_chunks_lorad
