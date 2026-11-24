/* hmx_words.S -- the routines of hmx_k2.S as instruction words (the assembler the engine is built with does not take the HMX
 * mnemonics). The HMX instructions are of the forms that llama.cpp's ggml-hexagon backend uses (ggml/src/ggml-hexagon/htp/hmx-utils.h,
 * MIT License). */
    .text
    .p2align 4
    .global hmx_mpy_chunks
    .type hmx_mpy_chunks, @function
hmx_mpy_chunks:
    .word 0x3c061c17
    .word 0x919dc048
    .word 0x6008c008
    .word 0x920041e2
    .word 0x9202e3ef
    .word 0x7f008000
    .word 0x38703872
    .word 0x7800c049
    .word 0x9206c3fe
    .word 0xa6e4e504
    .word 0x529fc000
    .word 0x7f00c000
    .global hmx_mpy_row
    .type hmx_mpy_row, @function
hmx_mpy_row:
    .word 0x78004049
    .word 0x7800400a
    .word 0x919d4006
    .word 0x919dc028
    .word 0x9206c3fe
    .word 0x6008c008
    .word 0x920041e2
    .word 0x9202e3ef
    .word 0xf3028502
    .word 0xb0844004
    .word 0xa6e4ea04
    .word 0x529fc000
    .global hmx_mpy_chunks_lora
    .type hmx_mpy_chunks_lora, @function
hmx_mpy_chunks_lora:
    .word 0x3c061c17
    .word 0x919d4048
    .word 0x919dc06a
    .word 0x919d408b
    .word 0x919dc0ae
    .word 0x60084018
    .word 0x78044f8c
    .word 0x7803ffed
    .word 0x920041e2
    .word 0x9202e3ef
    .word 0x7f008000
    .word 0x38703872
    .word 0x600ec008
    .word 0x920a4ce2
    .word 0x920bedef
    .word 0xb08a800a
    .word 0xb08bc00b
    .word 0x7800c049
    .word 0x9206c3fe
    .word 0xa6e4e504
    .word 0x529fc000
    .word 0x7f004000
    .word 0x7f004000
    .word 0x7f00c000
    .global hmx_mpy_group
    .type hmx_mpy_group, @function
hmx_mpy_group:
    .word 0x3c061c17
    .word 0x919d4048
    .word 0x919dc069
    .word 0x7800400b
    .word 0x919dc08a
    .word 0x9206c3fe
    .word 0x6029c008
    .word 0x60084018
    .word 0x7060400c
    .word 0x7062c00d
    .word 0x920c41e2
    .word 0x920de3ef
    .word 0xf30c870c
    .word 0xf30dc70d
    .word 0x7060400c
    .word 0xf3024a0d
    .word 0xa6e4eb04
    .word 0x6008c008
    .word 0x920c41e2
    .word 0x920de3ef
    .word 0xf30c870c
    .word 0xf30dc70d
    .word 0xb084c00e
    .word 0xf3004500
    .word 0xb1048004
    .word 0xa6eeeb04
    .word 0x529fc000
    .word 0x7f00c000
    .global hmx_mpy_chunks_lorad
    .type hmx_mpy_chunks_lorad, @function
hmx_mpy_chunks_lorad:
    .word 0x3c061c17
    .word 0x919d4048
    .word 0x919dc06a
    .word 0x919d408b
    .word 0x919dc0ae
    .word 0x60084010
    .word 0x8c0ecb4c
    .word 0x920041e2
    .word 0x9202e3ef
    .word 0x7f008000
    .word 0x38703872
    .word 0xbfec7fed
    .word 0x768ccf8c
    .word 0x920a4ce2
    .word 0x920bedef
    .word 0x9206c3fe
    .word 0xa6e4e504
    .word 0x529fc000
