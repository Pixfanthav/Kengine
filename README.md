**English** | [한국어](README.ko.md)

# Kengine

Kengine is the inference engine of the [Kroma](https://github.com/Pixfanthav/Kroma) Android app. It runs diffusion and text models on the Hexagon NPU of Snapdragon phones, in fp16 on the HMX matrix unit and the HVX vector units, with the weights read from the phone's storage block by block.

It is not a general graph runtime but an engine written for the models Kroma ships: each model's blocks, VAE and text encoder have their own code.

## Models

| Model | |
|---|---|
| Anima (Cosmos DiT) | the DiT blocks, input and output layers |
| Krea 2 | the DiT blocks, the text encoder's products |
| MOSS-Image | the DiT blocks, the text encoder's products |
| FLUX.2 [klein] 4B / 9B | the double and single blocks, the VAE encoder and decoder |
| SDXL | the UNet (runs of layers per call), the VAE decoder |
| Qwen-Image VAE | the encoder and the decoder |
| Real-ESRGAN (compact) | the x4 upscale |
| YuE2 | the NAR layers, the AR prefill and the audio decoder, called from yue2.cpp |

Weights are fp16. Q8 and Q4 block files can also be read: they are dequantized to fp16 on the HVX before the HMX products.

## Chips

The DSP library is built for Hexagon v73, v75, v79 and v81:

| Architecture | Snapdragon |
|---|---|
| v73 | 8 Gen 2 |
| v75 | 8 Gen 3 |
| v79 | 8 Elite |
| v81 | 8 Elite Gen 5 |

The engine runs as an unsigned DSP module through FastRPC and needs at least 8 MB of VTCM. On the 8s and 7+ series chips, included for trial, it asks for 4 MB.

## Layout

```
src/        the engine
  kengine.c        host side (JNI for dev.kroma.open.Engine): opening models, reading blocks, the step loops, LoRA
  lbake.inc        LoRAs baked into a copy of the weights
  lrun.inc         LoRAs applied while the blocks run
  qte.inc, kte.inc text encoders (MOSS-Image, Krea 2 / FLUX.2 klein)
  klein.inc        VAE decoder and encoder, Real-ESRGAN, large linear products
  kxl_host.inc     SDXL
  moss.inc         MOSS-Image's CPU parts
  kblock_imp.c     DSP side: products, attention, the blocks
  kvae.inc         DSP side: convolutions, VAE stages, Real-ESRGAN
  kxl.inc          DSP side: SDXL layers
  kaud.inc         DSP side: YuE2's audio decoder
  kpool.inc        DSP worker threads
  kdsp.h, kremote.h  the system interfaces the engine calls, by value
  hmx_k2.S         the HMX routines
  hmx_words.S      their encodings (what the build assembles)
  kblock.idl       the CPU-to-DSP calls
kconv/      the on-phone converter: a safetensors or GGUF checkpoint to the engine's files
tools/      build.sh, kipc_gen.py (the call stubs from kblock.idl)
yue2/       Kroma's patch to yue2.cpp, which calls the engine for YuE2 (yue2/README.md)
```

## Build

Requirements: the Android NDK (r28) and Zig (its clang targets Hexagon). Python 3 generates the call stubs.

```
NDK=/path/to/android-ndk ZIG=zig bash tools/build.sh
```

Output in `out/`:

- `libkengine.so`: the host library (arm64-v8a)
- `libkengine_skel_v73.so` … `_v81.so`: the DSP library for each architecture
- `libkblock.so`: the call stubs alone, for the YuE2 synthesizer
- `libkconv.so`: the converter (a program, named .so so that Android installs it)

The Hexagon SDK is not needed. Zig's newest Hexagon target is v79, so v81 is compiled as v79 with the architecture macros set to 81.

## Diagnostics

- An empty file named `kdebug` in a model's engine folder logs each block's largest value, inf / NaN count and mean (logcat tag `kengine`).

## Caveats

- The HMX routines are kept as instruction words (`hmx_words.S`): the assembler in the toolchain does not take the HMX mnemonics, so a new HMX routine cannot be built from source with these tools.
- The engine depends on interfaces of the phone's DSP system (power, compute resources, threads, unsigned modules) whose structures are not published by the vendor. A firmware change can stop it.
- The DSP address space is 32-bit: buffers and kept blocks are sized to stay under about 3.3 GB.
- There are no automated tests; changes have been checked on phones by comparing block outputs and pictures.

## License

GPL-3.0 (see `LICENSE`), with an additional permission for linking with the phone's DSP system libraries. Sources of the hardware details and third-party license texts are in `NOTICE`.

The model weights are not part of this repository and keep their own licenses.

Claude (Anthropic) was used during development.
