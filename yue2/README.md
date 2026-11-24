**English** | [한국어](README.ko.md)

# YuE2 on Kengine

`yue2-kroma.patch` is Kroma's change to [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) (MIT License), against commit `83b65f5`. It moves three parts of a song's generation onto the phone's DSP through Kengine, and leaves everything else to ggml on the CPU:

- the NAR (flow matching) layers: `src/kroma_nar.h`, on when `YUE_ENGINE` names a folder with the layers' engine files
- the AR prompt prefill: `src/kroma_ar.h`, on when `YUE_ENGINE_AR` is set
- the audio decoder: `src/kroma_vae.h`, on when `YUE_ENGINE_VAE` is set

`KROMA_SKEL` names the DSP library to load (Kroma passes `libkengine_skel_<arch>.so`). If a DSP call fails, that part falls back to ggml.

The patch also adds a length target with a soft end (`KROMA_YUE_MIN`: no end token before that many seconds), a fade-out when a song is cut by the length budget, and CPU flash attention when `YUE_CPU_FA=1`.

Build: apply the patch, put Kengine's `src/kblock.h` and `src/kremote.h` in `../kroma_inc`, and configure with `-DKROMA_ENGINE_DIR=<folder with Kengine's libkblock.so and a libcdsprpc.so to link against>` (Kengine's `tools/build.sh` leaves both in `out/` and `out/obj/`).

```
git clone https://github.com/ServeurpersoCom/yue2.cpp && cd yue2.cpp && git checkout 83b65f5
git apply ../Kengine/yue2/yue2-kroma.patch
```

The engine files for the layers are made by Kroma's converter tools, which are not in this repository.
