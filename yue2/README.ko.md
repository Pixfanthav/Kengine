[English](README.md) | **한국어**

# Kengine에서의 YuE2

`yue2-kroma.patch`는 Kroma가 [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp)(MIT License)를 고친 패치로, 커밋 `83b65f5`를 기준으로 합니다. 곡을 생성하는 과정 가운데 다음 세 부분을 Kengine을 거쳐 폰의 DSP로 옮기고, 나머지는 전부 CPU의 ggml에 맡깁니다.

* NAR(flow matching) 층: `src/kroma_nar.h`. `YUE_ENGINE`에 이 층의 엔진 파일이 든 폴더를 지정하면 켜집니다.
* AR 프롬프트 프리필: `src/kroma_ar.h`. `YUE_ENGINE_AR`을 설정하면 켜집니다.
* 오디오 디코더: `src/kroma_vae.h`. `YUE_ENGINE_VAE`를 설정하면 켜집니다.

`KROMA_SKEL`에는 불러올 DSP 라이브러리 이름을 넣습니다(Kroma는 `libkengine_skel_<arch>.so`를 넘깁니다). DSP 호출이 실패하면 그 부분은 ggml이 대신 계산합니다.

패치에는 다음 기능도 들어 있습니다.

* 끝을 못 박지 않는 길이 목표(`KROMA_YUE_MIN`: 이 초 수가 지나기 전에는 끝 토큰을 내지 않음)
* 길이 한도에 걸려 곡이 잘릴 때 넣는 페이드아웃
* `YUE_CPU_FA=1`일 때 쓰는 CPU 플래시 어텐션

빌드하려면 패치를 적용하고 Kengine의 `src/kblock.h`와 `src/kremote.h`를 `../kroma_inc`에 넣은 다음, `-DKROMA_ENGINE_DIR=<Kengine의 libkblock.so와 링크용 libcdsprpc.so가 든 폴더>`를 주고 구성합니다. 두 파일은 Kengine의 `tools/build.sh`를 돌리면 `out/`과 `out/obj/`에 생깁니다.

```
git clone https://github.com/ServeurpersoCom/yue2.cpp && cd yue2.cpp && git checkout 83b65f5
git apply ../Kengine/yue2/yue2-kroma.patch
```

층 엔진 파일은 Kroma의 변환 도구로 만들며, 이 도구는 저장소에 들어 있지 않습니다.
