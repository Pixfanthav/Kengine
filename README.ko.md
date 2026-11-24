[English](README.md) | **한국어**

# Kengine

Kengine은 [Kroma](https://github.com/Pixfanthav/Kroma) 안드로이드 앱의 추론 엔진입니다. Snapdragon 폰의 Hexagon NPU에서 확산 모델과 텍스트 모델을 돌립니다. HMX 행렬 유닛과 HVX 벡터 유닛에서 fp16으로 계산하고, 가중치는 폰 저장공간에서 블록 단위로 읽습니다.

범용 그래프 실행기가 아니라 Kroma가 제공하는 모델에 맞춰 짠 엔진으로, 모델마다 블록, VAE, 텍스트 인코더 코드가 따로 있습니다.

## 모델

|모델| |
|---|---|
|Anima (Cosmos DiT)|DiT 블록, 입력·출력 층|
|Krea 2|DiT 블록, 텍스트 인코더의 행렬곱|
|MOSS-Image|DiT 블록, 텍스트 인코더의 행렬곱|
|FLUX.2 \[klein\] 4B / 9B|이중·단일 블록, VAE 인코더와 디코더|
|SDXL|UNet(호출마다 층 묶음), VAE 디코더|
|Qwen-Image VAE|인코더와 디코더|
|Real-ESRGAN (compact)|4배 업스케일|
|YuE2|NAR 층, AR 프리필, 오디오 디코더 (yue2.cpp에서 호출)|

가중치는 fp16입니다. Q8·Q4 블록 파일도 읽을 수 있으며, HMX 행렬곱 전에 HVX에서 fp16으로 역양자화하여 사용합니다.

## 칩

DSP 라이브러리는 Hexagon v73, v75, v79, v81용으로 빌드합니다.

|아키텍처|Snapdragon|
|---|---|
|v73|8 Gen 2|
|v75|8 Gen 3|
|v79|8 Elite|
|v81|8 Elite Gen 5|

엔진은 FastRPC를 통해 서명 없는 DSP 모듈로 실행되며 VTCM이 8 MB 이상 필요합니다. 시험용으로 포함된 8s 계열과 7+ 계열에서는 4 MB를 요청합니다.

## 구성

```
src/        엔진
  kengine.c        호스트 쪽(dev.kroma.open.Engine의 JNI): 모델 열기, 블록 읽기, 스텝 루프, LoRA
  lbake.inc        가중치 사본에 LoRA 굽기
  lrun.inc         블록 실행 중 LoRA 적용
  qte.inc, kte.inc 텍스트 인코더 (MOSS-Image, Krea 2 / FLUX.2 klein)
  klein.inc        VAE 디코더·인코더, Real-ESRGAN, 큰 선형 곱
  kxl_host.inc     SDXL
  moss.inc         MOSS-Image의 CPU 부분
  kblock_imp.c     DSP 쪽: 행렬곱, 어텐션, 블록
  kvae.inc         DSP 쪽: 합성곱, VAE 단계, Real-ESRGAN
  kxl.inc          DSP 쪽: SDXL 층
  kaud.inc         DSP 쪽: YuE2 오디오 디코더
  kpool.inc        DSP 작업 스레드
  kdsp.h, kremote.h  엔진이 부르는 시스템 인터페이스의 값
  hmx_k2.S         HMX 루틴
  hmx_words.S      그 루틴의 명령어 인코딩 (빌드가 어셈블하는 파일)
  kblock.idl       CPU에서 DSP로 가는 호출
kconv/      폰에서 도는 변환기: safetensors나 GGUF 체크포인트를 엔진 파일로
tools/      build.sh, kipc_gen.py (kblock.idl에서 호출 코드 생성)
yue2/       YuE2용 yue2.cpp 패치, 엔진을 호출하는 부분 (yue2/README.ko.md)
```

## 빌드

Android NDK(r28)와 Zig(clang이 Hexagon을 지원)가 필요하며 호출 코드 생성에 Python 3을 사용합니다.

```
NDK=/path/to/android-ndk ZIG=zig bash tools/build.sh
```

`out/`에 생기는 파일:

* `libkengine.so`: 호스트 라이브러리 (arm64-v8a)
* `libkengine_skel_v73.so` … `_v81.so`: 아키텍처별 DSP 라이브러리
* `libkblock.so`: YuE2 합성기용 호출 코드
* `libkconv.so`: 변환기 (안드로이드가 설치하도록 .so 이름을 붙인 실행 파일)

Hexagon SDK는 필요 없습니다. Zig의 최신 Hexagon 대상이 v79라서 v81은 v79로 컴파일하고 아키텍처 매크로를 81로 맞춥니다.

## 진단

* 모델 엔진 폴더에 `kdebug`라는 빈 파일을 두면 블록마다 최대값, inf/NaN 개수, 평균을 로그(logcat 태그 `kengine`)에 남깁니다.

## 주의사항

* HMX 루틴은 명령어 단어 파일(`hmx_words.S`)로 들어 있습니다. 툴체인의 어셈블러가 HMX 니모닉을 받지 않아서, 이 도구로는 새 HMX 루틴을 소스에서 빌드할 수 없습니다.
* 엔진은 폰 DSP 시스템의 인터페이스(전원, 연산 자원, 스레드, 서명 없는 모듈)에 의존하며, 그 구조체는 제조사가 공개하지 않습니다. 펌웨어가 바뀌면 동작하지 않을 수 있습니다.
* DSP 주소 공간이 32비트입니다. 버퍼와 상주 블록은 약 3.3 GB 안에 들어가게 잡습니다.
* 자동 시험이 없습니다. 변경은 폰에서 블록 출력과 그림을 비교해 확인했습니다.

## 라이선스

GPL-3.0(`LICENSE`)이며, 폰 DSP 시스템 라이브러리와의 링크에 대한 추가 허가가 붙어 있습니다. 하드웨어 정보의 출처와 서드파티 라이선스 전문은 `NOTICE`에 있습니다.

모델 가중치는 이 저장소에 들어 있지 않으며 각자의 라이선스를 따릅니다.

개발 과정에 Claude(Anthropic)가 사용되었습니다.
