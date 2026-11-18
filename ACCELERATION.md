# Wan 2.2 가속 설정

## 무엇을 바꿨나

Wan 2.2 I2V A14B FP8 본체, UMT5 텍스트 인코더, Wan 2.1 VAE는 그대로 사용합니다. 기존 설치를 보존하고 별도 Python 가상환경에서 다음을 검증했습니다.

- PyTorch 2.10.0+cu128 → 2.10.0+cu130. torchvision 0.25.0, torchaudio 2.10.0도 cu130으로 맞춤
- GB10 `sm_121` 대상으로 직접 빌드한 SageAttention 2.2.0 사용
- LightX2V 1022 고노이즈·저노이즈 LoRA 추가
- 기본 20스텝·CFG 3.5·shift 8과, 가속 4/8스텝·CFG 1·shift 5 비교

가속 LoRA는 단순히 스텝 숫자만 줄이는 것과 다릅니다. 적은 계산으로 생성하도록 학습된 보정 가중치입니다. 화풍과 인물, 움직임에 영향을 줄 수 있습니다. CFG 1에서는 일반적인 네거티브 프롬프트의 효과를 기대하면 안 됩니다.

## 실제 측정

NVIDIA DGX Spark / GB10, 128GB 통합 메모리에서 2026-11-18에 측정했습니다. 아래는 ComfyUI 실행부터 영상 저장까지의 시간이며, 대기열 대기와 설치 시간은 제외합니다.

| 환경·모드 | 해상도 / 프레임 | 실행 시간 | 모델 상태 |
|---|---|---:|---|
| 기존 cu128, 기본 20스텝·CFG 3.5 | 1024×768 / 81 | 4,125초 (68분 45초) | 사전 검증 후 본체 재사용 |
| cu130 + Sage, 기본 20스텝·CFG 3.5 | 512×384 / 81 | 421.64초 | 새 서버의 첫 실행, 모델 로딩 포함 |
| cu130 + Sage + LightX2V 4스텝 | 512×384 / 81 | 65.30초 | 기본 모델 재사용, 새 LoRA 적용 시간 포함 |
| cu130 + Sage + LightX2V 8스텝 | 512×384 / 81 | 53.54초 | 직전 LoRA와 모델 재사용 |
| cu130 + Sage + LightX2V 4스텝 | 1024×768 / 81 | 237.31초 (3분 57초) | LoRA·모델 재사용, 일부 가중치 오프로딩 발생 |

4스텝보다 8스텝의 전체 시간이 짧게 보이는 것은 모델 재사용 조건이 달랐기 때문입니다. 샘플러 계산 자체는 4스텝이 약 18초, 8스텝이 약 40초였습니다(512×384). 실제 반복 작업의 시간은 메모리, 캐시, 다른 프로그램에 따라 달라집니다.

기존 1024 결과 대비 새 4스텝 실행은 약 17.4배 빨랐지만, 여러 설정과 프롬프트를 함께 바꿨습니다. 이 수치를 CUDA나 SageAttention만의 개선율로 해석하지 않습니다. 입력 그림과 연출도 달라 기존 결과와 동일 화질을 보장하지 않습니다.

작은 attention 연산 검사도 수행했습니다. BF16 `[1, 40, 4096, 128]`에서 cu128 SDPA 약 6.4ms, cu130 SDPA 약 3.9ms였습니다. SageAttention 검사 결과는 유한했고, 같은 실행의 SDPA 대비 상대 RMSE는 약 0.039였습니다. 이것은 영상 품질이나 무손실을 보장하는 지표가 아닙니다.

## 영상으로 비교

[512 비교 영상](examples/optimized/comparison-512.mp4)의 **왼쪽은 기본 20스텝, 가운데는 가속 4스텝, 오른쪽은 가속 8스텝**입니다. 같은 시작 그림·프롬프트·시드를 사용했습니다.

[가속 4스텝 1024×768 시안](examples/optimized/light4-1024.mp4) · [다음 10초 시안](examples/optimized/next-10-seconds.mp4) · [전체 15초](examples/optimized/opening-15-seconds.mp4)

이어지는 시험지 숏도 같은 4스텝으로 229.35초에 생성했습니다. 움직임은 있으나, 종이를 넓게 펼쳐 내미는 모습에 가깝고 원래 의도한 마법 같은 신장은 추가 조정이 필요합니다. 이 숏은 `shot03-paper-lightx2v-4step-*` 워크플로와 `assets/shot03-start-light4.png`를 사용합니다.

소년이 어른을 보고 다시 종이를 드는 동작은 세 모드에서 모두 나타났습니다. 가속 4·8스텝에서는 정면을 보는 얼굴의 눈이 커지고 피부의 붓 자국이 약해지는 경향이 있었습니다. 8스텝이 이 장면에서 인물·화풍을 결정적으로 개선하지는 않았습니다. 4스텝은 빠른 시안 탐색에 유용하며, 최종 장면은 인물과 화풍을 확인하고 선택해야 합니다.

## LoRA 준비와 변환

대용량 가중치는 GitHub에 넣지 않습니다. [모델 목록](models-lightx2v.json)의 두 원본 파일을 ComfyUI의 `models/loras/`에 받습니다.

```bash
hf download lightx2v/Wan2.2-Distill-Loras \
  wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --local-dir /path/to/ComfyUI/models/loras
python3 scripts/check_models.py /path/to/ComfyUI/models --manifest models-lightx2v.json
python3 scripts/convert_lightx2v.py --models-dir /path/to/ComfyUI/models
```

`hf`가 없다면 목록의 다운로드 URL로 파일을 받아도 됩니다. 변환 스크립트에는 ComfyUI 환경의 PyTorch와 safetensors가 필요합니다. 원본은 보존하고 다음 사본을 만듭니다.

- `lightx2v_1022_high_native.safetensors`
- `lightx2v_1022_low_native.safetensors`

공식 예제는 WanVideoWrapper 형식입니다. 이 프로젝트의 기본 ComfyUI 노드용으로 고노이즈의 `.diff_m` 41개 이름을 `.modulation.diff`로 바꾸고, 저노이즈에 포함된 Wan 2.1 이미지 모듈 중 Wan 2.2 본체에 존재하지 않는 290개 항목은 제외합니다. 유지하는 텐서의 값은 바꾸지 않습니다. 알 수 없는 다른 항목이 있으면 변환을 중단하며, 결과 파일도 덮어쓰지 않습니다. 변환 기록은 출력 폴더에 저장합니다.

## Spark 실행 환경

아래는 이 저장소에서 검증한 Linux ARM64 / GB10 구성을 재현하는 방법입니다. B300 서버나 다른 CPU 아키텍처에는 그 장비에 맞는 환경과 커널 빌드가 필요합니다.

기존 ComfyUI 의존성을 재사용하는 별도 환경을 만들었습니다. 현재 설치를 덮어쓰지 않습니다.

```bash
python3 -m venv --system-site-packages .venv-wan-fast
.venv-wan-fast/bin/python -m pip install --force-reinstall \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
.venv-wan-fast/bin/python -m pip install \
  numpy==2.3.5 fsspec==2026.2.0 pillow==10.1.0
```

이전 torch가 상속돼 이미 설치됐다고 표시된다면 가상환경의 pip에 `--force-reinstall`을 지정합니다. 실제 검증에서는 `--force-reinstall`로 cu130을 설치했습니다. `python -c 'import torch; print(torch.__version__, torch.version.cuda)'`로 새 환경의 버전을 확인합니다.

SageAttention 공식 소스를 다음 커밋으로 고정해 빌드했습니다.

```bash
mkdir -p .build-src
git clone https://github.com/thu-ml/SageAttention.git .build-src/SageAttention
git -C .build-src/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.1 \
  MAX_JOBS=4 EXT_PARALLEL=2 NVCC_APPEND_FLAGS='--threads 2' \
  .venv-wan-fast/bin/python -m pip install --no-build-isolation --no-deps .build-src/SageAttention
```

빌드에는 `Python.h`가 필요합니다. 검증 PC에서는 Ubuntu 공식 `libpython3.12-dev`와 `python3.12-dev`의 deb를 작업 폴더에 내려받아 풀고, 시스템에 설치하지 않은 채 `CPATH`에 지정했습니다. 같은 방법을 쓰면 압축을 푼 `usr/include` 경로를 `PYTHON_DEV_INCLUDE_ROOT`에 설정하고, 빌드 명령에 `CPATH="$PYTHON_DEV_INCLUDE_ROOT:$PYTHON_DEV_INCLUDE_ROOT/python3.12"`를 추가합니다.

SageAttention은 실제 GPU에서 작동을 확인했습니다. PyTorch 본체에는 GB10의 12.1에 대해 빌드 범위 최대 12.0을 표시하는 경고가 남아 있습니다. 모든 경고가 해소된 환경이라고 표현하지 않으며, 실제 생성 성공과 측정값으로 판단합니다.

## 새 워크플로 실행

1. `assets/shot02-start.png`를 ComfyUI의 `input/`에 복사합니다.
2. `scripts/run_comfy_optimized.sh /path/to/ComfyUI /path/to/.venv-wan-fast 8197`로 실행합니다.
3. `workflows/shot02-lightx2v-4step-comfyui.json` 또는 `shot02-lightx2v-8step-comfyui.json`을 엽니다.

런처는 `--use-sage-attention --disable-dynamic-vram`과 기본 노드만 사용합니다. 같은 포트에 ComfyUI가 이미 실행 중이면 먼저 종료하거나 다른 포트를 사용합니다. 별도 경로에 Python 헤더를 풀었다면 런처에도 `PYTHON_DEV_INCLUDE_ROOT`를 전달합니다.

스크립트 실행 예시:

```bash
python3 scripts/submit_workflow.py \
  --url http://127.0.0.1:8197 \
  --workflow workflows/shot02-lightx2v-4step-api.json \
  --image assets/shot02-start.png --wait
```

## 참고 자료

- [LightX2V LoRA 공식 모델 카드](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- [LightX2V MoE 4스텝 설명](https://huggingface.co/lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v)
- [SageAttention 공식 소스](https://github.com/thu-ml/SageAttention)
- [NVIDIA Spark ComfyUI案内](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-comfyui/README.md)
