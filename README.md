# Juvenile_delinquent

Wan 모델 영상생성 — 대사 없는 회화풍 단편 애니메이션의 첫 장면 제작 기록입니다.

첫 장면은 어두운 방에서 시험지를 내려다보는 어린 소년입니다. 짙은 남색·자주색과 따뜻한 책상 조명, 유화와 과슈의 붓 자국을 사용합니다. 소년이 조금 움츠러들고 창문의 가느다란 금이 번지는 움직임을 목표로 합니다. 이 이미지는 imagegen으로 제작·수정했으며, 영상 생성은 로컬 ComfyUI의 Wan으로 수행합니다.

![첫 장면 기준 그림](assets/shot01-younger.png)

## 포함 파일

| 파일 | 용도 |
|---|---|
| `workflows/shot01-comfyui.json` | ComfyUI 화면에서 열 수 있는 노드 워크플로 |
| `workflows/shot01-api.json` | 스크립트 제출용 API 워크플로 |
| `assets/shot01-younger.png` | 약 10~11세 소년으로 수정한 시작 그림 |
| `models.json` | 필요한 모델의 파일명, 크기, 다운로드 URL |
| `examples/shot01-younger.mp4` | 생성 결과를 정확히 5초로 정리한 MP4 시안 |
| `scripts/check_models.py` | 모델 파일의 존재·크기·헤더 검사 |
| `scripts/submit_workflow.py` | 그림 업로드, 생성 요청, 결과 다운로드 |

**대용량 모델 가중치는 이 저장소에 포함하지 않습니다.** 아래 네 파일을 별도로 준비합니다. API 키나 유료 온라인 Wan 서비스는 필요하지 않습니다.

## 필요한 모델

ComfyUI의 `models/` 아래에 다음 구조로 배치합니다. 용량은 GB(10진수), 합계 약 35.58GB입니다.

| 폴더 | 파일명 | 용량 |
|---|---|---:|
| `diffusion_models/` | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | 14.29GB |
| `diffusion_models/` | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | 14.29GB |
| `text_encoders/` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.74GB |
| `vae/` | `wan_2.1_vae.safetensors` | 0.25GB |

이 **14B I2V 워크플로는 Wan 2.1 VAE를 사용**합니다. 이름이 더 최신이라는 이유로 `wan2.2_vae.safetensors`로 바꾸지 않습니다. LoRA, CLIP Vision, WanVideoWrapper는 이번 기본 워크플로에 필요하지 않습니다.

- [ComfyUI 공식 Wan 2.2 안내](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- [Comfy-Org 배포 파일](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files)
- [Wan 공식 저장소](https://github.com/Wan-Video/Wan2.2)

다운로드 URL은 `models.json`에 있습니다. 사용하는 모델의 원본 라이선스를 확인합니다. 이 저장소는 해당 가중치의 소유권을 주장하지 않습니다.

## ComfyUI 화면에서 실행

1. NVIDIA GPU를 사용할 수 있는 ComfyUI를 실행합니다.
2. 위 모델 네 파일을 배치합니다.
3. `assets/shot01-younger.png`를 ComfyUI의 `input/` 폴더에 복사합니다.
4. `workflows/shot01-comfyui.json`을 ComfyUI 화면으로 끌어 놓거나 열기 기능으로 불러옵니다.
5. LoadImage에 `shot01-younger.png`가 선택됐는지 확인한 뒤 실행합니다.

ComfyUI 기본 노드만 사용합니다. 처음 실행할 때 모델 로딩 시간이 별도로 들며, 작업 완료 전에는 결과 영상이 저장되지 않습니다.

## 생성 설정

| 항목 | 값 |
|---|---|
| 모델 | Wan 2.2 I2V A14B, 고노이즈·저노이즈 FP8 scaled |
| 해상도 | 1024×768, 4:3 |
| 프레임 | 81프레임 |
| 저장 재생률 | 16fps, 약 5.06초 |
| 스텝 | 총 20: 고노이즈 10 + 저노이즈 10 |
| CFG / shift | 3.5 / 8 |
| 샘플러 / 스케줄러 | Euler / simple |
| 시드 | 20261118 |
| 출력 | 무음 WebM, VP9 |

정확히 5초로 편집하려면 결과의 마지막 한 프레임을 잘라 80프레임으로 저장합니다. 해상도를 낮추거나 프레임 수·스텝을 줄이면 결과와 소요 시간이 달라집니다. 인물 형태와 붓 자국 유지, 창문 금의 움직임은 생성 결과를 보고 평가해야 합니다.

## 스크립트로 실행

아래 보조 스크립트는 Python 표준 라이브러리만 사용합니다. 영상 생성 라이브러리와 GPU 실행은 **별도로 실행한 ComfyUI 서버**가 담당합니다.

```bash
python3 scripts/check_models.py /path/to/ComfyUI/models
python3 scripts/submit_workflow.py --dry-run
python3 scripts/submit_workflow.py --url http://127.0.0.1:8188 --wait
```

기본 주소는 8188입니다. ComfyUI를 다른 포트로 실행했다면 `--url`을 변경합니다. `--wait`를 생략하면 작업 번호를 받은 뒤 종료하고, 서버는 생성을 계속합니다. 대기 시간 초과가 발생해도 작업이 자동 취소되는 것은 아니므로 같은 작업을 바로 중복 제출하지 않습니다.

결과와 실제 제출 설정은 저장소의 `outputs/`에 내려받습니다. 다른 시드로 후보를 만들 때는 `--seed 20261119`처럼 지정합니다. 여러 GPU를 사용할 때는 서버 인스턴스마다 GPU와 포트, 출력 경로를 나눠 서로 다른 작업을 배정하는 방식부터 검토합니다. 이 스크립트 자체는 다중 GPU 분산 추론을 구현하지 않습니다.

정확히 5초의 MP4로 변환하는 예시입니다. FFmpeg는 별도 설치가 필요합니다.

```bash
ffmpeg -i result.webm -t 5 -an -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart result.mp4
```

## 확인한 로컬 환경과 현재 검증 상태

- NVIDIA DGX Spark / GB10, 128GB 통합 메모리, ARM64 Linux
- Python 3.12.3, PyTorch 2.10.0+cu128
- ComfyUI 0.17.0, 커밋 `f6b869d7d35f7160bf2fdeabaed378d737834540`
- 모델 네 파일의 헤더와 기록된 파일 길이 정상
- CUDA 기본 연산 성공
- 256×256, 9프레임, 2스텝의 작은 검증 영상 생성 성공: 모델 초기 로딩 포함 약 199초
- 1024×768, 81프레임, 20스텝의 첫 장면 생성 성공: **1시간 8분 45초(4,125초)**
- 원본 WebM은 약 5.063초이며, 마지막 한 프레임을 잘라 MP4 5.000초·80프레임·16fps로 변환
- 첫·중간·마지막 프레임을 점검했고 인물과 회화풍은 대체로 유지됨. 움직임은 작게 나왔으며, 창문의 금이 퍼지는 연출은 뚜렷하게 구현되지 않아 후속 조정이 필요함

[첫 5초 MP4 시안](examples/shot01-younger.mp4) · [실행 기록](examples/shot01-run.json)

현재 환경은 CUDA 최적화 관련 경고가 있고 SageAttention·FlashAttention이 설치되지 않았습니다. 위 수치를 최적화된 Spark나 다른 GPU의 성능으로 일반화하지 않습니다. 모델 검사는 파일 내용 전체의 SHA-256 검증이 아닌 헤더·길이 검사입니다.

[NVIDIA Spark ComfyUI 안내](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-comfyui/README.md)는 CUDA 13.0 지원 환경을 안내합니다. 기존 환경을 바로 덮어쓰기보다 별도 환경에서 비교하는 편이 재현에 유리합니다.

## 폐쇄망으로 옮길 때

이 저장소만 복사해서는 실행 환경이 완성되지 않습니다. 모델 네 파일과 ComfyUI 소스, 대상 GPU 및 CPU 아키텍처에 맞는 Python·PyTorch·CUDA 의존성을 함께 준비해야 합니다. Spark의 ARM64용 설치 파일을 x86_64 서버에 그대로 사용할 수는 없습니다.

외부망에서 환경과 모델을 준비하고 작은 검증 작업부터 실행합니다. 반입 후에는 다운로드나 API 호출 없이 실행할 수 있도록 모든 파일을 포함합니다. `check_models.py`로 파일을 확인하고, 각 결과의 워크플로·시드·로그를 영상과 함께 반환하면 다음 수정 작업을 재현할 수 있습니다.
