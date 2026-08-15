# ARLO DUV `Unet_only` 학습 설계 명세

**상태:** 설계 초안 — 사용자 승인 설계 반영, 명세 검토 대기

**작성일:** 2026-08-13

**근거 문서:** `ARLO_DUV_4인_공통_성과지표_추출방법_논문근거_표준프로토콜.md`

## 1. 목표

네 명의 동료가 VS Code, Kaggle, Colab 등 서로 다른 환경에서 동일한 조건으로 실행할 수 있는 독립형 `Unet_only` 학습 뼈대를 만든다.

현재 단계의 산출물은 다음 입력-출력 관계만 구현한다.

```text
MetalSet/target 입력
    → pinned LithoBench U-Net
    → 256 × 256 probability mask
    → pixelILT 정답 mask와 probability BCE 학습
    → internal validation BCE 기준 best.pt
```

`pixelILT`는 이미 준비된 지도학습 정답 mask로만 사용한다. simulator나 physical evaluator는 이 학습 명세와 구현 범위에 포함하지 않는다.

## 2. 범위

### 2.1 이번 단계에 포함

- MetalSet `target`과 `pixelILT`의 sample ID 대응 확인
- frozen split 검증
- 256 × 256 입력·출력
- pinned LithoBench `neuralilt.UNet()` 구조와 동일한 U-Net backbone
- deterministic horizontal flip·vertical flip augmentation
- probability-output BCE 학습
- AdamW 30 epoch 학습
- internal validation BCE 최저 checkpoint 선택
- `best.pt`, `last.pt`, 초기 state, history, manifest 저장
- parameter count·tensor shape·FLOPs 구조 audit
- CPU·CUDA·Kaggle·Colab·VS Code를 고려한 경로·device 설정

### 2.2 이번 단계에서 제외

- simulator·LithoSim·physical evaluator 실행
- L2, PVB, EPE, Shot 등 physical metric 계산
- `adaptive-boxes` 및 기타 physical evaluator 의존성
- 2048 × 2048 변환과 process-corner simulation
- final holdout 1,648개 성과지표 산출
- per-sample physical metric CSV와 후속 통계
- Attention module
- PPO, RL, reward, iterative mask refinement

후속 평가 단계는 별도 명세로 분리한다. 이 문서는 학습과 checkpoint 산출에 필요한 내용만 다룬다.

## 3. 표준 프로토콜에서 고정하는 값

아래 값은 `ARLO_DUV_4인_공통_성과지표_추출방법_논문근거_표준프로토콜.md`의 2026-08-13 최종 합의값을 그대로 반영한다.

| 항목 | `Unet_only` primary 값 |
|---|---|
| benchmark source | LithoBench |
| pinned LithoBench commit | `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6` |
| task subset | MetalSet simulation protocol |
| train pool | 14,824 |
| train split | 13,342 |
| internal validation split | 1,482 |
| final holdout test split | 1,648 |
| model input | 256 × 256 |
| model output | 1 channel, 256 × 256 |
| actual batch size | 4 |
| gradient accumulation | 1 |
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-4` |
| scheduler | 없음 |
| loss | probability BCE |
| maximum epoch | 30 |
| early stopping | 사용하지 않음 |
| seed | 42 |
| AMP | 사용하지 않음 |
| numerical dtype | FP32 |
| best checkpoint | 256 × 256 internal validation BCE 최저 checkpoint |
| output contract | probability mask, 1 channel, 256 × 256 |

과거 노트북에 남아 있는 512 입력, batch 2·8, epoch 20, AMP 사용, 다른 loss 설정은 이번 primary 코드의 기준으로 사용하지 않는다.

## 4. 데이터 계약

### 4.1 디렉터리 계약

기본 데이터 구조는 다음과 같다. 여기서 `DATASET_ROOT`는 실행 환경별 실제 경로로 치환한다.

```text
DATASET_ROOT/
├─ target/
│  ├─ sample_00001.png
│  └─ ...
└─ pixelILT/
   ├─ sample_00001.png
   └─ ...
```

다음 조건을 모두 만족해야 한다.

1. `target`과 `pixelILT`가 모두 존재한다.
2. 두 폴더의 sample ID가 일치한다.
3. 같은 sample ID는 하나의 target과 하나의 pixelILT에만 대응한다.
4. target과 pixelILT의 orientation과 좌표계가 동일하다.
5. 전체 대응 쌍이 16,472개다.
6. 이미지 파일은 grayscale로 읽을 수 있다.
7. pairing 실패·중복·누락은 학습 시작 전에 오류로 중단한다.

sample ID는 frozen split에 기록된 확장자 포함 파일명으로 보존한다. pairing key는 운영체제의 대소문자 규칙에 의존하지 않도록 casefold한 파일명으로 만들며, casefold 후 중복되는 파일은 오류로 거부한다.

### 4.2 split 계약

primary 실행은 random train-test split을 매번 새로 만들지 않는다. frozen split manifest를 입력으로 받고 다음 split만 허용한다.

```text
train:             13,342
internal_validation: 1,482
final_holdout_test: 1,648
total:            16,472
```

split manifest의 최소 필드는 다음과 같다.

```text
sample_id,target_path,mask_path,split
```

`split` 값은 `train`, `internal_validation`, `final_holdout_test` 중 하나다.

학습 시 다음을 검증한다.

- split별 개수
- sample ID 중복 없음
- 세 split 간 교집합 없음
- target·pixelILT path가 실제 파일을 가리킴
- split manifest의 normalized LF SHA-256
- final holdout sample이 train 또는 internal validation에 포함되지 않음

split manifest가 없을 때 후보 목록을 생성하는 보조 모드는 둘 수 있지만, primary 학습은 검증된 manifest 없이는 실행하지 않는다. 후보 생성은 sorted filename을 이용하며, 생성된 목록을 동료에게 배포한 후 primary manifest로 고정한다.

### 4.3 이미지 전처리

각 이미지는 다음 순서로 처리한다.

1. grayscale로 로드한다.
2. 원본 크기가 256 × 256이 아니면 nearest interpolation으로 256 × 256으로 변환한다.
3. 정수 intensity를 `[0, 1]` float32로 변환한다.
4. `value >= 0.5`를 1.0, 나머지를 0.0으로 변환한다.
5. channel dimension을 추가해 `[1, 256, 256]`으로 만든다.

target은 모델 입력이고 pixelILT는 BCE 정답이다. 두 tensor에는 동일한 orientation과 동일한 paired augmentation을 적용한다.

모델 출력은 BCE 계산 전까지 확률값으로 보존한다. 학습 중 threshold를 적용해 binary output으로 바꾸지 않는다.

## 5. deterministic augmentation과 DataLoader

### 5.1 augmentation

학습 dataset에만 다음 augmentation을 적용한다.

- horizontal flip: 독립적으로 50% 확률
- vertical flip: 독립적으로 50% 확률
- flip 순서: horizontal 후 vertical
- target과 pixelILT에 같은 flip 적용
- validation에는 적용하지 않음

random state에 의존해 worker 실행 순서가 바뀌는 것을 막기 위해 다음 deterministic key를 사용한다.

```text
sha256(f"{seed}:{epoch}:{sample_id}")
```

digest의 첫 번째 byte bit 0을 horizontal flip, bit 1을 vertical flip에 사용한다. 이 방식은 epoch와 sample ID가 같으면 환경이 달라도 같은 augmentation 결정을 재현한다.

### 5.2 DataLoader 운영값

문서에서 직접 고정하지 않은 DataLoader 값은 cross-environment 기본값으로 명시한다.

| 항목 | 값 |
|---|---|
| training shuffle | 사용 |
| validation shuffle | 사용하지 않음 |
| `drop_last` | `False` |
| 마지막 잔여 batch | 사용 |
| default `num_workers` | 0 |
| `pin_memory` | CUDA device일 때만 사용 |
| `persistent_workers` | 사용하지 않음 |
| epoch별 loader generator | seed 42와 epoch를 이용해 고정 |

13,342개 train sample은 batch size 4로 순회하며, 마지막 잔여 batch도 버리지 않는다. 실제 optimizer update의 기본 batch parameter는 4이고 gradient accumulation은 1이다.

## 6. pinned U-Net backbone 계약

### 6.1 구조

현재 단계의 baseline은 Attention 없는 pinned U-Net이다.

| 단계 | 공간 크기 | 출력 채널 | 처리 |
|---|---:|---:|---|
| input | 256 × 256 | 1 | binary target input |
| encoder 1 | 256 × 256 | 64 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| pool 1 | 128 × 128 | 64 | MaxPool2d 2 × 2, stride 2 |
| encoder 2 | 128 × 128 | 128 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| pool 2 | 64 × 64 | 128 | MaxPool2d 2 × 2, stride 2 |
| encoder 3 | 64 × 64 | 256 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| pool 3 | 32 × 32 | 256 | MaxPool2d 2 × 2, stride 2 |
| encoder 4 / bottleneck | 32 × 32 | 512 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| upsample 4 | 64 × 64 | 512 | bilinear, scale factor 2, `align_corners=True` |
| decoder 4 input | 64 × 64 | 768 | upsample 512 + encoder 3 skip 256 concat |
| decoder 4 | 64 × 64 | 256 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| upsample 3 | 128 × 128 | 256 | bilinear, scale factor 2, `align_corners=True` |
| decoder 3 input | 128 × 128 | 384 | upsample 256 + encoder 2 skip 128 concat |
| decoder 3 | 128 × 128 | 128 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| upsample 2 | 256 × 256 | 128 | bilinear, scale factor 2, `align_corners=True` |
| decoder 2 input | 256 × 256 | 192 | upsample 128 + encoder 1 skip 64 concat |
| decoder 2 | 256 × 256 | 64 | 3 × 3 convolution 2회, BatchNorm2d, ReLU |
| output head | 256 × 256 | 1 | 3 × 3 convolution, Sigmoid |

convolution 입출력 channel 순서는 다음과 같이 고정한다.

| Block | convolution channel sequence |
|---|---|
| conv1 | 1 → 64, 64 → 64 |
| conv2 | 64 → 128, 128 → 128 |
| conv3 | 128 → 256, 256 → 256 |
| conv4 | 256 → 512, 512 → 512 |
| deconv4 | 768 → 256, 256 → 256 |
| deconv3 | 384 → 128, 128 → 128 |
| deconv2 | 192 → 64, 64 → 64 |
| deconv1 | 64 → 1 |

모든 convolution은 kernel `3 × 3`, stride `1`, padding `1`, dilation `1`, groups `1`이다. encoder·decoder block은 `Conv2d → BatchNorm2d → ReLU` 순서를 지킨다. output head에는 BatchNorm2d와 ReLU를 넣지 않는다.

dropout, deep supervision, 추가 encoder·decoder stage는 사용하지 않는다.

### 6.2 source와 구현 방식

현재 단계에서는 `pylitho`나 전체 LithoBench runtime package를 import하지 않는다. `lithobench.ilt.neuralilt.UNet()`은 공통 U-Net 구조를 확인하기 위한 upstream reference이며, 실제 학습 코드는 그 구조를 독립적인 `unet_only_backbone.py`에 보존한다. 여기서 `ilt`는 모델 소스 경로의 이름일 뿐, ILT simulator를 실행한다는 뜻이 아니다.

manifest에는 다음을 구분해 기록한다.

- `backbone_reference_source`: `lithobench.ilt.neuralilt.UNet`
- `backbone_reference_commit`: `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6`
- `backbone_source_hash`: 공유하는 local `unet_only_backbone.py`의 LF-normalized SHA-256
- `backbone_parameter_count`: runtime 계산값
- `backbone_flops`: runtime 계산값

local source hash는 네 사람이 같은 파일을 사용했는지 확인하는 값이며, upstream repository 전체 파일의 hash라고 표현하지 않는다.

### 6.3 forward 순서와 shape audit

baseline의 canonical forward는 다음과 같다.

```text
e1 = encoder1(input)                  # 64  × 256 × 256
e2 = encoder2(pool(e1))               # 128 × 128 × 128
e3 = encoder3(pool(e2))               # 256 × 64  × 64
b  = encoder4(pool(e3))               # 512 × 32  × 32

u4 = upsample(b)                      # 512 × 64  × 64
d4 = decoder4(concat(u4, e3))         # 768 -> 256 × 64  × 64
u3 = upsample(d4)                     # 256 × 128 × 128
d3 = decoder3(concat(u3, e2))         # 384 -> 128 × 128 × 128
u2 = upsample(d3)                     # 128 × 256 × 256
d2 = decoder2(concat(u2, e1))         # 192 -> 64  × 256 × 256
output = sigmoid(output_head(d2))     # 1 × 256 × 256
```

구조 audit는 다음을 확인한다.

- batch 1과 batch 2에서 입력·출력 shape
- output range가 `[0, 1]`인지
- parameter count가 `7,787,905`인지
- 각 stage spatial shape와 channel 수
- skip concat channel 수
- `align_corners=True`
- output head 뒤 Sigmoid 존재

### 6.4 parameter count와 FLOPs

pinned backbone parameter count는 다음 값과 일치해야 한다.

```text
sum(parameter.numel() for parameter in model.parameters()) == 7_787_905
```

FLOPs는 batch 1, input shape `1 × 256 × 256`, forward only에서 convolution multiply-add만 센다. Conv2d 한 번의 계산은 다음으로 정의한다.

$$
\mathrm{FLOPs}_{\mathrm{Conv2d}}
=
2H_oW_oC_o\left(\frac{C_i}{G}\right)K_hK_w
$$

ReLU, Sigmoid, BatchNorm2d, MaxPool2d, bilinear interpolation은 primary backbone FLOPs에 포함하지 않는다.

이 구조에서 audit counter가 산출해야 하는 기준 FLOPs는 `84,708,163,584`다. counter 구현과 source hash도 manifest에 기록한다.

## 7. 재현성과 device

### 7.1 seed

seed 42를 다음에 적용한다.

- Python `random`
- NumPy
- `torch.manual_seed`
- CUDA seed when available
- DataLoader generator
- deterministic paired augmentation

CUDA가 사용되면 다음을 설정한다.

- cuDNN deterministic: enabled
- cuDNN benchmark: disabled
- TF32: disabled when the backend exposes the setting
- deterministic algorithms: enabled with warning-only fallback where necessary

하드웨어가 Intel XPU, RTX 2060, T4 등으로 다르면 seed가 같아도 bitwise identical 결과를 보장하지 않는다. AMP를 끄고 FP32를 사용해 수치 정밀도 차이를 줄이는 것이 primary 비교의 목적이다.

### 7.2 device와 dtype

- `device=auto`는 CUDA, XPU, MPS, CPU 순서로 사용 가능한 backend를 선택한다.
- 사용자가 `cpu`, `cuda`, `xpu`, `mps` 중 하나를 명시할 수 있다.
- 모델·입력·정답은 FP32로 유지한다.
- AMP와 gradient scaler는 생성하지 않는다.
- XPU와 MPS는 설치된 PyTorch가 해당 backend를 제공할 때만 선택한다.
- device와 PyTorch version은 manifest에 기록한다.

## 8. loss와 optimizer

모델은 Sigmoid probability를 출력하므로 probability BCE를 사용한다.

```python
prediction = model(target)
loss = torch.nn.functional.binary_cross_entropy(
    prediction.float().clamp(1e-6, 1.0 - 1e-6),
    mask.float(),
    reduction="mean",
)
```

학습 optimizer는 다음과 같이 고정한다.

```python
torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
    amsgrad=False,
)
```

`betas`, `eps`, `amsgrad`는 원 표준 문서에 숫자가 직접 지정되어 있지 않으므로, PyTorch 기본값을 명시적으로 고정한 구현 세부값이다. 모든 동료가 같은 값을 사용해야 한다.

학습 epoch마다 다음 순서로 실행한다.

1. train dataset의 epoch를 설정한다.
2. train loader를 만든다.
3. 각 batch에 대해 `zero_grad → forward → probability BCE → backward → optimizer.step`을 실행한다.
4. train loss를 전체 sample 기준 평균으로 저장한다.
5. augmentation 없는 internal validation을 실행한다.
6. validation BCE가 현재 best보다 낮으면 `best.pt`를 갱신한다.
7. `last.pt`, history, status, manifest를 atomic write한다.

early stopping은 사용하지 않으므로 validation이 좋아지지 않아도 30 epoch까지 진행한다. validation BCE가 같은 경우에는 먼저 나온 epoch를 best로 유지한다.

## 9. checkpoint와 산출물 계약

권장 디렉터리 구조는 다음과 같다.

```text
unet_only_output/
├─ checkpoints/
│  ├─ initial_unet_state.pt
│  ├─ last.pt
│  └─ best.pt
├─ history.csv
├─ run_manifest.json
├─ training_contract.json
├─ source_hashes.json
├─ split_manifest.csv
└─ training_status.json
```

### 9.1 checkpoint payload

`last.pt`와 `best.pt`에는 최소한 다음을 포함한다.

- model state dict
- optimizer state dict
- completed epoch
- best validation BCE
- current validation BCE
- training history
- configuration fingerprint
- split hash
- source hash
- model variant

`initial_unet_state.pt`는 seed 42로 model을 생성한 직후의 CPU state dict를 저장한다. 후속 Attention 모델이 같은 baseline backbone 초기 state를 사용할 수 있도록 보존한다.

### 9.2 history.csv

최소 필드는 다음과 같다.

```text
epoch,train_bce,internal_validation_bce,epoch_seconds,learning_rate
```

epoch는 1부터 30까지 기록한다.

### 9.3 run_manifest.json

manifest에는 표준 문서의 backbone fields를 포함한다.

```json
{
  "model_variant": "unet_only",
  "backbone_source": "lithobench.ilt.neuralilt.UNet",
  "backbone_commit": "9c74e82218e377eaf6d02d113fc1ce6e36c92aa6",
  "backbone_source_hash": "runtime LF-normalized SHA-256 of unet_only_backbone.py",
  "backbone_input_shape": [1, 256, 256],
  "backbone_output_shape": [1, 256, 256],
  "backbone_parameter_count": 7787905,
  "backbone_flops_definition": "conv2d_multiply_add_only",
  "backbone_flops_counter_source": "unet_only_backbone.py",
  "backbone_flops_counter_version": "runtime source SHA-256",
  "backbone_flops": 84708163584,
  "attention_module_is_common": false,
  "attention_variant_id": null,
  "attention_source_hash": null,
  "flops_definition": "conv2d_multiply_add_only",
  "flops_counter_source": "unet_only_backbone.py",
  "flops_counter_version": "runtime source SHA-256",
  "flops_input_shape": [1, 256, 256],
  "total_model_flops": 84708163584
}
```

추가로 다음을 기록한다.

- protocol ID
- dataset root의 절대 경로가 아닌 실행 환경별 표시명
- split counts와 split hash
- seed
- input size와 output size
- batch size와 gradient accumulation
- optimizer, betas, eps, weight decay
- scheduler 없음
- probability BCE와 reduction
- augmentation rule
- AMP off와 FP32
- device backend와 PyTorch version
- best epoch와 best validation BCE
- actual completed epoch
- checkpoint hash

절대 경로는 재현성 fingerprint에 넣지 않는다. 동료 환경이 달라도 동일한 데이터·코드 hash를 비교할 수 있어야 한다.

## 10. 코드 파일 설계

구현은 다음 최소 파일로 구성한다.

```text
C:\Users\leska\문서\AI티카타카\unet_only\
├─ unet_only_backbone.py
├─ unet_only_train.py
├─ requirements.txt
├─ README.md
└─ tests\
   └─ test_unet_only.py
```

### `unet_only_backbone.py`

- `conv2d`
- `repeat2d`
- `UNet`
- parameter counter
- Conv2d-only FLOPs counter
- forward shape audit
- LF-normalized source hash helper

이 파일에는 LithoSim, `pylitho`, `pyilt`, dataset I/O를 import하지 않는다.

### `unet_only_train.py`

- CLI argument parsing
- configuration validation
- split manifest loading·검증
- target·pixelILT dataset
- deterministic augmentation
- DataLoader
- seed·device 설정
- probability BCE
- AdamW 학습 loop
- internal validation
- best·last checkpoint
- history·manifest·status atomic write
- resume fingerprint 검증

### `requirements.txt`

ILT simulator나 LithoBench runtime package를 넣지 않는다. 최소 의존성은 다음으로 제한한다.

- PyTorch
- NumPy
- Pillow

PyTorch의 CPU·CUDA 설치 방법은 실행 환경별로 달라질 수 있으므로 `torch` wheel은 README에서 환경별 설치 안내로 분리한다.

### `README.md`

- dataset directory contract
- split manifest 준비 방법
- CPU·CUDA·Kaggle·Colab 실행 예시
- output artifact 설명
- primary protocol에서 바꾸면 안 되는 값
- resume 방법
- ILT evaluator가 이번 범위에 없다는 설명

## 11. CLI와 portability

primary 실행 인터페이스는 다음 형태로 설계한다.

```text
python unet_only_train.py \
  --dataset-root DATASET_ROOT \
  --split-csv FROZEN_SPLIT_CSV \
  --output-root OUTPUT_ROOT \
  --device auto \
  --num-workers 0
```

공통 protocol 값을 임의로 바꾸는 CLI option은 제공하지 않는다. smoke test용 mode에서는 synthetic 또는 소량 fixture를 사용할 수 있지만, output manifest에 `mode=smoke`를 기록하고 primary 결과로 취급하지 않는다.

다음 값은 실행 중 자동 탐색할 수 있다.

- CPU 또는 CUDA device
- output root
- dataset root
- input file path

다음 값은 실행 중 자동 변경하지 않는다.

- split count
- input size
- batch size
- learning rate
- weight decay
- epoch
- loss
- threshold
- early stopping

## 12. 테스트와 완료 기준

### 12.1 구조 테스트

- parameter count가 `7,787,905`
- batch 1·2 output shape가 각각 `[B, 1, 256, 256]`
- output 범위가 `[0, 1]`
- encoder·decoder stage channel과 spatial shape가 계약과 일치
- skip concat channel이 768, 384, 192
- output head에 Sigmoid가 존재
- FLOPs가 `84,708,163,584`

### 12.2 데이터·재현성 테스트

- 동일 sample ID의 target·pixelILT pairing
- 누락·중복 ID 거부
- split count와 leakage 검증
- 같은 seed·epoch·sample ID에서 flip 결정 동일
- validation dataset에 augmentation 미적용
- final holdout이 train/validation loader에 포함되지 않음

### 12.3 학습 테스트

- probability BCE가 확률 output에 계산됨
- output을 thresholding한 뒤 BCE를 계산하지 않음
- AdamW 값과 scheduler 없음이 manifest에 기록됨
- gradient accumulation이 1로 동작
- validation BCE 최저 epoch가 best checkpoint가 됨
- early stopping 없이 지정 epoch까지 실행
- interrupted run에서 fingerprint가 다른 checkpoint를 거부
- smoke fixture로 forward·backward·checkpoint 저장을 완료

### 12.4 완료 조건

다음 조건을 모두 만족해야 현재 단계가 완료된 것으로 판정한다.

1. 테스트가 먼저 실패한 뒤 최소 구현으로 통과한다.
2. U-Net parameter count와 FLOPs audit가 통과한다.
3. 표준 split이 누출 없이 검증된다.
4. 256 입력·출력 shape가 통과한다.
5. probability BCE와 AdamW 설정이 통과한다.
6. AMP off·FP32·seed 42가 manifest에 기록된다.
7. 30 epoch 또는 resume 후 총 30 epoch까지 학습할 수 있다.
8. internal validation BCE 최저 `best.pt`가 존재한다.
9. simulator·physical evaluator를 import하지 않는다.
10. 초기 state·best checkpoint·training manifest가 보존된다.

## 13. 해석상 제한

이번 산출물은 U-Net mask prediction 학습 코드다. `best.pt`는 internal validation BCE 기준 checkpoint일 뿐이며, physical simulation이나 성과지표를 검증한 결과가 아니다.

현재 단계에서 허용되는 표현은 다음과 같다.

```text
256 × 256 internal validation BCE 기준 best checkpoint를 선택했다.
```
