응. **지금 단계에서는 “새 `Unet_only` 학습 설계 명세가 무조건 1순위”**로 두고, 팀원이 만든 코드를 베이스로 유지하면서 **기존 train 코드에서 새 명세에 도움이 되는 부분만 가져오는 방향**이 가장 좋다.

그리고 먼저 중요한 결론부터 말하면:

> **팀원 코드의 기본 구조는 새 `Unet_only` 명세를 상당히 잘 반영하고 있다.**
>
> 하지만 **그대로 실행해서 최종 학습을 시작하면 안 된다.**
>
> 특히 **① AMP, ② frozen split manifest, ③ 초기 state/manifest 저장, ④ pinned commit/source hash 기록, ⑤ checkpoint에 학습 계약 정보 저장** 부분은 수정 또는 보강하는 게 좋다.

---

# 1. 전체 판단

새 학습 명세를 기준으로 우선순위를 정하면:

| 구분                       | 현재 팀원 코드                        | 판단                   |
| ------------------------ | ------------------------------- | -------------------- |
| 공식 NeuralILT U-Net       | `lithobench.ilt.neuralilt.UNet` | ✅ 그대로 유지             |
| 256×256                  | 적용                              | ✅ 그대로 유지             |
| 16,472 전체                | 적용                              | ✅                    |
| train pool 14,824        | 적용                              | ✅                    |
| train 13,342             | 적용                              | ✅                    |
| internal val 1,482       | 적용                              | ✅                    |
| final holdout 1,648      | 적용                              | ✅                    |
| Seed 42                  | 적용                              | ✅                    |
| Batch 4                  | 적용                              | ✅                    |
| AdamW                    | 적용                              | ✅                    |
| LR `1e-4`                | 적용                              | ✅                    |
| WD `1e-4`                | 적용                              | ✅                    |
| scheduler 없음             | 적용                              | ✅                    |
| BCE                      | 적용                              | ✅                    |
| 30 epoch                 | 적용                              | ✅                    |
| early stopping 없음        | 적용                              | ✅                    |
| H/V flip                 | 적용                              | ✅                    |
| random crop 없음           | 적용                              | ✅                    |
| AMP False                | ⚠️ 코드 전체 확인 필요 / 기존 코드에는 AMP 사용 | **수정 필요**            |
| FP32                     | ⚠️ AMP 사용 시 깨짐                  | **수정 필요**            |
| frozen split manifest    | ❌                               | **중요 수정**            |
| split SHA-256            | ❌                               | **추가 권장/사실상 명세상 필요** |
| pinned LithoBench commit | ❌ 코드에서 검증 없음                    | **추가 권장**            |
| source file hash         | ❌                               | **추가 권장**            |
| 초기 model state 저장        | ❌                               | **추가**               |
| training history         | ❌/부족                            | **추가**               |
| manifest 저장              | ❌                               | **추가**               |
| parameter count          | 구현                              | ✅                    |
| forward shape audit      | 구현                              | ✅                    |
| checkpoint               | 구현                              | ✅                    |
| resume                   | 기존 코드에 있음                       | **가져올 가치 있음**        |
| GPU/환경 확인                | 구현                              | ✅                    |
| TF32 off                 | 기존 코드에 있음                       | **가져올 가치 있음**        |

즉 **모델 구조나 optimizer 등을 뜯어고치는 게 아니라, 실험 재현성/기록 부분을 보강하는 것이 핵심**이야.

---

# 2. 기존 train 코드에서 그대로 가져와도 좋은 것

기존 네 코드에서 새 명세와 충돌하지 않는 것만 골라보면 꽤 있다.

## ① CUDA 강제 확인

기존 코드:

```python
assert torch.cuda.is_available(), (
    "CUDA GPU가 필요합니다. "
    "Kaggle에서는 Settings -> Accelerator -> GPU를 선택하세요."
)
```

이건 새 명세와 충돌하지 않는다.

오히려 여러 명이 같은 학습 프로토콜을 수행한다면 **GPU가 없는 환경에서 CPU로 몰래 학습되는 것을 방지**하므로 좋다.

다만 새 명세에는 `device: auto`가 들어 있으므로 팀 코드가 정말 `auto`를 의도했다면 굳이 강제 CUDA로 바꾸지는 말자.

**→ 팀원 코드 유지.**

---

# 3. 기존 코드의 `TF32 off`는 가져오는 게 좋다

기존 코드에는:

```python
if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False
```

가 있었다.

새 학습 명세가

> numerical dtype = FP32

이고 여러 GPU 환경에서 최대한 동일한 학습을 하려는 목적이므로 이 부분은 **그대로 추가하는 것을 추천**한다.

특히 이전에 팀 코드에는 CUDA deterministic 설정이 있어도 cuDNN TF32까지 명시하지 않은 버전이 있었는데, 네 기존 코드가 더 명확하다.

### 추가 추천

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False
```

이건 새 학습 설계를 바꾸는 게 아니라 **재현성 강화**다.

---

# 4. 그런데 기존 코드의 AMP는 절대 가져오면 안 됨

이건 매우 중요해.

기존 코드에는:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(target)
```

그리고:

```python
scaler = torch.amp.GradScaler("cuda")
```

가 있다.

그런데 새 `Unet_only` 명세는 명확하게:

```text
AMP = False
numerical dtype = FP32
```

다.

따라서 **기존 train 코드의 AMP 부분은 가져오면 안 된다.**

팀원이 보낸 전체 코드에도 이 부분이 들어있다면 반드시 제거해야 한다.

학습은 그냥:

```python
output = model(target)

loss = criterion(
    output.float(),
    mask.float()
)
```

처럼 해야 한다.

그리고:

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

도 제거.

그 대신:

```python
loss.backward()
optimizer.step()
```

로 해야 한다.

### 결론

**AMP는 새 명세와 정면으로 충돌하므로 무조건 제거.**

---

# 5. 가장 중요한 문제: 현재 split 방식

현재 팀원 코드에는 대략:

```python
train_pool_indices = list(range(TRAIN_POOL_COUNT))

split_rng = random.Random(SEED)
split_rng.shuffle(train_pool_indices)

train_indices = train_pool_indices[:TRAIN_COUNT]

internal_val_indices = train_pool_indices[
    TRAIN_COUNT:
    TRAIN_COUNT + INTERNAL_VAL_COUNT
]
```

방식이 들어가 있다. 

숫자는 정확하다.

```text
14,824
 ├─ 13,342 train
 └─  1,482 validation

+ 1,648 final holdout
= 16,472
```

이 자체는 좋다.

**하지만 새 명세는 단순히 Seed 42로 매번 split하는 것을 요구하지 않는다.**

새 명세에서는:

> frozen split manifest를 입력으로 받고
> primary 학습은 검증된 manifest 없이는 실행하지 않는다.

가 핵심이다.

즉,

### 현재 방식

```text
파일 목록
   ↓
Seed 42 shuffle
   ↓
13,342 / 1,482
```

### 새 방식

```text
frozen split.csv
       ↓
검증
       ↓
train 13,342
validation 1,482
holdout 1,648
```

이어야 한다.

---

# 6. 그래서 기존 코드의 split 로직도 그대로 가져오면 안 됨

이건 약간 의외일 수 있는데,

**기존 train 코드도 여기서는 버리는 게 맞다.**

기존 코드의:

```python
split_rng = random.Random(SEED)
split_rng.shuffle(...)
```

방식도 새 명세 기준으로는 부족하다.

왜냐하면 **4명이 각각 자기 환경에서 파일 목록이 조금이라도 다르면 동일 split을 보장하는 장치가 없기 때문**이다.

새 명세에서 요구하는 게 바로 이 문제를 해결하기 위한:

```text
split manifest
+
SHA-256
```

이다.

---

# 7. 팀 코드에 반드시 추가하면 좋은 것: split manifest

예를 들어:

```text
split_manifest.csv
```

를 만들어서:

```csv
sample_id,target_path,mask_path,split
sample_00001.png,target/sample_00001.png,pixelILT/sample_00001.png,train
...
```

형태로 고정한다.

그리고 코드 시작 부분에서:

```python
assert len(train_samples) == 13342
assert len(val_samples) == 1482
assert len(test_samples) == 1648
```

검증.

그리고:

```python
manifest_hash = hashlib.sha256(
    normalized_manifest_text.encode("utf-8")
).hexdigest().upper()
```

같은 방식으로 hash도 기록.

이게 **새 학습 명세에서 가장 중요한 추가사항 중 하나**다.

---

# 8. `DataILT`는 그대로 유지하는 게 좋음

팀원 코드는:

```python
from lithobench.dataset import filesMaskOpt, DataILT
from lithobench.ilt.neuralilt import UNet
```

를 사용한다. 

그리고:

```python
base_dataset = DataILT(
    target_files,
    pixelilt_files,
    crop=False,
    size=INPUT_SIZE,
    cache=False,
)
```

이다. 

이 부분은 **굳이 기존 코드 방식으로 바꿀 필요 없다.**

오히려 새 명세가:

> pinned LithoBench U-Net
> 공식 dataset
> 256×256

을 요구하므로 공식 `DataILT`를 사용하는 현재 구조가 좋다.

---

# 9. H/V flip도 현재 팀 코드 유지

팀 코드의:

```python
if random.randint(0, 1) == 1:
    target = target.flip(1)
    mask = mask.flip(1)

if random.randint(0, 1) == 1:
    target = target.flip(2)
    mask = mask.flip(2)
```

은 H/V flip만 사용한다. 

그리고:

```python
crop=False
```

이므로 random crop도 없다. 

새 명세와 방향이 맞다.

### 단, 한 가지 주의

명세의 표현이 **"deterministic horizontal vertical flip"**이므로, 우리가 최종 코드에서는

> Seed 42로 난수 sequence를 통제한 H/V flip

이라는 의미를 명확하게 기록하는 게 좋다.

즉 **flip 자체가 항상 똑같다는 뜻이 아니라**, Seed/DataLoader까지 고정해서 실행 순서가 같으면 동일한 augmentation sequence가 나오도록 만드는 것이다.

---

# 10. Batch 4는 반드시 유지

팀원의 새 코드:

```python
"batch_size": 4
```

이고 새 명세와 정확히 일치한다.

기존 네 코드의:

```python
BATCH_SIZE = 2
```

는 **가져오면 안 된다.**

이건 단순한 코드 스타일 차이가 아니라 학습 trajectory에 영향을 주는 중요한 hyperparameter다.

---

# 11. BCE도 그대로 유지

팀 코드:

```python
criterion = nn.BCELoss()
```

이고 새 명세:

```text
loss = probability BCE
```

와 맞는다. 

공식 NeuralILT 출력이 probability라는 전제에서 `BCELoss`를 사용하는 현재 방식은 그대로 두는 게 좋다.

기존 네 코드도 BCE였으므로 여기서는 동일하다.

---

# 12. LR / WD / optimizer도 그대로

팀 코드:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)
```

새 명세와 정확히 일치한다. 

**건드리지 않는 게 맞다.**

---

# 13. 30 epoch도 그대로

새 명세:

```text
maximum epoch = 30
early stopping = 사용하지 않음
```

팀 코드도 30 epoch 구조다.

```python
for epoch in range(1, EPOCHS + 1):
```

따라서 이것도 그대로.

기존 코드의 `EPOCHS=30` 역시 동일하지만, 예전에 이야기했던 **20 이후 early stopping 방식은 새 명세에 포함시키면 안 된다.**

---

# 14. Best checkpoint 방식은 그대로 사용

새 명세:

> internal validation BCE 최저 checkpoint

팀 코드도:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(checkpoint, CHECKPOINT_PATH)
```

구조다. 

따라서 **이 부분은 아주 잘 맞는다.**

---

# 15. 그런데 `last.pt` + resume은 가져오는 게 좋음

이건 기존 코드에서 가져올 가치가 매우 크다.

네 기존 코드에는:

```python
RESUME_PATH = "./last (7).pt"
```

그리고:

```python
model.load_state_dict(...)
optimizer.load_state_dict(...)
```

등의 resume 로직이 있었다.

새 명세가 반드시 resume을 요구하는 것은 아니지만, 네가 실제로 Kaggle/VS Code에서 학습하면서 세션이 끊긴 경험이 있으므로 **실험 운용상 매우 유용하다.**

다만 새 명세를 깨지 않게:

```text
resume = True
```

로 하고,

**resume 대상은 같은 학습 계약에서 생성된 `last.pt`만 허용**하는 게 좋다.

예를 들어:

```text
Seed 42
Batch 4
LR 1e-4
BCE
FP32
30 epoch
같은 manifest hash
같은 model
```

인 checkpoint만 resume.

---

# 16. 초기 state 저장은 새 코드에 추가하는 게 좋음

새 명세에서:

> `best.pt`, `last.pt`, **초기 state**, history, manifest 저장

을 요구한다.

현재 팀 코드에는 `best.pt`, `last.pt`는 있지만 **초기 state 저장이 없다.**

따라서 학습 직전에:

```python
initial_state = {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
}

torch.save(
    {
        "model": "lithobench.ilt.neuralilt.UNet",
        "seed": SEED,
        "input_size": list(INPUT_SIZE),
        "parameter_count": parameter_count,
        "model_state_dict": initial_state,
    },
    "initial_state.pt"
)
```

같은 걸 추가하는 것이 좋다.

이건 특히 네가 아까 물어본

> "4명이 같은 U-Net을 학습하면 최종 모델이 완전히 같을 수 있나?"

와도 연결된다.

초기 state를 저장해두면 **최종 모델 차이가 어디서 발생했는지 추적할 수 있다.**

---

# 17. Training history도 추가

새 명세에서 history 저장을 요구한다.

현재 팀 코드가 전체적으로 history를 충분히 저장하지 않는다면:

```python
history = []
```

그리고 epoch마다:

```python
history.append({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
})
```

를 저장해서:

```text
training_history.json
```

으로 남기는 게 좋다.

이건 기존 코드에서 이미 했던 부분이라 **그대로 가져와도 된다.**

---

# 18. Manifest도 저장해야 함

여기서 중요한 건 단순히 `split.csv`를 읽는 것에서 끝나지 않는다는 거야.

최종 checkpoint와 함께:

```text
split_manifest.csv
manifest_sha256.txt
```

또는 manifest 정보를 checkpoint 안에 넣는 것이 좋다.

예:

```python
checkpoint = {
    ...
    "split_manifest_sha256": manifest_hash,
}
```

이렇게.

그러면 나중에:

```text
best.pt
```

만 봐도

> 이 모델은 어떤 split으로 학습했는가?

를 추적할 수 있다.

---

# 19. pinned LithoBench commit은 꼭 기록하는 게 좋음

새 명세에:

```text
pinned LithoBench commit
9c74e82218e377eaf6d02d113fc1ce6e36c92aa6
```

가 있다.

현재 팀 코드가 단순히:

```python
from lithobench.dataset import ...
from lithobench.ilt.neuralilt import UNet
```

만 한다면 **현재 설치된 LithoBench가 그 commit인지 자동 검증하지 않는다.**

따라서 최소한 checkpoint manifest에:

```python
"lithobench_commit":
    "9c74e82218e377eaf6d02d113fc1ce6e36c92aa6",
```

를 기록하는 게 좋다.

가능하면 실제 git repo에서 현재 commit도 검사하는 게 더 좋다.

---

# 20. 그런데 여기서 중요한 구분

**공통 평가 셀의 SHA-256과 학습 코드의 SHA-256은 목적이 다르다.**

평가 쪽에서는 이미:

```text
neuralilt.py
evaluation.py
lithosimple.txt
```

등의 hash를 검사하는 구조가 있다. 

이건 매우 좋은 구조다.

새 학습 코드에서도 가능하면:

```text
neuralilt.py hash
dataset.py hash
split_manifest hash
train.py hash
```

등을 기록하면 좋다.

하지만 **이걸 지금 당장 복잡하게 구현할 필요는 없다.**

새 명세의 핵심을 먼저 만족시키는 게 우선이다.

---

# 21. 현재 팀 코드에서 오히려 건드리지 말아야 할 것

이 부분도 중요해.

### ❌ 기존 코드에서 가져오지 말 것

| 기존 코드                             | 이유                          |
| --------------------------------- | --------------------------- |
| Batch 2                           | 새 명세는 4                     |
| `/kaggle/input/...` 절대경로          | 팀 코드의 상대경로가 더 범용적           |
| AMP                               | 새 명세와 충돌                    |
| GradScaler                        | AMP와 함께 제거                  |
| random crop                       | 새 명세와 충돌                    |
| 512×512                           | 새 명세는 256                   |
| MSE                               | 새 명세는 BCE                   |
| LR `1e-3`                         | 새 명세는 `1e-4`                |
| epoch 20 이후 early stopping        | 새 명세는 early stopping 없음     |
| 기존 random split                   | frozen manifest로 대체         |
| final 1,648을 validation으로 사용하는 구조 | final holdout은 학습에 사용하면 안 됨 |

---

# 22. 반대로 기존 코드에서 가져올 것

| 기존 코드                      | 새 코드에 적용    |
| -------------------------- | ----------- |
| CUDA availability 확인       | ✅           |
| GPU 이름 출력                  | ✅           |
| `cudnn.deterministic=True` | ✅           |
| `cudnn.benchmark=False`    | ✅           |
| cuDNN TF32 off             | ✅           |
| CUDA matmul TF32 off       | ✅           |
| `training_history.json`    | ✅           |
| `last.pt`                  | ✅           |
| resume                     | ✅           |
| epoch별 checkpoint          | ✅           |
| checkpoint에 학습 설정 기록       | ✅           |
| parameter count            | 이미 팀 코드에 있음 |
| forward shape test         | 이미 팀 코드에 있음 |

---

# 23. 최종적으로 우리가 만들 코드는 이런 구조가 가장 좋음

새로운 `train_unet_only.py`의 흐름을 딱 잡으면:

```text
① 환경 / seed 설정
        ↓
② deterministic + TF32 off
        ↓
③ LithoBench version/hash 검증
        ↓
④ split_manifest.csv 로드
        ↓
⑤ manifest hash 검증
        ↓
⑥ 13,342 / 1,482 / 1,648 검증
        ↓
⑦ 공식 DataILT
        ↓
⑧ H/V flip augmentation
        ↓
⑨ DataLoader batch=4
        ↓
⑩ 공식 NeuralILT UNet
        ↓
⑪ parameter count / shape audit
        ↓
⑫ initial_state.pt 저장
        ↓
⑬ AdamW
    lr=1e-4
    wd=1e-4
        ↓
⑭ FP32
    AMP=False
        ↓
⑮ BCE
        ↓
⑯ 30 epochs
        ↓
⑰ 매 epoch
    train BCE
    validation BCE
        ↓
⑱ val BCE 최저면 best.pt
        ↓
⑲ 매 epoch last.pt
        ↓
⑳ training_history.json
        ↓
㉑ manifest / 환경 정보 / 설정 저장
```

그리고 **학습 단계에서는**

```text
❌ LithoSim
❌ L2
❌ PVB
❌ EPE
❌ Shot
❌ 2048 simulation
❌ PPO
❌ Attention
```

전부 들어가지 않는다.

---

# 24. 내가 보기에 수정 우선순위

### 🔴 반드시 수정

**1. AMP 제거**

새 명세가 `AMP=False`이므로 가장 중요.

**2. random split → frozen split manifest**

이것도 새 명세상 중요.

---

### 🟠 강력 권장

**3. initial_state.pt 저장**

**4. training_history.json 저장**

**5. manifest hash 저장**

**6. LithoBench pinned commit 기록/검증**

**7. TF32 완전 off**

---

### 🟢 그대로 사용

**8. 공식 UNet**

**9. 256×256**

**10. Batch 4**

**11. AdamW**

**12. LR 1e-4**

**13. WD 1e-4**

**14. BCE**

**15. 30 epoch**

**16. H/V flip**

**17. crop=False**

**18. best.pt 기준 validation BCE**

**19. 14,824 / 13,342 / 1,482 / 1,648**

---

## 핵심적으로 하나만 기억하면

지금은 **기존 train.py를 다시 만드는 작업이 아니다.**

팀원의 새 코드를 기준으로:

> **"새 `Unet_only` 명세에 이미 맞는 부분은 절대 건드리지 않고, 명세에서 요구하지만 빠진 재현성·기록 부분만 보강한다."**

이 방향이 맞아.

그리고 **특히 AMP는 기존 코드에서 가져오면 안 된다.** 새 학습 설계의 `FP32 + AMP=False`가 최우선이다.

또 하나 중요한 점은 **현재 팀 코드의 `Seed=42 + shuffle`을 그대로 두면 새 명세의 frozen split 요구사항을 충족하지 못한다는 것**이야. 이건 단순한 개선사항이 아니라 **새 학습 설계에 맞추기 위해 수정해야 하는 부분**이다. 팀 코드의 현재 split 방식 자체는 13,342/1,482/1,648 숫자는 정확하지만, 고정 manifest 방식으로 바꾸는 게 맞다. 

원한다면 다음 단계에서는 **팀원이 보내준 전체 코드에서 딱 필요한 부분만 수정해서 `새 Unet_only 최종 train.py` 형태로 통째로 정리**하면 된다. 기존 코드의 좋은 부분은 가져오되, 새 명세와 충돌하는 부분은 내가 제외해서 만들면 된다.
