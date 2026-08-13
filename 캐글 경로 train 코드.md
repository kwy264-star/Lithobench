응. **이번에는 바로 학습을 시작해도 되도록**, 네가 말한 과거 Kaggle 문제까지 반영해서 `Unet_only` 최종 `train.py`를 구성하는 게 맞아.

그리고 기준은 네가 강조한 대로 **새 `Unet_only` 학습설계가 무조건 1순위**이고, 이전 코드에서 필요한 부분만 가져왔어.

[최종 `train_unet_only_kaggle.py` 다운로드](sandbox:/mnt/data/train_unet_only_kaggle.py)

### 이번 코드에서 특히 반영한 것

| 항목                      | 최종 코드                                |
| ----------------------- | ------------------------------------ |
| 모델                      | 공식 `lithobench.ilt.neuralilt.UNet`   |
| 입력/출력                   | 256×256 / 1채널                        |
| Train pool              | 14,824                               |
| Train                   | 13,342                               |
| Internal validation     | 1,482                                |
| Final holdout           | 1,648                                |
| Split                   | **`random.Random(42).shuffle()` 유지** |
| Batch                   | **4**                                |
| Epoch                   | **30**                               |
| Loss                    | **Probability BCE**                  |
| Optimizer               | **AdamW**                            |
| LR                      | **1e-4**                             |
| Weight decay            | **1e-4**                             |
| Scheduler               | 없음                                   |
| Early stopping          | 없음                                   |
| Augmentation            | H/V flip만                            |
| AMP                     | **OFF**                              |
| dtype                   | **FP32**                             |
| Seed                    | **42**                               |
| Resume                  | **기본 OFF → 새 모델부터 시작**               |
| `last.pt`               | **매 epoch 저장**                       |
| `best.pt`               | **validation BCE가 개선될 때 저장**         |
| `initial_state.pt`      | 학습 시작 전 초기 모델 저장                     |
| `training_history.json` | **매 epoch 갱신**                       |
| `split_manifest.csv`    | 이번 실행에서 실제 선택된 이미지 기록                |
| `run_manifest.json`     | 학습 조건 기록                             |

특히 `last.pt`와 `best.pt`는 **서로 별개의 파일**로 저장된다. 이전 코드처럼 `last.pt`는 매 epoch마다 갱신하고, `best.pt`는 validation BCE가 가장 좋았을 때의 모델만 보존한다. 기존 코드에서도 매 epoch `last.pt` 저장 구조가 있었기 때문에 이 부분은 유지·강화했다. 

### Kaggle 경로도 처리해놨어

Kaggle에서는 자동으로

```text
/kaggle/input/datasets/kangkyeongdeok/metalset/MetalSet/MetalSet
```

을 데이터 경로로 사용하고,

```text
/kaggle/working/
```

에 결과를 저장하도록 했다.

이 경로는 네가 이전 학습에서 실제 사용했던 경로와 동일하다. 

따라서 Kaggle에서 **별도로 `./work/MetalSet` 같은 로컬 경로로 수정할 필요가 없다.**

### 그리고 네가 겪었던 가장 중요한 문제

이번에는 단순히

```python
!python train.py
```

만 실행하는 방식에 의존하지 않도록 **코드 자체에서 결과 저장 경로를 `/kaggle/working`으로 명확하게 잡았다.**

학습이 시작되면:

```text
Epoch 1
 ↓
/kaggle/working/last.pt 저장
/kaggle/working/best.pt 저장
/kaggle/working/training_history.json 갱신
 ↓
Epoch 2
 ↓
last.pt 갱신
best.pt 필요시 갱신
history 갱신
 ↓
...
```

이렇게 된다.

또한 매 epoch 끝에 실제 파일이 존재하는지:

```python
assert CHECKPOINT_LAST.is_file()
assert CHECKPOINT_BEST.is_file()
```

까지 확인하도록 해놨다.

그래서 **경로가 잘못돼 저장이 안 되는 상황을 학습이 한참 진행된 뒤에 발견하는 게 아니라, 첫 epoch 종료 시점에 바로 오류로 확인할 수 있다.**

### 한 가지 중요한 점

이번 학습은 **완전히 새 학습**이므로:

```python
RESUME = False
```

로 해놨다.

따라서 예전에 가지고 있던

```text
last (7).pt
best.pt
last.pt
```

등은 **절대 읽지 않는다.**

모델을 새로 생성하고 → 초기 상태를 `initial_state.pt`로 저장하고 → epoch 1부터 학습한다.

그리고 `split_manifest.csv`는 **split 방식을 frozen manifest로 바꾼 것이 아니다.** 네가 결정한 `random.Random(42).shuffle()` 결과를 이번 실행에서 실제로 어떤 이미지가 train/validation/holdout에 들어갔는지 기록하는 **사후 기록용 파일**이다.

즉,

> **split 방식은 팀원이 만든 코드 그대로 유지 + 실제 split 결과만 CSV로 기록**

이다.

---

**결론적으로 지금 단계에서는 이 `train_unet_only_kaggle.py`를 새 `Unet_only` 학습용 최종본으로 사용하면 된다.**

다만 Kaggle에서 실행하기 전에 **LithoBench 코드 자체가 Kaggle Dataset/Notebook 환경에서 `from lithobench.dataset import ...`로 정상 import되는지만 한 번 확인**하면 된다. 그 부분까지 통과하면 학습을 시작하면 돼.



from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0. PRIMARY CONFIG
#    New Unet_only learning specification is the 1st priority.
#    Split policy intentionally follows the team's agreed
#    random.Random(42).shuffle() method.
# ============================================================

SEED = 42

INPUT_SIZE = (256, 256)

TRAIN_POOL_COUNT = 14_824
TRAIN_COUNT = 13_342
INTERNAL_VAL_COUNT = 1_482
FINAL_TEST_COUNT = 1_648

EPOCHS = 30
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

THRESHOLD = 0.5

# New training starts from scratch.
# Set True only if you intentionally want to resume a checkpoint.
RESUME = False
RESUME_PATH = None

# Kaggle paths
KAGGLE_DATA_ROOT = Path(
    "/kaggle/input/datasets/kangkyeongdeok/metalset/MetalSet/MetalSet"
)
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working")

# Local fallback (for testing outside Kaggle)
LOCAL_DATA_ROOT = Path("./work/MetalSet")
LOCAL_OUTPUT_ROOT = Path("./outputs")

# Augmentation required by the primary specification
AUGMENTATION = "deterministic_horizontal_vertical_flip"

# Primary specification: no AMP, FP32
AMP = False
DTYPE = "float32"

NUM_WORKERS = 0
PIN_MEMORY = True


# ============================================================
# 1. Output / dataset path
# ============================================================

IS_KAGGLE = Path("/kaggle").exists()

if IS_KAGGLE:
    DATA_ROOT = KAGGLE_DATA_ROOT
    OUTPUT_ROOT = KAGGLE_OUTPUT_ROOT
else:
    DATA_ROOT = LOCAL_DATA_ROOT
    OUTPUT_ROOT = LOCAL_OUTPUT_ROOT

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_BEST = OUTPUT_ROOT / "best.pt"
CHECKPOINT_LAST = OUTPUT_ROOT / "last.pt"
INITIAL_STATE = OUTPUT_ROOT / "initial_state.pt"
HISTORY_PATH = OUTPUT_ROOT / "training_history.json"
SPLIT_MANIFEST = OUTPUT_ROOT / "split_manifest.csv"
RUN_MANIFEST = OUTPUT_ROOT / "run_manifest.json"

print("=" * 80)
print("ARLO DUV / Unet_only PRIMARY TRAINING")
print("=" * 80)
print("Environment :", "Kaggle" if IS_KAGGLE else "Local")
print("DATA_ROOT   :", DATA_ROOT)
print("OUTPUT_ROOT :", OUTPUT_ROOT)
print("=" * 80)


# ============================================================
# 2. Seed / deterministic settings
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


set_seed()


# ============================================================
# 3. Device
# ============================================================

if IS_KAGGLE:
    assert torch.cuda.is_available(), (
        "Kaggle GPU가 필요합니다. "
        "Settings -> Accelerator -> GPU에서 GPU를 선택하세요."
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 4. Official LithoBench dataset / official NeuralILT UNet
# ============================================================

from lithobench.dataset import filesMaskOpt, DataILT
from lithobench.ilt.neuralilt import UNet


# ============================================================
# 5. Dataset files
# ============================================================

assert DATA_ROOT.is_dir(), (
    f"MetalSet 경로를 찾을 수 없습니다:\n{DATA_ROOT}\n"
    "Kaggle Dataset 경로 또는 LOCAL_DATA_ROOT를 확인하세요."
)

glp_files, target_files, pixelilt_files = filesMaskOpt(str(DATA_ROOT))

assert len(target_files) == len(pixelilt_files), (
    "target과 pixelILT 개수가 다릅니다."
)

assert len(target_files) == TRAIN_POOL_COUNT + FINAL_TEST_COUNT, (
    f"전체 sample 수가 예상과 다릅니다: "
    f"{len(target_files)} != {TRAIN_POOL_COUNT + FINAL_TEST_COUNT}"
)

print("Total matched samples:", len(target_files))
print("Train pool          :", TRAIN_POOL_COUNT)
print("Final holdout        :", FINAL_TEST_COUNT)


# ============================================================
# 6. Split
#
# Team-agreed policy:
#   1) first 14,824 samples = train pool
#   2) random.Random(42).shuffle(train_pool_indices)
#   3) first 13,342 = train
#   4) remaining 1,482 = internal validation
#   5) last 1,648 = final holdout
#
# This split is generated once at program start, not every epoch.
# ============================================================

train_pool_indices = list(range(TRAIN_POOL_COUNT))

final_test_indices = list(
    range(
        TRAIN_POOL_COUNT,
        TRAIN_POOL_COUNT + FINAL_TEST_COUNT,
    )
)

split_rng = random.Random(SEED)
split_rng.shuffle(train_pool_indices)

train_indices = train_pool_indices[:TRAIN_COUNT]
internal_val_indices = train_pool_indices[
    TRAIN_COUNT:TRAIN_COUNT + INTERNAL_VAL_COUNT
]

assert len(train_indices) == TRAIN_COUNT
assert len(internal_val_indices) == INTERNAL_VAL_COUNT
assert len(final_test_indices) == FINAL_TEST_COUNT

assert set(train_indices).isdisjoint(internal_val_indices)
assert set(train_indices).isdisjoint(final_test_indices)
assert set(internal_val_indices).isdisjoint(final_test_indices)

print("=" * 80)
print("DATA SPLIT")
print("Training pool       :", TRAIN_POOL_COUNT)
print("Training            :", TRAIN_COUNT)
print("Internal validation :", INTERNAL_VAL_COUNT)
print("Final holdout       :", FINAL_TEST_COUNT)
print("=" * 80)


# ============================================================
# 7. Save actual split manifest
#
# This does NOT replace random.Random(42).shuffle().
# It records exactly which files were used in this run.
# ============================================================

with SPLIT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "sample_id", "target_path", "pixelilt_path", "split"])

    for idx in train_indices:
        writer.writerow([
            idx,
            Path(target_files[idx]).name,
            str(target_files[idx]),
            str(pixelilt_files[idx]),
            "train",
        ])

    for idx in internal_val_indices:
        writer.writerow([
            idx,
            Path(target_files[idx]).name,
            str(target_files[idx]),
            str(pixelilt_files[idx]),
            "internal_validation",
        ])

    for idx in final_test_indices:
        writer.writerow([
            idx,
            Path(target_files[idx]).name,
            str(target_files[idx]),
            str(pixelilt_files[idx]),
            "final_holdout_test",
        ])

print("Split manifest saved:", SPLIT_MANIFEST)


# ============================================================
# 8. Official DataILT + H/V flip augmentation
# ============================================================

base_dataset = DataILT(
    target_files,
    pixelilt_files,
    crop=False,
    size=INPUT_SIZE,
    cache=False,
)


class FlipOnlyDataset(Dataset):
    def __init__(self, dataset, indices, augmentation=False):
        self.dataset = dataset
        self.indices = list(indices)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        target, mask = self.dataset[real_idx]

        target = target.float()
        mask = mask.float()

        if self.augmentation:
            # H flip
            if random.randint(0, 1) == 1:
                target = target.flip(-1)
                mask = mask.flip(-1)

            # V flip
            if random.randint(0, 1) == 1:
                target = target.flip(-2)
                mask = mask.flip(-2)

        return target, mask


train_dataset = FlipOnlyDataset(
    base_dataset,
    train_indices,
    augmentation=True,
)

val_dataset = FlipOnlyDataset(
    base_dataset,
    internal_val_indices,
    augmentation=False,
)


# ============================================================
# 9. DataLoader
# ============================================================

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

val_generator = torch.Generator()
val_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY if DEVICE.type == "cuda" else False,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    generator=val_generator,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY if DEVICE.type == "cuda" else False,
    drop_last=False,
)

print("Train batches:", len(train_loader))
print("Val batches  :", len(val_loader))


# ============================================================
# 10. Official NeuralILT U-Net
# ============================================================

model = UNet()

parameter_count = sum(p.numel() for p in model.parameters())
EXPECTED_PARAMETERS = 7_787_905

assert parameter_count == EXPECTED_PARAMETERS, (
    f"공식 UNet 파라미터 수 불일치: "
    f"{parameter_count} != {EXPECTED_PARAMETERS}"
)

model = model.to(DEVICE)

# Forward audit
probe = torch.zeros(
    2,
    1,
    INPUT_SIZE[0],
    INPUT_SIZE[1],
    device=DEVICE,
)

with torch.inference_mode():
    probe_output = model(probe)

assert probe_output.shape == (
    2,
    1,
    INPUT_SIZE[0],
    INPUT_SIZE[1],
)
assert torch.isfinite(probe_output).all()
assert 0.0 <= probe_output.min().item() <= probe_output.max().item() <= 1.0

del probe, probe_output

print("Official NeuralILT UNet")
print("Parameter count:", parameter_count)
print("Forward shape audit: PASS")


# ============================================================
# 11. Save initial model state
#     New training starts from this state.
# ============================================================

initial_state_cpu = {
    k: v.detach().cpu().clone()
    for k, v in model.state_dict().items()
}

torch.save(
    {
        "model_state_dict": initial_state_cpu,
        "seed": SEED,
        "model": "lithobench.ilt.neuralilt.UNet",
        "parameter_count": parameter_count,
        "input_size": list(INPUT_SIZE),
        "dtype": DTYPE,
        "amp": AMP,
    },
    INITIAL_STATE,
)

print("Initial state saved:", INITIAL_STATE)


# ============================================================
# 12. Loss / optimizer
# ============================================================

criterion = nn.BCELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Primary specification: AMP OFF.
assert AMP is False


# ============================================================
# 13. Optional resume
#     Default is False, so old checkpoints are never loaded
#     accidentally during the new primary run.
# ============================================================

start_epoch = 1
best_val_loss = float("inf")

if RESUME:
    assert RESUME_PATH is not None
    resume_path = Path(RESUME_PATH)
    assert resume_path.is_file(), f"Resume checkpoint not found: {resume_path}"

    ckpt = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_val_loss = float(ckpt["best_val_loss"])

    print("Resumed from:", resume_path)
    print("Start epoch :", start_epoch)
    print("Best val BCE:", best_val_loss)
else:
    print("NEW RUN: old checkpoint is NOT loaded.")


# ============================================================
# 14. Validation
# ============================================================

@torch.no_grad()
def validate() -> float:
    model.eval()

    total_loss = 0.0
    total_count = 0

    for target, mask in val_loader:
        target = target.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        # FP32 only: AMP disabled by primary specification.
        output = model(target)

        output = torch.clamp(
            output.float(),
            min=1e-7,
            max=1.0 - 1e-7,
        )

        loss = criterion(output, mask.float())

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


# ============================================================
# 15. Run manifest
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


run_manifest = {
    "protocol": "ARLO-DUV-Unet-only-primary",
    "seed": SEED,
    "input_size": list(INPUT_SIZE),
    "train_pool_count": TRAIN_POOL_COUNT,
    "training_count": TRAIN_COUNT,
    "internal_validation_count": INTERNAL_VAL_COUNT,
    "final_holdout_count": FINAL_TEST_COUNT,
    "split_method": "random.Random(42).shuffle(train_pool_indices)",
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": None,
    "loss": "probability_bce",
    "epochs": EPOCHS,
    "early_stopping": False,
    "amp": False,
    "dtype": "float32",
    "augmentation": AUGMENTATION,
    "threshold": THRESHOLD,
    "model": "lithobench.ilt.neuralilt.UNet",
    "parameter_count": parameter_count,
    "data_root": str(DATA_ROOT.resolve()),
    "output_root": str(OUTPUT_ROOT.resolve()),
    "split_manifest": str(SPLIT_MANIFEST.resolve()),
    "initial_state": str(INITIAL_STATE.resolve()),
}

with RUN_MANIFEST.open("w", encoding="utf-8") as f:
    json.dump(run_manifest, f, ensure_ascii=False, indent=2)

print("Run manifest saved:", RUN_MANIFEST)


# ============================================================
# 16. Training
# ============================================================

history = []

print("=" * 80)
print("TRAINING START")
print("30 epochs / batch 4 / AdamW / BCE / FP32 / AMP OFF")
print("=" * 80)

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_start = time.time()

    model.train()

    total_train_loss = 0.0
    total_train_count = 0

    for target, mask in train_loader:
        target = target.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Primary specification: FP32, AMP OFF.
        output = model(target)

        output = torch.clamp(
            output.float(),
            min=1e-7,
            max=1.0 - 1e-7,
        )

        loss = criterion(output, mask.float())

        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_count += batch_size

    train_loss = total_train_loss / total_train_count
    val_loss = validate()

    epoch_time = time.time() - epoch_start

    record = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "epoch_time_sec": float(epoch_time),
    }
    history.append(record)

    print(
        f"[Epoch {epoch:02d}/{EPOCHS}] "
        f"train_loss={train_loss:.6f} | "
        f"val_loss={val_loss:.6f} | "
        f"time={epoch_time / 60:.1f} min"
    )

    # --------------------------------------------------------
    # LAST: save after EVERY epoch.
    # This is the crash/session-interruption checkpoint.
    # --------------------------------------------------------
    last_checkpoint = {
        "checkpoint_type": "last",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "seed": SEED,
        "input_size": list(INPUT_SIZE),
        "train_pool_count": TRAIN_POOL_COUNT,
        "train_count": TRAIN_COUNT,
        "internal_val_count": INTERNAL_VAL_COUNT,
        "final_test_count": FINAL_TEST_COUNT,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "probability_bce",
        "amp": False,
        "dtype": "float32",
        "augmentation": AUGMENTATION,
        "threshold": THRESHOLD,
        "model": "lithobench.ilt.neuralilt.UNet",
        "parameter_count": parameter_count,
        "split_manifest": str(SPLIT_MANIFEST.resolve()),
    }

    torch.save(last_checkpoint, CHECKPOINT_LAST)

    # --------------------------------------------------------
    # BEST: only when internal validation BCE improves.
    # --------------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_checkpoint = dict(last_checkpoint)
        best_checkpoint["checkpoint_type"] = "best"
        best_checkpoint["best_val_loss"] = best_val_loss

        torch.save(best_checkpoint, CHECKPOINT_BEST)

        print(
            f"  -> BEST saved: {CHECKPOINT_BEST} "
            f"(val_BCE={best_val_loss:.6f})"
        )

    # --------------------------------------------------------
    # Save history after EVERY epoch as well.
    # --------------------------------------------------------
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Explicit existence check so a path failure is visible
    # immediately in the Kaggle cell output.
    assert CHECKPOINT_LAST.is_file(), (
        f"last.pt 저장 실패: {CHECKPOINT_LAST}"
    )
    assert CHECKPOINT_BEST.is_file(), (
        f"best.pt 저장 실패: {CHECKPOINT_BEST}"
    )

    print(
        f"  -> last.pt updated | "
        f"best.pt exists | "
        f"output={OUTPUT_ROOT}"
    )


# ============================================================
# 17. Final summary
# ============================================================

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("Best validation BCE:", best_val_loss)
print("Best checkpoint    :", CHECKPOINT_BEST)
print("Last checkpoint    :", CHECKPOINT_LAST)
print("Initial state      :", INITIAL_STATE)
print("History            :", HISTORY_PATH)
print("Split manifest     :", SPLIT_MANIFEST)
print("Run manifest       :", RUN_MANIFEST)
print("Model              : official NeuralILT UNet")
print("Parameter count    :", parameter_count)
print("=" * 80)

# Final file existence checks
for path in [
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    INITIAL_STATE,
    HISTORY_PATH,
    SPLIT_MANIFEST,
    RUN_MANIFEST,
]:
    assert path.is_file(), f"필수 결과 파일이 없습니다: {path}"

print("All required output files: PASS")

