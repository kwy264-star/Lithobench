# ============================================================
# 6. Split
#
# 1) 마지막 1,648개 = Final Holdout
# 2) 나머지 14,824개 = Train Pool
# 3) Train Pool을 shuffle
# 4) 13,342 = Train
# 5) 1,482 = Validation
# ============================================================

# 숫자 cell 번호 기준으로 정렬
target_files = sorted(
    target_files,
    key=cell_number
)

pixelilt_files = sorted(
    pixelilt_files,
    key=cell_number
)

# 마지막 1,648개를 먼저 Holdout으로 분리
train_pool_indices = list(
    range(TRAIN_POOL_COUNT)
)

final_test_indices = list(
    range(
        TRAIN_POOL_COUNT,
        TRAIN_POOL_COUNT + FINAL_TEST_COUNT
    )
)

# 남은 14,824개만 Train/Validation으로 섞음
split_rng = random.Random(SEED)
split_rng.shuffle(train_pool_indices)

train_indices = train_pool_indices[:TRAIN_COUNT]

internal_val_indices = train_pool_indices[
    TRAIN_COUNT:
    TRAIN_COUNT + INTERNAL_VAL_COUNT
]
