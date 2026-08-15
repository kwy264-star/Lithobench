부분	최종 코드
Seed	42
모델	공식 NeuralILT UNet
입력	256×256
전체 데이터	16,472
Train pool	14,824
Train	13,342
Validation	1,482
Final holdout	1,648
Train/Val split	random.Random(42).shuffle()
Frozen manifest	사용 안 함
Holdout	숫자 cell 순으로 정렬했을 때 마지막 실제 1,648개
Batch	2
LR	1e-4
Weight decay	1e-4
Optimizer	AdamW
Scheduler	없음
Loss	BCE
Augmentation	H/V flip만
Random crop	없음
Epoch	30 고정
Early stopping	없음
Best 기준	validation BCE 최저
Threshold	0.5
AMP	CUDA에서 사용
TF32	matmul/cuDNN 모두 OFF
기존 checkpoint resume	하지 않음
last.pt	매 epoch 저장
best.pt	validation 개선 시 저장
history	매 epoch 갱신
split 기록	split_manifest.csv로 기록만 함
