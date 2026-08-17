LithoBench MetalSet
        │
        ▼
Target layout
        │
        │ 전처리 / split / resize
        ▼
256 × 256 Target
        │
        ▼
┌───────────────────────────┐
│        U-Net              │
│                           │
│ Encoder                   │
│   ↓                       │
│ Bottleneck                │
│   ↓                       │
│ Decoder + Skip Connection │
└───────────────────────────┘
        │
        ▼
256 × 256 초기 Mask
        │
        │ 2048 physical grid로 변환
        ▼
2048 × 2048 초기 Mask
        │
        ▼
      LithoSim
        │
        ├──────────────┐
        ▼              ▼
 Nominal Print    Max / Min Print
        │              │
        └──────┬───────┘
               │
               ▼
        초기 Mask 평가
        ├── L2
        ├── PVB
        ├── EPE
        ├── Shots
        └── Runtime
               │
               ▼
       PPO의 초기 상태 / 초기 mask
               │
               ▼
          [PPO 단계]
