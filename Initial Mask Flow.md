응. 지금까지 네가 보내준 **ARLO 논문 내용 + LithoBench 관련 코드/구조 + 네 실제 U-Net 학습 코드 + 팀에서 확정한 평가 protocol**을 기준으로 보면, **PPO에 들어가기 전까지의 과정**은 아래처럼 정리하는 게 가장 정확해.

다만 먼저 하나를 바로잡아야 해.

> **“2048×2048 target을 U-Net Encoder에 넣고 → 2048×2048로 복원한다”는 흐름은 현재 네가 실제 학습한 코드와는 다르다.**

네가 실제 학습한 U-Net은 로그에서 확인했듯이 **256×256 입력 → 256×256 출력**이다. 따라서 2048은 **U-Net 추론 단계가 아니라 physical evaluation 단계에서 사용하는 해상도**로 보는 것이 현재 네 코드와 맞다.

---

# 1. 전체 흐름

PPO 이전 단계만 크게 보면:

```text
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
```

즉 **U-Net이 만들어주는 것은 “최적화된 최종 mask”가 아니라 PPO가 추가 최적화하기 위한 좋은 초기 mask**라고 이해하는 게 핵심이야.

---

# 2. ① 원본 Target Layout

먼저 MetalSet에 있는 target layout이 출발점이야.

네가 실제로 확인한 데이터는:

```text
Total matched samples = 16,472
```

이고 네 최종 split은:

```text
Training pool       = 14,824
Training            = 13,342
Internal validation = 1,482
Final holdout       = 1,648
```

이렇게 분리되어 있어.

그리고 final holdout은:

```text
cell15017 ~ cell16753
```

범위로 고정했었지.

---

# 3. ② Target 전처리

여기서 중요한 게 **모델 입력 해상도**야.

현재 네 실제 U-Net 학습에서는:

```text
INPUT_SIZE = (256, 256)
```

이었고 실제 forward test에서도:

```text
Input  = (4, 1, 256, 256)
Output = (4, 1, 256, 256)
```

가 통과했어.

따라서 현재 네 pipeline에서는:

[
T_{raw}
\rightarrow
Preprocess
\rightarrow
T_{256}
]

가 된다.

즉,

[
T_{256}\in\mathbb{R}^{1\times256\times256}
]

을 U-Net에 넣는 거야.

---

# 4. ③ U-Net Encoder

전처리된 target이 U-Net으로 들어가.

```text
Target 256×256
       │
       ▼
     Encoder
       │
       ├── Downsampling
       ├── feature extraction
       ├── spatial resolution 감소
       └── channel 증가
```

Encoder의 역할은 단순히 이미지를 작게 만드는 게 아니라 **target pattern의 공간적인 특징을 feature representation으로 추출하는 것**이야.

네가 사용한 것은 직접 만든 임의의 U-Net이 아니라:

> `lithobench.ilt.neuralilt.UNet`

즉 **공식 NeuralILT U-Net**이었고,

parameter count도:

[
7,787,905
]

개로 확인했어.

---

# 5. ④ Bottleneck

Encoder를 지나면 가장 깊은 feature representation으로 들어간다.

개념적으로:

```text
256×256
   ↓
128×128
   ↓
64×64
   ↓
32×32
   ↓
...
   ↓
Bottleneck
```

이 부분에서 target layout의 더 추상적인 feature가 표현돼.

ARLO의 경우에는 이 U-Net 구조를 기반으로 하고 **bottleneck 부근에 attention을 추가한 구조**를 사용한다는 것이 네가 앞에서 분석했던 내용이야.

하지만 현재 네가 학습 완료한 baseline은:

```text
Official NeuralILT U-Net
```

이고, PPO 이전의 초기 mask 생성 단계에서는 이 U-Net이 중요한 역할을 해.

---

# 6. ⑤ Decoder + Skip Connection

Bottleneck 이후에는 Decoder가 다시 spatial resolution을 높인다.

```text
Bottleneck
    │
    ▼
 Decoder
    │
    ├── Upsampling
    ├── feature reconstruction
    └── Encoder feature와 Skip Connection
    │
    ▼
256×256
```

Encoder에서 잃어버린 세부적인 spatial information을 skip connection을 통해 Decoder에 전달한다.

결국:

[
F_{bottleneck}
\rightarrow
Decoder
\rightarrow
M_{256}
]

가 된다.

---

# 7. ⑥ U-Net 출력 = 초기 Mask

여기가 상당히 중요해.

네 U-Net의 출력은:

```text
Input:
256 × 256 Target

        ↓ U-Net

Output:
256 × 256 Mask
```

이고 출력 범위가:

[
0\le M_{256}(x,y)\le1
]

인 **sigmoid probability map**이야.

네 forward test에서도:

```text
Output range:
0.0238 ~ 0.6592
```

처럼 0~1 범위의 값이 나왔지.

따라서 이 단계의 출력은 아직 반드시:

```text
0 또는 1
```

인 binary mask가 아니라,

```text
0.13
0.82
0.47
...
```

같은 continuous probability map이야.

---

# 8. ⑦ Threshold → 초기 Binary Mask

평가 또는 simulator에 넣기 전에 공통 threshold를 적용한다.

팀의 평가 기준에서는:

[
B(M)_{ij}
=========

\begin{cases}
1 & M_{ij}\ge0.5\
0 & M_{ij}<0.5
\end{cases}
]

로 고정했어.

따라서:

[
M_{256}
\xrightarrow{threshold=0.5}
B_{256}
]

가 된다.

이것이 **U-Net이 생성한 초기 binary mask**야.

---

# 9. ⑧ 2048×2048 Physical Evaluation Grid

그리고 여기서 네가 질문했던 **2048**이 등장해.

현재 네 학습 모델 자체는:

[
256\times256
]

이지만, physical evaluation에서는:

[
2048\times2048
]

grid를 사용하기로 팀 protocol을 정했어.

따라서 개념적으로:

[
B_{256}
\rightarrow
\operatorname{NearestResize}
\rightarrow
B_{2048}
]

이다.

팀 기준에서는 resize interpolation을 **nearest**로 고정했어.

즉:

```text
U-Net
 ↓
256×256 mask
 ↓
threshold
 ↓
256×256 binary mask
 ↓
Nearest Resize
 ↓
2048×2048 binary mask
```

야.

### 그래서 중요한 정정

네가 처음 말한:

> 2048 Target → U-Net → 2048 Mask

가 아니라,

**현재 네 실제 학습 코드 기준으로는**

> **Target → 256×256 → U-Net → 256×256 초기 mask → 2048×2048 physical evaluation**

이라고 해야 해.

---

# 10. ⑨ LithoSim에 초기 Mask 입력

2048 mask가 만들어지면 이제 **U-Net의 영역이 끝나고 physical evaluation 영역**으로 넘어가.

```text
Initial Mask 2048×2048
          │
          ▼
       LithoSim
          │
          ├──────────────┐
          ▼              ▼
      Nominal         Process Corners
                         │
                    ┌────┴────┐
                    ▼         ▼
                 Maximum    Minimum
```

LithoSim은 mask를 실제 lithography process를 거친 것처럼 simulation한다.

따라서 여기서 중요한 차이가 생겨.

### Mask

U-Net이 예측한 것

### Printed image

LithoSim을 거쳐 실제 wafer에 출력될 것으로 simulation된 것

둘은 같은 게 아니야.

---

# 11. ⑩ Nominal Print

정상적인 nominal process condition에서 출력된 결과:

[
P_{nominal}
]

을 얻는다.

그리고 이것과 target:

[
T
]

를 비교해서 L2를 계산한다.

공식 LithoBench의 기본 형태는:

[
L2_{MSE}
========

\frac1N
\sum_{i=1}^{N}
(P_{nominal,i}-T_i)^2
]

이야.

즉:

```text
Target
   │
   │
   └──────────────┐
                  ▼
             L2 comparison
                  ▲
                  │
Mask → LithoSim → Nominal
```

---

# 12. ⑪ Maximum / Minimum Process Corner

Lithography process에는 process variation이 있으니까 nominal만 보는 게 아니라 process corner를 본다.

LithoSim에서:

[
P_{max}
]

와

[
P_{min}
]

을 얻는다.

그리고 이 둘의 차이를 이용해서 PVB를 평가한다.

공식 LithoBench reference 구현에서는:

[
PVB_{MSE}
=========

\frac1N
\sum_i
(P_{max,i}-P_{min,i})^2
]

형태의 `MSE`를 사용한다.

즉:

```text
                ┌── Nominal → L2
Mask → LithoSim ┤
                ├── Maximum
                └── Minimum
                       │
                       └── PVB
```

이 구조야.

---

# 13. ⑫ EPE

다음은 **Edge Placement Error**야.

이건 단순 pixel difference가 아니야.

Target boundary와 printed boundary를 비교해서:

```text
Target edge
     │
     │ ← displacement → Printed edge
     │
     ▼
    EPE
```

를 계산한다.

공식 evaluator에는 `EPEChecker`가 있고:

```text
EPEChecker
    │
    ├── EPE inner
    ├── EPE outer
    └── EPE total
```

형태로 계산된다.

그리고 공식 인터페이스에서는:

[
EPE_{total}
===========

EPE_{inner}
+
EPE_{outer}
]

로 사용한다.

네 팀 protocol에서는 여기에 probe 관련 조건, boundary sample 상한, custom threshold 등을 명시적으로 고정해야 한다.

---

# 14. ⑬ Shot Count

마지막으로 manufacturability 측면의 proxy가 **Shot count**야.

초기 mask:

[
B_{2048}
]

를 가지고 adaptive rectangular decomposition을 수행하고,

```text
Binary Mask
    ↓
Adaptive Box Decomposition
    ↓
Rectangles
    ↓
Number of Shots
```

를 얻는다.

LithoBench의 `ShotCounter`가 이 역할을 하고, adaptive-boxes dependency를 사용한다.

따라서 Shot은:

> **pixel이 몇 개인가**

가 아니라

> **mask를 lithography writing shot으로 분해했을 때 얼마나 많은 shot이 필요한가**

를 보는 지표라고 이해하면 돼.

---

# 15. ⑭ Runtime

그리고 runtime은 별도로 측정한다.

최종 protocol에서는 적어도 구분해야 해.

### Model inference runtime

[
T_{model}
]

```text
256 Target
   ↓
U-Net
   ↓
256 Mask
```

### Physical evaluation runtime

[
T_{physical}
]

```text
2048 Mask
 ↓
LithoSim
 ↓
L2/PVB/EPE
```

### Shot runtime

[
T_{shot}
]

```text
Mask
 ↓
adaptive-boxes
 ↓
Shots
```

그리고 필요하다면:

[
T_{total}
=========

T_{model}
+
T_{physical}
+
T_{shot}
]

로 별도 보고할 수 있어.

하지만 이건 **어디까지 runtime에 포함할지 4개 코드가 동일하게 고정해야 한다.**

---

# 16. 그래서 PPO 직전의 "초기 Mask 생성"만 떼어 보면

핵심적으로는 이거야.

```text
                 ┌──────────────────────┐
                 │   Target Layout      │
                 │     2048 원본         │
                 └──────────┬───────────┘
                            │
                            ▼
                     전처리 / Resize
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Target 256 × 256     │
                 └──────────┬───────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   U-Net     │
                     │             │
                     │  Encoder    │
                     │      ↓      │
                     │ Bottleneck  │
                     │      ↓      │
                     │  Decoder    │
                     └──────┬──────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Probability Mask     │
                 │    256 × 256         │
                 │       0~1             │
                 └──────────┬───────────┘
                            │
                       threshold 0.5
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Binary Initial Mask  │
                 │    256 × 256         │
                 └──────────┬───────────┘
                            │
                    Nearest Resize
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Physical Mask        │
                 │   2048 × 2048        │
                 └──────────┬───────────┘
                            │
                            ▼
                       LithoSim
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
           Nominal        Maximum        Minimum
              │             │              │
              ▼             └──────┬───────┘
             L2                    PVB
              │
              └──────────┐
                         ▼
                       EPE
                         
        Physical Mask ─────────────→ Shot
```

---

# 17. 그리고 이 단계가 ARLO에서 왜 필요한가

여기서 **U-Net과 PPO의 역할을 분리해서 이해하는 게 중요해.**

### U-Net

[
Target \rightarrow Initial\ Mask
]

즉 target layout을 보고 **좋은 초기 mask를 한 번에 예측**한다.

### PPO

[
Initial\ Mask
\rightarrow
Mask_1
\rightarrow
Mask_2
\rightarrow
\cdots
\rightarrow
Optimized\ Mask
]

즉 U-Net이 만든 초기 mask를 출발점으로 해서 **LithoSim feedback을 이용하여 mask를 추가로 개선**한다.

그래서 전체 ARLO 구조는:

```text
                    TARGET
                      │
                      ▼
                  Preprocess
                      │
                      ▼
                 ┌─────────┐
                 │  U-Net  │
                 └────┬────┘
                      │
                      ▼
                Initial Mask
                      │
                      ▼
             ┌────────────────┐
             │ Litho Simulator│
             └───────┬────────┘
                     │
              평가 / feedback
                     │
                     ▼
                  ┌─────┐
                  │ PPO │
                  └──┬──┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Mask 1      Mask 2     Mask ...
          │          │          │
          └──────────┼──────────┘
                     ▼
                Optimized Mask
```

이라고 보면 된다.

---

## 18. 네 현재 프로젝트에서 딱 한 줄로 요약하면

**PPO 이전 단계는**

> **MetalSet의 target layout을 256×256으로 전처리하여 공식 NeuralILT U-Net에 입력하고, U-Net이 256×256 probability mask를 생성하면 이를 0.5 threshold로 binary initial mask로 만든 뒤, physical evaluation을 위해 2048×2048 grid로 변환하여 LithoSim을 수행하고, 그 결과를 기반으로 L2·PVB·EPE·Shot 및 Runtime을 평가하는 과정**이야.

그리고 여기서 **U-Net의 목적은 “최종 mask optimization”이 아니라 PPO가 refinement를 시작할 수 있는 좋은 `initial mask`를 생성하는 것**이라고 정리하면 돼.

### 단, 현재 문서에서 반드시 확정해야 할 한 가지

네 팀 문서에는 **모델 입력 512×512**라고 적혀 있지만, 네 실제 학습 결과는 **256×256**이야. 따라서 최종 발표/논문용 pipeline 그림에서는 이걸 그대로 섞으면 안 돼.

현재 네 실제 학습 코드에 근거하면:

[
\boxed{2048_{\text{target/original}}
\rightarrow256_{\text{U-Net}}
\rightarrow256_{\text{mask}}
\rightarrow2048_{\text{physical evaluation}}}
]

로 그리는 것이 현재까지 확인된 사실에 가장 잘 맞아.

다음 단계에서는 이 흐름을 기준으로 **“학습 단계(train/validation)”와 “학습 완료 후 평가 단계(final 1,648)”를 완전히 분리해서**, 네 프로젝트의 실제 코드 기준으로 입력·출력 tensor shape까지 붙인 전체 pipeline을 만들어 놓으면 돼.
