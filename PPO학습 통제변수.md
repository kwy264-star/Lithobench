응. 지금 단계에서는 **PPO를 코딩하기 전에 팀에서 PPO의 실험 정의를 먼저 고정하는 것**이 맞아. 질문 3개를 순서대로 정리할게.

---

# 1. State / Action / Episode length가 정확히 뭐야?

PPO를 우리가 하려는 문제에 대입하면 아주 쉽게 이해할 수 있어.

## ① State = PPO가 현재 보고 있는 정보

우리가 정의한 것은:

> **현재 grayscale mask (M_t)**

즉 PPO가 현재 가지고 있는 마스크를 보고 다음 행동을 결정하는 거야.

예를 들어 처음에는 U-Net이

[
M_0
]

이라는 초기 마스크를 만들어 줘.

그러면:

```text
Target
   ↓
U-Net(+Attention)
   ↓
Initial mask M0
   ↓
PPO
```

PPO가 한 번 수정하면

[
M_1 = M_0 + \Delta M_0
]

다시 수정하면

[
M_2 = M_1 + \Delta M_1
]

이런 식으로 가는 거야.

따라서 현재 state는

```text
State at t=0 : M0
State at t=1 : M1
State at t=2 : M2
...
```

라고 생각하면 된다.

### 중요한 점

현재 우리가 생각하는 가장 단순한 구조에서는 state에

```text
현재 mask
```

만 넣는다.

나중에 필요하면

```text
현재 mask
+ target
+ lithography simulation 결과
+ 이전 action
```

등으로 확장할 수도 있지만, **첫 PPO 실험에서는 state를 단순하게 유지하는 게 좋다.**

---

# ② Action = PPO가 마스크를 어떻게 수정할지

우리가 정한 게:

> **pixel-wise (\Delta M_t), 범위 ±0.1**

이거야.

즉 PPO가 각 pixel에 대해

[
\Delta M_t(x,y)
]

를 결정하는 거야.

예를 들어 어떤 pixel의 현재 값이

```text
M_t = 0.62
```

이고 PPO가

```text
ΔM_t = +0.07
```

을 선택하면

```text
M_{t+1} = 0.69
```

가 되는 식.

반대로

```text
M_t = 0.62
ΔM_t = -0.10
```

이면

```text
M_{t+1} = 0.52
```

가 된다.

일반적으로는

[
M_{t+1}
=======

clip(M_t+\Delta M_t,0,1)
]

처럼 처리하는 게 안전하다.

---

# ③ Episode length = PPO가 한 번에 몇 번 수정할 것인가

이게 상당히 중요해.

예를 들어 우리가

> episode length = 5

로 정하면:

```text
U-Net
 ↓
M0

PPO step 1
M0 → M1

PPO step 2
M1 → M2

PPO step 3
M2 → M3

PPO step 4
M3 → M4

PPO step 5
M4 → M5

        ↓
   최종 mask
        ↓
Lithography simulation
        ↓
metrics
```

이렇게 된다.

즉 **episode length는 한 초기 마스크를 PPO가 몇 번 연속 수정할 것인가**다.

---

# 2. U-Net이 만든 최초 마스크는 Binary야? Grayscale이야?

### 결론부터 말하면:

## 지금 우리가 학습한 U-Net의 출력은 **grayscale probability mask**다.

이건 굉장히 중요해.

현재 U-Net 코드에서:

```python
self.sigmoid(...)
```

를 통해 출력하고 있고,

loss도

```python
criterion = nn.BCELoss()
```

를 사용했지.

따라서 U-Net의 출력은 기본적으로

[
0 \leq M(x,y) \leq 1
]

범위의 **continuous value**야.

예를 들어:

```text
0.00
0.03
0.17
0.42
0.51
0.73
0.96
1.00
```

같은 값이 나올 수 있다.

즉:

```text
U-Net output
       ↓
grayscale / continuous mask
```

다.

---

## 그런데 `THRESHOLD = 0.5`가 있잖아?

이 부분 때문에 헷갈릴 수 있는데,

현재 코드의

```python
THRESHOLD = 0.5
```

는 **U-Net을 binary output으로 학습시키는 설정이 아니다.**

현재 학습 코드에서는 threshold를 이용해서 output을

```text
0 / 1
```

로 만드는 과정이 없다.

따라서 모델 자체는 계속

```text
0 ~ 1 continuous
```

출력을 한다.

---

# 그럼 PPO에는 이게 오히려 좋은가?

## 나는 현재 목적에서는 오히려 좋다고 본다.

우리가 하려는 PPO action이

[
\Delta M_t \in [-0.1,0.1]
]

이잖아.

만약 초기 마스크가 완전히 binary라면:

```text
0
1
```

뿐인데,

여기에 ±0.1을 적용하면

```text
0 → 0.1
1 → 0.9
```

같은 식으로 다시 continuous mask가 된다.

반면 현재 구조는 처음부터

```text
0.02
0.17
0.48
0.63
0.91
...
```

처럼 **세밀한 grayscale mask**를 가지고 시작한다.

그러면 PPO가

```text
0.63 → 0.68
0.68 → 0.61
0.61 → 0.56
```

처럼 미세 조정을 할 수 있다.

이건 **ILT/OPC를 refinement하는 PPO의 목적과 상당히 잘 맞는다.**

---

## 따라서 다시 U-Net을 만들 필요는 없다.

현재 우리가 만든:

> **U-Net + Attention → continuous grayscale mask**

구조 그대로 사용하면 된다.

다만 한 가지는 반드시 구분해야 한다.

### U-Net 학습용 target mask

데이터셋의 `pixelILT`가 binary인지 grayscale인지는 **원본 데이터의 실제 값 분포를 확인해서 확정**해야 한다.

하지만 **현재 U-Net 모델의 출력 자체는 sigmoid 때문에 continuous [0,1]**이라는 것은 확실하다.

---

# 3. PPO에서 팀원끼리 반드시 통제해야 할 변수

이게 사실 가장 중요하다.

지금 팀에서 예를 들어

> "나는 L2 개선을 중심으로 PPO를 돌려볼게."

> "나는 PVB 개선을 중심으로 PPO를 돌려볼게."

라고 한다면,

**reward만 다르게 하고 나머지는 최대한 동일하게 만들어야 실험 결과를 비교할 수 있다.**

---

# PPO 실험에서 고정해야 할 것들

나는 아래처럼 **실험 프로토콜을 하나 만들어서 팀 전체가 공유하는 것**을 추천한다.

| 항목                      |                권장 1차 설정 | 중요도   |
| ----------------------- | ----------------------: | ----- |
| Initial mask            |    동일 U-Net(+Attention) | ★★★★★ |
| State                   | 현재 grayscale mask (M_t) | ★★★★★ |
| Action                  |           pixel-wise ΔM | ★★★★★ |
| Action range            |                    ±0.1 | ★★★★★ |
| Episode length          |                    5~10 | ★★★★★ |
| Reward                  |            실험 목적에 따라 변경 | ★★★★★ |
| Lithography simulator   |                      동일 | ★★★★★ |
| Simulator config        |                      동일 | ★★★★★ |
| Metric resolution       |                      동일 | ★★★★★ |
| Train images            |                      동일 | ★★★★★ |
| Validation images       |                      동일 | ★★★★★ |
| Final holdout           |                   사용 금지 | ★★★★★ |
| Random seed             |                      동일 | ★★★★★ |
| PPO architecture        |                      동일 | ★★★★  |
| γ (discount)            |                      동일 | ★★★★  |
| GAE λ                   |                      동일 | ★★★★  |
| clip ε                  |                      동일 | ★★★★  |
| Actor LR                |                      동일 | ★★★★  |
| Critic LR               |                      동일 | ★★★★  |
| Entropy coefficient     |                      동일 | ★★★★  |
| Value coefficient       |                      동일 | ★★★★  |
| rollout length          |                      동일 | ★★★★  |
| minibatch size          |                      동일 | ★★★★  |
| PPO update epochs       |                      동일 | ★★★★  |
| gradient clipping       |                      동일 | ★★★   |
| advantage normalization |                      동일 | ★★★   |
| action clipping         |                      동일 | ★★★★  |

---

# 특히 중요한 PPO 변수

## ① Reward

이게 우리가 실험에서 바꿀 핵심 변수야.

예를 들어:

### Experiment A — L2 집중

[
R=-L2_{norm}
]

### Experiment B — 균형형

[
R=-(L2_{norm}+0.5PVB_{norm})
]

### Experiment C — PVB 집중

[
R=-(L2_{norm}+1.0PVB_{norm})
]

이렇게.

**이 세 실험에서 다른 PPO 조건은 동일하게 유지하는 게 핵심이다.**

---

# ② γ (gamma)

discount factor.

현재 reward와 미래 reward 중 무엇을 얼마나 중요하게 볼지 결정한다.

예:

```text
γ = 0.99
```

이면 미래 reward도 상당히 중요하게 보는 것.

우리 문제에서는 PPO가 한 번 수정하고 끝나는 게 아니라

```text
M0 → M1 → M2 → M3 → ...
```

로 이어지므로 중요하다.

**첫 실험에서는 0.99 정도로 고정**하는 것을 추천한다.

---

# ③ GAE λ

PPO에서 advantage를 계산할 때 사용하는 값.

보통:

```text
λ = 0.95
```

정도로 시작한다.

첫 실험에서는 이것도 **팀 전체 동일하게 고정**.

---

# ④ PPO clip ε

PPO의 핵심 변수.

보통:

```text
clip ε = 0.2
```

정도로 시작.

너무 크게 바꾸면 policy가 한 번에 너무 많이 변할 수 있다.

따라서 이것도 고정.

---

# ⑤ Actor learning rate

Policy를 얼마나 빠르게 학습할지 결정.

예:

```text
actor_lr = 3e-4
```

---

# ⑥ Critic learning rate

현재 state에서 앞으로 받을 reward를 예측하는 value network의 학습률.

예:

```text
critic_lr = 1e-4
```

혹은 actor와 동일하게 둘 수도 있다.

**중요한 건 팀 전체가 같은 설정을 쓰는 것.**

---

# ⑦ Entropy coefficient

PPO가 너무 빨리 한 가지 행동만 선택하지 않도록 탐색을 유지하는 역할.

예:

```text
entropy_coef = 0.01
```

처음에는 작은 값으로 두는 게 무난하다.

---

# ⑧ Value coefficient

PPO loss에서 critic의 영향을 얼마나 줄지 결정.

예:

```text
value_coef = 0.5
```

---

# ⑨ PPO update epochs

수집한 rollout 데이터를 몇 번 반복해서 학습할지.

예:

```text
PPO epochs = 4
```

---

# ⑩ Rollout / minibatch

예를 들어:

```text
episode
  ↓
5 steps
  ↓
trajectory 수집
  ↓
PPO update
```

또는 여러 이미지에서 trajectory를 모아서

```text
N samples × 5 steps
```

를 만든 다음 PPO를 업데이트할 수 있다.

이것 역시 팀원끼리 통일해야 한다.

---

# 그리고 아주 중요한 것 하나

## Reward를 "절대 metric"으로 할지 "변화량"으로 할지도 정해야 한다.

예를 들어:

### 방식 A

현재 상태의 성능을 직접 reward:

[
R_t=-L2_t
]

### 방식 B

얼마나 개선했는지를 reward:

[
R_t=L2_{t-1}-L2_t
]

두 번째 방식이면

```text
L2가 감소
→ positive reward

L2가 증가
→ negative reward
```

가 된다.

PVB도 마찬가지.

나는 **우리가 PPO를 처음 구현할 때는 이 부분을 팀에서 반드시 먼저 합의하고 시작하는 것**을 강하게 추천한다. Reward 설계가 달라지면 PPO의 학습 목표 자체가 달라지기 때문이다.

---

# 그리고 데이터 분할도 이렇게 해야 한다

현재 우리가 U-Net에서 만든 분할을 그대로 생각하면:

```text
전체 MetalSet
16,472
│
├── Train
│   13,342
│
├── Internal Validation
│   1,482
│
└── Final Holdout
    1,648
```

PPO에서는:

### PPO training

```text
13,342 train
```

사용 가능.

### PPO model/setting 선택

```text
1,482 validation
```

사용.

### 최종 결과

```text
1,648 holdout
```

**마지막까지 건드리지 않음.**

---

# 우리가 최종적으로 가져가야 할 구조

전체 파이프라인을 그리면 이거야.

```text
                    ┌──────────────────┐
Target image ──────►│ U-Net + Attention│
                    └────────┬─────────┘
                             │
                             ▼
                     Initial mask M0
                     (grayscale 0~1)
                             │
                             ▼
                    ┌────────────────┐
                    │      PPO       │
                    │   policy agent │
                    └───────┬────────┘
                            │
                ΔM0 ∈ [-0.1, +0.1]
                            │
                            ▼
                       Mask M1
                            │
                       LithoSim
                            │
                            ▼
                  L2 / PVB / EPE / Shot
                            │
                         Reward
                            │
                            ▼
                       PPO update
                            │
                            ▼
                       Mask M2
                            │
                           ...
                            │
                            ▼
                    최종 optimized mask
                            │
                            ▼
                 최종 Lithography simulation
                            │
                            ▼
                  L2 / PVB / EPE / Shot
```

그리고 **최종 holdout 1,648개는 이 전체 설계가 끝날 때까지 건드리지 않는 것**이 가장 중요하다.

---

## 내가 지금 팀에서 가장 먼저 고정할 12개

코드를 작성하기 전에 아래를 문서로 확정하면 좋다.

```text
1. Initial mask
   = 동일한 U-Net + Attention

2. State
   = current grayscale mask Mt

3. Action
   = pixel-wise ΔMt

4. Action range
   = [-0.1, +0.1]

5. Mask update
   = clip(Mt + ΔMt, 0, 1)

6. Episode length
   = 5 또는 10

7. Reward definition
   = 팀 실험 목적에 따라 결정

8. γ
   = 0.99

9. GAE λ
   = 0.95

10. PPO clip ε
    = 0.2

11. PPO optimizer / LR / batch / update epochs
    = 전 실험 동일

12. Dataset split
    = Train 13,342 / Val 1,482 / Holdout 1,648
```

**특히 7번 Reward만 바꾸고 A/B/C 실험을 하는 것이 우리가 "L2 중심 PPO가 정말 L2를 더 많이 개선하는가?", "PVB 중심 PPO가 정말 PVB를 더 많이 개선하는가?"를 가장 깔끔하게 검증하는 방법**이야.

그리고 네가 올린 그림의 **“U-Net Frozen / PPO만 학습”**도 현재 단계에서는 그대로 유지하는 게 맞다. U-Net(+Attention)을 PPO와 동시에 다시 학습시키면 PPO 실험에서 무엇이 성능을 변화시켰는지 분리하기 어려워진다.
