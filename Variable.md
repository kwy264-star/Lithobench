# 1. 논문에서 사용한 5가지 지표 정리

논문(Table 3, Evaluation Section 기준)

---

## ① L2 Loss

### 논문 의미

초기 mask를 lithography simulator에 통과시킨 후

생성된 printed pattern과

원래 target layout의 차이.

즉

```
Mask
↓

Lithography Simulator

↓

Printed Pattern

↓

Target Pattern

↓

L2 Loss
```

측정 목적

* 얼마나 target과 비슷하게 인쇄되는가

↓

낮을수록 좋음

---

### 현재 코드

```
printed_nominal

↓

target

↓

MSE
```

거의 동일

다만

현재는

```
mean()
```

사용

논문은 normalization 공개 X

---

# ② PVB (Process Variation Band)

논문 의미

공정 조건을

```
Nominal

Max Dose

Min Dose
```

로 바꾸었을 때

인쇄 경계가 얼마나 흔들리는지.

즉

```
printed_max

printed_min

↓

variation
```

측정

↓

낮을수록

공정 변화에 강함

---

논문 계산식 공개 X

---

# ③ EPE

논문 의미

Edge Placement Error

```
Target edge

↓

Printed edge

↓

거리
```

즉

Edge끼리 얼마나 떨어졌는가

↓

반도체 OPC에서 가장 중요한 지표

↓

낮을수록 좋음

# ④ Shot Count

논문 의미

전자빔 장비가

mask를 몇 번 찍어야 하는가

논문

```
adaptive-boxes
```

사용

↓

낮을수록 좋음

---

현재

run-length

proxy

논문와 가장 차이 큼

---

# ⑤ Runtime

논문 의미

모델 추론시간

```
100회 반복

↓

평균
```

↓

낮을수록 좋음

---
## STEP2

PVB 수정

현재

```python
mean((max-min)^2)
```

↓

추천

```python
torch.mean(torch.abs(printed_max - printed_min)).item()
```

이유

Process Variation Band는

band의 폭을 보는 것이므로

square보다 abs가 더 자연스럽습니다.

---

## STEP3

EPE 수정

여기가 핵심입니다.

현재

```
edge pixel 개수
```

↓

추천

```
Target edge

Printed edge

↓

KDTree

↓

평균거리
```

논문와 가장 비슷한 구현입니다.

---

## STEP4

Shot Count 수정

현재

run-length

↓

adaptive-boxes

사용

---

## STEP5

Runtime 추가

evaluate.py에서

```
100회

↓

평균
```

---

# 3. 지금 당장 수정할 것

## PVB

metrics.py

현재

```python
def compute_pvb(printed_max, printed_min):
    return torch.mean((printed_max - printed_min) ** 2).item()
```

↓

추천

```python
def compute_pvb(printed_max, printed_min):
    return torch.mean(torch.abs(printed_max - printed_min)).item()
```

---

## L2

현재

```python
return torch.mean((printed_nominal-target)**2)
```

그대로 유지

---

## Runtime

evaluate.py 마지막에

```python
import time

start=time.perf_counter()

for _ in range(100):
    model(target)

runtime=(time.perf_counter()-start)/100
```

추가

---

# 4. 지금은 수정하지 않는 것

### EPE

이건

KDTree 구현이 들어갑니다.

30줄 정도 됩니다.

---

### Shot

adaptive-boxes 연결해야 합니다.

현재 프로젝트에

```
adaptive-boxes
```

설치부터 해야 합니다.

---

# 제가 추천하는 진행 순서

오늘

✅ L2 유지

✅ PVB 수정

✅ Runtime 추가

↓

내일

KDTree EPE

↓

마지막

adaptive-boxes Shot Count

---

# 1. L2

### 현재

```python
torch.mean((printed-target)**2)
```

### 수정

**수정하지 않습니다.**

논문도 simulator output과 target의 L2를 사용합니다.

이건 그대로 유지하는 것이 맞습니다.

---

# 2. PVB

현재

```python
torch.mean((printed_max-printed_min)**2)
```

논문에서는

"PVB"

즉

공정 변화에 따른

band width입니다.

보통

```python
abs(max-min)
```

를 사용합니다.

따라서

### 수정

```python
def compute_pvb(printed_max, printed_min):
    return torch.mean(torch.abs(printed_max - printed_min)).item()
```

---

# 3. EPE

현재

```python
edge pixel 개수
```

논문

```
Target edge

↓

Printed edge

↓

거리
```

즉

거리입니다.

그래서

현재 proxy보다

KDTree 방식이 훨씬 논문에 가깝습니다.

---

### 추천 코드

```python
from scipy.spatial import cKDTree
from scipy.ndimage import sobel
import numpy as np

def compute_epe_proxy(printed_nominal, target):

    target_np = target.squeeze().cpu().numpy()
    printed_np = printed_nominal.squeeze().cpu().numpy()

    gx=sobel(target_np,axis=0)
    gy=sobel(target_np,axis=1)
    target_edge=np.argwhere(np.hypot(gx,gy)>0.3)

    gx=sobel(printed_np,axis=0)
    gy=sobel(printed_np,axis=1)
    printed_edge=np.argwhere(np.hypot(gx,gy)>0.3)

    if len(target_edge)==0 or len(printed_edge)==0:
        return 0

    tree=cKDTree(printed_edge)

    dist,_=tree.query(target_edge)

    return float(np.mean(dist))
```

이게 지금 구현 가능한 방식 중에서는 논문와 가장 가깝습니다.

---

# 4. CTHR

논문에는 없습니다.

현재

Connected Component

차이

사용.

이건 그대로 유지.

---

# 5. Shot Count

현재

run-length

입니다.

논문은

adaptive-boxes

사용.

---

지금은

adaptive-boxes를 아직 안 쓰므로

run-length보다

Connected Component 기반이

조금 더 논문에 가까워집니다.

추천

```python
def compute_shot_proxy(mask):

    binary=(mask>0.5).float().cpu().numpy().squeeze()

    _,n=cc_label(binary)

    return int(n)
```

---

# 최종 추천

| Metric | 추천                                     |
| ------ | -------------------------------------- |
| L2     | 그대로                                    |
| PVB    | abs(max-min)                           |
| EPE    | KDTree                                 |
| CTHR   | 그대로                                    |
| Shot   | adaptive-boxes 전까지 Connected Component |

---

# 제 추천

솔직히 말하면

**지금 가장 논문와 가까운 구현은**

```
L2

↓

PVB(abs)

↓

EPE(KDTree)

↓

Shot(adaptive-boxes)

↓

Runtime
```

입니다.
