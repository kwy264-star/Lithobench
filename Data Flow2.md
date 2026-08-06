좋아. 이 부분은 발표할 때 **"코드를 그대로 읽는 것이 아니라, 중요한 코드만 집어서 왜 필요한지 설명"** 하는 것이 가장 이해하기 쉽다.

추천하는 순서는 아래와 같다.

> **dataset.py → model.py → train.py → real_litho_sim.py → evaluate.py**

이 순서가 실제 데이터가 흘러가는 순서와 동일하다.

---

# 1. dataset.py

(데이터를 읽는 부분)

가장 먼저 실행되는 파일이다.

---

### (1) Dataset 클래스 선언

```python
class LithoDataset(Dataset):
```

설명

> PyTorch에서 데이터를 사용할 수 있도록 만든 클래스이다.
>
> 이후 train.py에서 DataLoader가 이 클래스를 이용해 데이터를 하나씩 가져온다.

---

### (2) 이미지 읽기

대부분 이런 코드가 있다.

```python
target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
```

설명

여기서는

MetalSet 안의

```
target/
mask/
```

이미지를 읽는다.

즉

```
target
↓

mask
```

한 쌍을 만든다.

---

### (3) Tensor 변환

```python
target = torch.from_numpy(target).float()/255.
```

설명

UNet은 numpy가 아니라

```
Tensor
```

를 입력으로 받는다.

그래서

이미지를 Tensor로 변환한다.

---

### (4) return

```python
return target, mask
```

설명

결국 Dataset은

```
(target,
mask)
```

를 train.py에 전달한다.

---

여기까지가

> 데이터를 준비하는 단계

이다.

---

# 2. model.py

(UNet)

이 부분이 발표의 핵심이다.

---

## (1) DoubleConv

대부분 이런 코드이다.

```python
nn.Conv2d(...)
nn.ReLU(...)
nn.Conv2d(...)
nn.ReLU(...)
```

설명

DoubleConv는

논문에서 말하는

```
Conv
↓

ReLU

↓

Conv

↓

ReLU
```

블록이다.

즉

패턴의 특징(feature)을 추출한다.

여기서

선폭

모서리

라인

패턴 형태

등을 학습한다.

---

## (2) MaxPooling

```python
nn.MaxPool2d(2)
```

설명

이 부분은

이미지 크기를

```
256

↓

128

↓

64

↓

32
```

처럼 줄인다.

왜 줄이냐?

큰 구조를 보기 위해서이다.

작은 노이즈보다

전체 패턴 구조를 이해하게 된다.

---

## (3) Bottleneck

여기다.

```python
self.bottleneck = DoubleConv(...)
```

설명

UNet의 가장 깊은 층이다.

가장 압축된 feature를 저장한다.

예를 들면

```
256

↓

128

↓

64

↓

32

↓

16

↓

Bottleneck
```

여기서는

전체 패턴을 가장 추상적으로 표현한다.

---

### 중요한 점

**이건 Attention이 아니다.**

많은 학생들이 헷갈리는데

Bottleneck은

그냥

가장 깊은 Conv Layer이다.

Attention과는 다른 개념이다.

---

## (4) UpSampling

```python
ConvTranspose2d(...)
```

설명

줄였던 크기를 다시 키운다.

```
16

↓

32

↓

64

↓

128

↓

256
```

---

## (5) Skip Connection

대부분

```python
torch.cat(...)
```

이 있다.

설명

Encoder에서 얻은

세밀한 정보를

Decoder에 그대로 전달한다.

그래서

모서리 정보가 살아남는다.

---

## (6) 마지막 출력

```python
Conv2d(...,1,...)
```

설명

최종적으로

```
1채널

Mask
```

를 출력한다.

---

# Attention은 어디 있나?

너희 코드에는

아마 이런 코드가 있다.

```python
if use_attention:
```

또는

```python
AttentionGate(...)
```

혹은

```python
SelfAttention(...)
```

---

train에서는

```
--attention
```

옵션을 줄 때만

실행된다.

---

이번 실험은

```
UNet Base
```

였으므로

Attention은

사용되지 않았다.

즉

실제로 실행된 흐름은

```
Input

↓

Encoder

↓

Bottleneck

↓

Decoder

↓

Output
```

이다.

---

# 3. train.py

이제 학습 부분이다.

---

## Dataset 불러오기

```python
train_dataset = LithoDataset(...)
```

여기서

dataset.py가 실행된다.

---

## DataLoader

```python
DataLoader(...)
```

설명

이미지를

```
batch
```

단위로 묶는다.

예를 들어

```
4장

8장

16장
```

씩 GPU에 넣는다.

---

## 모델 생성

```python
model = UNet(...)
```

여기서

model.py가 실행된다.

---

## Forward

```python
pred = model(target)
```

여기가 가장 중요하다.

흐름은

```
Target

↓

UNet

↓

Predicted Mask
```

이다.

---

## Loss

대부분

```python
criterion(...)
```

또는

```python
BCE
```

가 있다.

설명

예측한 Mask와

정답 Mask를 비교한다.

차이가 크면

Loss가 커진다.

---

## Optimizer

```python
optimizer.step()
```

설명

Loss가 줄어들도록

UNet 가중치를 업데이트한다.

---

이 과정을

30 Epoch

반복한 결과가

```
unet_base_c16_b4_e30.pth
```

이다.

---

# 4. real_litho_sim.py

여기서부터

학습이 아니라

평가 단계이다.

---

가장 중요한 부분

```python
pred_mask = torch.sigmoid(...)
```

설명

UNet 출력은

실수이다.

```
-5

1

3

0.8
```

등이 나온다.

Sigmoid를 이용해

```
0~1
```

범위로 만든다.

---

그 다음

```python
binary_mask=(mask>0.5).float()
```

설명

실제 Lithography는

Binary Mask를 사용하므로

```
0

1
```

로 변환한다.

---

그 다음

```python
printed_nom,
printed_max,
printed_min
```

설명

LithoBench의

공식 Simulator를 실행한다.

출력은

```
Nominal

Max Dose

Min Dose
```

패턴이다.

---

# 5. evaluate.py

여기서

최종 성능을 계산한다.

---

모델 불러오기

```python
load_state_dict(...)
```

↓

학습 완료된

```
unet_base_c16_b4_e30.pth
```

를 읽는다.

---

예측

```python
pred_mask=model(target)
```

↓

Mask 생성

---

Simulator

```python
printed=sim.simulate(...)
```

↓

Printed Pattern 생성

---

평가

```python
compute_l2(...)
```

↓

L2

---

```python
compute_pvb(...)
```

↓

PVB

---

```python
compute_epe_proxy(...)
```

↓

EPE

---

마지막으로

```python
json.dump(...)
```

↓

결과 저장

```
{
L2
PVB
EPE
...
}
```

---

# 발표에서는 이 흐름 하나만 이해하면 충분하다

```text
MetalSet
   │
   ▼
dataset.py
(데이터 읽기)
   │
   ▼
model.py
(UNet)
   │
   ▼
Predicted Mask
   │
   ▼
real_litho_sim.py
(리소그래피 시뮬레이션)
   │
   ▼
Printed Pattern
   │
   ▼
evaluate.py
(L2, PVB, EPE 등 계산)
```

이 흐름대로 설명하면 **"데이터 준비 → UNet 학습 및 예측 → 리소그래피 시뮬레이션 → 성능 평가"**가 자연스럽게 이어져 발표에서도 이해하기 쉽다.
