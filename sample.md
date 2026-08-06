좋은 질문이야. 두 번째 질문이 특히 중요한데, **UNet와 Attention(Bottleneck)**을 조금 구분해서 생각해야 해.

---

# 1. 각 파일별로 코드를 보여주면서 설명 가능?

가능해. 오히려 발표라면 그게 가장 이해하기 쉽다.

내가 추천하는 순서는

```
dataset.py
      ↓
model.py (UNet)
      ↓
train.py
      ↓
evaluate.py
      ↓
real_litho_sim.py
      ↓
metrics.py
```

이 순서가 실제 데이터가 흘러가는 순서와 거의 같다.

---

예를 들면 dataset.py에서는

```python
class LithoDataset(Dataset):
```

이 부분

↓

설명

> LithoDataset 클래스를 정의한다.
>
> PyTorch에서 데이터를 불러오기 위한 Dataset 클래스이다.

---

그리고

```python
def __getitem__(self, idx):
```

설명

> 하나의 데이터를 읽어오는 함수이다.

여기서

```
Target
Mask
```

를 읽어온다.

---

다음

model.py에서는

예를 들면

```python
self.pool = nn.MaxPool2d(2)
```

설명

> Max Pooling이다.

역할

```
256×256

↓

128×128

↓

64×64

↓

32×32
```

이렇게 Feature Map 크기를 줄여준다.

---

다음

```python
nn.Conv2d(...)
```

설명

> Convolution Layer이다.

역할

```
선
모서리
패턴
형태
```

같은 특징(feature)을 추출한다.

---

다음

```python
nn.ConvTranspose2d(...)
```

설명

> Up Sampling이다.

```
32

↓

64

↓

128

↓

256
```

다시 원래 크기로 복원한다.

---

이런 식으로 실제 코드와 같이 설명하는 게 PPT에서 가장 이해하기 쉽다.

---

# 2. Bottleneck이 Attention 아닌가?

여기서 많이 헷갈리는 부분이다.

결론부터 말하면

**아니다.**

---

UNet 구조는

```
Encoder

↓

Bottleneck

↓

Decoder
```

이다.

즉

```
Encoder

↓

Bottleneck

↓

Decoder
```

는 원래부터 존재하는 구조이다.

---

Attention UNet은

```
Encoder

↓

Attention

↓

Bottleneck

↓

Attention

↓

Decoder
```

처럼

Attention Module이 추가된 것이다.

즉

```
UNet

+

Attention
```

이다.

---

그래서

```
Bottleneck
```

은

UNet에도 있다.

Attention에도 있다.

둘 다 있다.

---

즉

```
Bottleneck = Attention
```

이 아니다.

---

# 그림으로 보면

기본 UNet

```
Target

↓

Encoder

↓

Encoder

↓

Encoder

↓

Bottleneck

↓

Decoder

↓

Mask
```

---

Attention UNet

```
Target

↓

Encoder

↓

Attention Gate

↓

Encoder

↓

Attention Gate

↓

Bottleneck

↓

Decoder

↓

Mask
```

즉

Attention은

**Skip Connection을 지나가는 Feature를 선택적으로 강조하는 모듈**이다.

---

# 3. 그럼 이번 실험은 Bottleneck이 빠진 건가?

아니다.

**Bottleneck은 반드시 있다.**

빠지면 UNet이 아니다.

---

이번 실험은

```
Target

↓

Encoder

↓

Bottleneck

↓

Decoder

↓

Mask
```

이다.

---

Attention은

없다.

---

# 왜 없다고 확신할 수 있냐?

evaluate.py를 보면

```python
model = UNet(
    base_channels=args.base_channels,
    use_attention=args.attention
)
```

여기서

```
use_attention
```

이라는 옵션이 있다.

---

그리고 실행할 때

너는

```
python evaluate.py --checkpoint unet_base_c16_b4_e30.pth
```

만 실행했다.

여기에는

```
--attention
```

옵션이 없다.

즉

```
args.attention=False
```

이다.

그래서

```
UNet(use_attention=False)
```

가 실행된다.

즉

**기본 UNet만 실행된 것이다.**

---

# 만약 Attention을 사용하려면

실행 명령이

```
python evaluate.py ^
--checkpoint unet_attention_c16_b4_e30.pth ^
--attention
```

처럼 되어야 한다.

여기서

```
--attention
```

이 들어가야

Attention Layer가 생성된다.

---

# 정리하면

이번에 실행된 모델은

```
Target

↓

Encoder
(Conv + ReLU + MaxPool)

↓

Bottleneck
(Conv)

↓

Decoder
(Transpose Conv + Skip Connection)

↓

Predicted Mask

↓

Lithography Simulation

↓

Printed Pattern

↓

L2
PVB
EPE
CTHR
Shot
```

이다.

즉,

* ✅ Encoder 사용
* ✅ Bottleneck 사용
* ✅ Decoder 사용
* ✅ Skip Connection 사용
* ❌ Attention 사용 안 함

---

내가 보기에는 이제 발표에서 가장 중요한 부분인 **`model.py`(UNet 구조)를 코드 한 줄씩 따라가며 "이 줄이 어떤 역할을 하는지" 설명**하는 자료를 만드는 단계로 넘어가면 된다. 이 부분이 발표의 핵심이 될 가능성이 가장 크다.
