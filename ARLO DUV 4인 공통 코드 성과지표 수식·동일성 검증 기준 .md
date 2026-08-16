ARLO DUV 4인 공통 코드 성과지표 수식·동일성 검증 기준
버전: 2.0
적용일: 2026-08-12
적용 대상: 동일 과제를 수행한 4개의 코드 및 그 결과 파일

1. 이 문서의 목적
이 문서는 코드에 점수를 매기기 위한 평가표가 아니다. 4개 코드가 산출한 L2, PVB, EPE, Shots, Runtime 등의 지표가 실제로 서로 비교 가능한지를 확인하기 위한 기준서다.

지표 이름이 같다는 이유만으로 같은 성과지표라고 판단하지 않는다. 다음 항목이 모두 같아야 동일 지표로 인정한다.

계산 수식
입력 데이터와 정답의 의미
전처리, threshold, resize, interpolation
출력의 해상도, dtype, 값의 범위
단위
합계·평균·최댓값 등 reduction 방식
표본 범위와 누락값 처리
최종 집계 방식
계산에 사용한 evaluator, simulator, 라이브러리의 버전
따라서 이 문서의 최종 산출물은 순위나 점수가 아니라 지표 정의표, 수식, 코드 근거, 동일성 판정이다.

2. 지표 동일성의 판정식
다음 조건을 하나의 계약으로 고정한다.

Metric Identity
= Formula
+ Input/Target Definition
+ Preprocessing
+ Threshold and Resampling
+ Unit
+ Reduction
+ Sample Coverage
+ Aggregation
+ Implementation Version
이 중 하나라도 다르면 다음 다섯 가지 중 하나로 표시한다.

판정
의미
동일
수식·입력·조건·단위·집계·구현이 모두 같음
단위 변환 후 동일
수식은 같고 초·밀리초처럼 단위만 변환 가능함
조건부 동일
핵심 수식은 같지만 해상도·threshold·표본 범위가 달라 조건을 맞춰야 함
동일명·비동일
이름은 같지만 수식, reduction, 입력 또는 의미가 다름
비교 불가
수식이나 구현 근거를 확인할 수 없어 동일성 판단이 불가능함
3. 4개 코드에 공통으로 고정할 평가 계약
아래 값은 현재 ARLO DUV 평가 코드와 프로젝트 문서에서 확인되는 공통 조건이다. 팀이 다른 값을 채택할 경우에는 첫 결과를 확인하기 전에 4개 코드에 동시에 적용하고, 변경 이력을 남긴다.

항목
현재 공통 기준
데이터
동일한 LithoBench 및 MetalSet 입력
test split
동일한 split 파일과 동일한 sample ID
test 표본 수
1,648개
random seed
42
모델 입력 해상도
512 × 512
simulator·지표 해상도
2048 × 2048
모델 출력 처리
sigmoid 출력 후 동일 threshold로 이진화
simulator 설정
동일한 LithoSim 설정 파일과 optical condition
기본 이진화 기준
simulator 입력과 출력은 0.5 기준
custom EPE threshold
2.0 px
custom EPE 점 샘플 상한
target boundary와 printed boundary 각각 최대 2,048개
interpolation
해상도 변환 시 nearest
결과 단위
표본별 결과를 먼저 저장한 뒤 동일한 방식으로 집계
3.1 평가 fingerprint
각 코드의 결과 파일에는 다음 값으로 만든 evaluation_fingerprint를 기록한다.

split 파일 hash
입력 데이터 또는 sample 목록 hash
evaluator commit 또는 파일 hash
simulator 설정 파일 hash
모델 출력 threshold
simulator·지표 해상도
interpolation 방식
EPE threshold와 boundary sample 상한
reduction 방식
runtime 측정 규약
사용한 공식 EPE·shot 모듈의 버전 또는 commit
fingerprint가 다르면 파일의 지표 이름이 같아도 같은 평가 결과로 묶지 않는다.

4. 공통 전처리와 이진화 수식
4.1 해상도 변환
현재 코드에서는 모델 입력과 출력이 512 해상도이고, LithoSim과 주요 지표 계산은 2048 해상도에서 수행된다. 이때 해상도 변환은 nearest interpolation으로 고정한다.

$$ X_{2048} = \operatorname{NearestResize}(X_{512}, 2048 \times 2048) $$

bilinear, bicubic, antialiasing resize를 사용하면 같은 모델 출력이라도 지표 입력이 달라지므로 동일 지표가 아니다.

4.2 이진화
모델의 연속 출력 M과 target 또는 simulator 출력 X를 이진 마스크로 바꾸는 규칙을 고정한다.

$$ B(X){ij} = \begin{cases} 1 & X{ij} \ge 0.5 \ 0 & X_{ij} < 0.5 \end{cases} $$

현재 공통 evaluator는 simulator에 넣을 mask, target, nominal, maximum, minimum을 0.5 기준으로 이진화한다. custom EPE의 경계 추출 코드도 이진 mask를 대상으로 하며, boundary 함수 내부에서는 > 0.5를 사용한다. 0.5 값이 실제 입력에 남을 수 있는 구현이라면 >와 >=의 차이까지 별도로 확인한다.

4.3 입력 의미
다음 입력을 서로 혼동하지 않는다.

target: 목표 pattern 또는 기준 pattern
mask: 모델이 예측한 mask 또는 pixelILT 기준 mask
nominal: 예측 mask를 simulator에 넣은 nominal print
maximum: process corner 중 maximum print
minimum: process corner 중 minimum print
예측 mask와 printed image를 바꿔 넣으면 수식이 같아도 다른 지표다.

5. 지표별 정식 정의
아래 정의를 이 프로젝트의 공통 evaluator 기준으로 사용한다. 각 코드가 결과를 제출할 때에는 표에 적힌 canonical name과 원래 코드의 변수명을 함께 기록한다.

5.1 L2 MSE
대상은 이진화된 nominal print P_nominal과 이진화된 target T다. 픽셀 수를 N이라고 할 때 현재 공통 evaluator의 l2_mse는 다음과 같다.

$$ \operatorname{L2_MSE}(P_{nominal}, T)
\frac{1}{N} \sum_{i=1}^{N} \left(P_{nominal,i} - T_i\right)^2 $$

현재 구현에서는 PyTorch F.mse_loss(nominal, target)의 기본 reduction="mean"에 해당한다.

별도로 저장하는 l2_official_sum은 평균이 아니라 합계다.

$$ \operatorname{L2_SUM}(P_{nominal}, T)
\sum_{i=1}^{N} \left(P_{nominal,i} - T_i\right)^2 $$

따라서 다음은 같은 이름의 변형이 아니라 서로 다른 집계 지표다.

l2_mse: 픽셀 평균 제곱 오차
l2_official_sum: 픽셀 제곱 오차의 합
l2_raw: basic.run의 내부 수식과 reduction을 확인하기 전에는 l2_mse와 동일하다고 볼 수 없음
두 값은 같은 입력과 같은 binary mask를 사용하고 N이 같을 때에만 l2_official_sum = l2_mse × N 관계를 확인할 수 있다. 관계가 성립하더라도 보고서에는 평균과 합계를 서로 다른 열로 표시한다.

