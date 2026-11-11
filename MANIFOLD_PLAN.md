# my_gbp manifold 확장 계획

작성일: 2026-11-10

상태: Euclidean·SE2 기하 연산, NdimGaussian 기준점 변환, variable·factor 연결 및 3개 포즈의 range/bearing 예제 검증 완료. 수렴 판정 보완과 추가 실험은 이후 진행.

## 1. 합의한 설계

- 기하 연산, Gaussian 표현, GBP 알고리즘을 분리한다.
- `SE2`는 Gaussian이나 변수 상태를 소유하지 않고 기하 연산을 제공한다.
- Gaussian이 manifold 객체와 기준 상태를 이용한다. `SE2`를 Gaussian에 의존시키지 않는다.
- 별도 `ManifoldGaussian` 클래스는 만들지 않고, 기존 `NdimGaussian` 하나에 `manifold`, `reference`, 기준점 변환을 추가한다.
- 진행 순서: Euclidean을 이용한 variable·factor 연결을 먼저 검증했고, 이후 사용자 요청에 따라 SE2 기하 연산을 추가했다.
- 기존 `FactorGraph`–`Factor`–`VariableNode` 구조를 유지하고 내부를 점진적으로 확장한다.
- 첫 지원 범위는 기존 Euclidean 공간과 SE(2)이다. 다른 manifold를 추가할 수 있는 인터페이스를 설계하되 SE(3) 등은 이번 구현에 포함하지 않는다.
- 기존 Euclidean 사용법과 Numerical·Analytic·AD 선택을 유지한다.
- `solve()`의 `(converged, n_iters, history)` 반환 형식을 유지한다.
- 모델, 측정 residual의 의미, Q/R, 실험 설정은 구현 과정에서 임의로 변경하지 않는다.

## 2. 기하 및 좌표 계약

- SE(2) 상태: `3×3` homogeneous transform.
- 접공간 벡터: `[rho_x, rho_y, theta]`, 회전 단위는 radian.
- 오른쪽 증분: `retract(X, delta) = X @ Exp(delta)`.
- 국소 좌표: `local(X, Y) = Log(inverse(X) @ Y)`.
- `Exp(delta)`의 이동 성분은 SE(2) exponential의 결합을 반영한다. 단순히 `delta[:2]`를 이동 성분으로 복사하지 않는다.
- `local`/`Log`는 국소 좌표이며, principal rotation branch와 ±pi 경계의 특성을 문서화한다.
- 상태 저장 형태와 접공간 차원을 구분한다. Gaussian의 차원은 행렬 원소 수가 아니라 접공간 차원이다.

## 3. 파일별 역할

| 파일 | 역할 |
|---|---|
| `my_gbp/manifold/` | 공통 manifold 인터페이스, `Euclidean`, `SE2`, 좌표 변환에 필요한 기하 Jacobian |
| `my_gbp/gaussian.py` | `NdimGaussian` 하나에 manifold, 기준 상태, canonical parameters 및 기준점 변환을 통합 |
| `my_gbp/variable_node.py` | 고정 prior와 수신 메시지를 공통 접공간에서 결합하고 belief 갱신 |
| `my_gbp/factor.py` | 접공간 residual 선형화, cavity 구성, Schur complement 주변화 |
| `my_gbp/differentiation.py` | `retract`를 통한 접공간 유한차분과 Jacobian 검증 |
| `my_gbp/autodiff.py` | Numerical·Analytic·JAX AD 선택과 residual Jacobian 연결 |
| `my_gbp/gbp.py` | 동기식 반복, 재선형화, 수렴 판정, history 관리 |
| `my_gbp/__init__.py` | 새 공개 클래스의 패키지 export |

## 4. 단계별 구현 및 검증

### 단계 0: 기존 동작을 회귀 검증 기준으로 정리

- [x] 현재 Euclidean toy example의 세 Jacobian 방식으로 mean, cov, 수렴 상태, 반복 횟수를 기록한다.
- [x] 선형 트리 문제의 직접 Gaussian 해와 GBP 결과를 비교한다.
- [x] history 길이·배열 shape·복사 독립성과 재호출 초기화를 확인한다.

### 단계 1: manifold 기하 연산

- [x] 공통 계약: `tangent_dim`, `retract(state, delta)`, `local(reference, state)`.
- [x] `Euclidean`: 덧셈·뺄셈으로 같은 계약을 구현한다. NumPy 및 JAX AD/JIT에서 검증했다.
- [x] `my_gbp/manifold/SE(2).py`에 `SE2`의 `identity`, `compose`, `inverse`, `hat`, `vee`, `exp`, `log`, `retract`, `local`을 구현했다. 패키지에서 `from my_gbp.manifold import SE2`로 불러온다.
- [x] `adjoint`, `right_jacobian`, `right_jacobian_inverse`, `retract_jacobians`, `local_jacobians`를 구현했다. 두 인자에 대한 Jacobian은 오른쪽 접공간 좌표를 사용한다.
- [x] SciPy `expm`, group identity, inverse, retract/local 역관계, 작은 각도, ±pi 및 principal branch 바깥의 Exp/Log 재구성을 검증했다.
- [x] 기본 backend는 NumPy이며, AD 경로에서는 `SE2(xp=jax.numpy)`를 사용한다. JAX forward/reverse AD, 영점의 Hessian, JIT, vmap을 검증했다.

완료 기준: solver를 변경하지 않은 상태에서 기하 연산과 해당 Jacobian을 독립적으로 검증한다.

### 단계 2: 기준 상태를 가진 Gaussian

- [x] 기존 `NdimGaussian` 하나에 `dim`, `eta`, `lam`, `manifold`, `reference`를 보관한다. 별도 Gaussian 클래스나 내부 canonical wrapper를 추가하지 않는다.
- [x] 기존 생성자 인자는 유지하고, `manifold`와 `reference`는 keyword-only로 추가한다. 기본 Euclidean 원점 기준에서 기존 `eta`, `lam`의 의미를 유지한다.
- [x] `rebase(reference)`는 정보벡터의 이동과 정밀도의 Jacobian 변환을 수행하고 새 `NdimGaussian`을 반환한다. 원본과 prior는 변경하지 않는다.
- [x] 이 변환은 국소 선형 근사임을 명시한다. Euclidean에서는 정확한 기준점 이동이다.
- [x] zero/rank-deficient 메시지를 표현한다. 생성과 `rebase`는 정밀도 역행렬을 요구하지 않는다.
- [x] Euclidean 기준점 이동, 왕복·연속 기준점 변경, 실제 mean·cov 및 상대 potential 보존을 검증했다. 비선형 테스트 chart에서도 Jacobian 변환 경로를 확인했다.
- [x] 실제 SE2 기하 구현을 연결해 작은 변위의 기준점 변환과 zero/rank-deficient 메시지를 검증했다.

완료 기준: 서로 다른 기준점에서 정의된 정보를 공통 접공간으로 옮기는 의미와 수치 결과가 일치한다.

### 단계 3: 팩터와 세 가지 미분 방식

- [x] Manifold 팩터는 변수별 상태 tuple을 받는다. 접공간 증분만 하나의 벡터로 쌓는다. 행렬 상태와 접공간 차원이 다른 테스트 chart로 확인했다.
- [x] 기존 Euclidean 측정 함수의 flat-vector 입력은 adapter로 유지한다.
- [x] `residual(retract(states, delta))`를 `delta=0`에서 선형화한다.
- [x] residual을 `h - z`로 정했다. 기존 Euclidean analytic callback의 `dh/dx`에는 retraction Jacobian을 곱하고, tuple callback은 `de/d(delta)`를 받는다.
- [x] Numerical은 상태 행렬 원소를 직접 perturb하지 않고 `retract`를 이용한다.
- [x] Analytic과 AD도 같은 접공간 Jacobian을 계산한다. 현재 Euclidean 경로를 검증했다.
- [x] Euclidean AD 경로의 기하 연산을 JAX backend로 연결했다. 향후 다른 manifold도 AD 사용 시 JAX 호환 구현이 필요하며, SE2 zero increment 검증은 아직 남아 있다.
- [x] `Lambda = J.T @ W @ J`, `eta = -J.T @ W @ e0`를 증분 좌표에서 구성한다.
- [ ] 측정도 manifold 위에 있다면 `local`로 벡터 residual을 정의한다. 측정 잡음 공분산의 기준 좌표계도 함께 명시한다.

완료 기준: 같은 residual에서 Numerical·Analytic·AD Jacobian이 허용 오차 내에서 일치한다.

### 단계 4: 메시지 및 belief 갱신 연결

- [x] 모든 variable-to-factor cavity를 먼저 준비한 뒤 factor-to-variable 메시지를 계산해 동기식 계산을 유지한다.
- [x] prior와 수신 메시지를 공통 접공간으로 옮긴 뒤 정보를 결합한다.
- [x] 수신 대상 팩터를 제외하고 나머지 메시지를 직접 합산해 cavity를 구성한다. Full belief에서 다른 기준의 메시지를 바로 빼지 않는다.
- [x] cavity와 팩터를 같은 좌표계에서 결합한 뒤 기존 Schur complement 구조를 사용한다.
- [x] factor-to-variable 메시지에 해당 접공간의 기준 상태를 함께 보관한다.
- [x] belief의 증분을 구해 `retract`로 mean을 갱신하고, cov를 새 mean의 접공간으로 변환한다.
- [x] prior mean은 고정된 확률 모델의 기준이다. 현재 추정값으로 prior를 덮어쓰지 않는다.
- [x] damping은 이전 메시지를 새 메시지의 기준점으로 옮긴 뒤 수행한다.

완료 기준: Euclidean 회귀 결과와 계산 순서·기준점 불변성을 검증했다. SE2의 기본 단일 팩터 문제에서도 상태·메시지·belief·history를 확인했다. 복합 포즈 그래프 검증은 별도로 진행한다.

### 단계 5: graph API, 수렴 판정, history

- [x] `add_variable(..., manifold=...)`를 추가했다. manifold 미지정 시 기존 Euclidean 동작을 유지한다.
- [x] 재선형화 및 상태 변화 측정에 `local(old, new)`를 사용한다. 기존 수렴 기준의 평균 변화 조건 자체는 유지했다.
- [ ] 기존의 평균 변화만 보는 조기 종료 문제를 반영해, 메시지·공분산 변화와 재선형화 대기 상태를 고려한다.
- [ ] 이동과 회전의 단위가 다르므로 수렴 metric/threshold는 구현 전에 사용자와 확정한다.
- [x] `history`의 `iter`, `means`, `covs`와 solve의 3개 반환값을 유지한다.
- [x] history의 mean은 실제 상태이며, cov는 해당 mean의 접공간에서 정의한다. 매 iteration 종료 후 복사하고, 초기 상태는 제외하며, solve 재호출 시 초기화한다.
- [x] SE2 시각화는 pose에서 위치·방향을 추출한다. `examples/utils/plotter.py`는 위치 공분산을 `R @ cov[:2, :2] @ R.T`로 월드 좌표계에 옮겨 타원을 그리고, 자세 방향을 화살표로 표시한다. 기존 Euclidean 2D 이력도 유지한다.

### 단계 6: 통합 검증

- [x] 기존 Euclidean toy example의 세 미분 방식과 직접 선형 해 비교를 다시 수행한다.
- [x] SE2 단일 변수의 위치 측정 테스트에서 Numerical·Analytic·AD 결과, prior 고정, rank-deficient 메시지 및 행렬 상태 history를 검증했다. 실제 포즈 그래프 실험을 대신하는 검증은 아니다.
- [x] 현재 range/bearing 예제에서 같은 prior, wrapped residual, covariance를 사용하는 SciPy 중앙집중 국소 최적화와 비교했다. GBP 결과에서 추가 최적화 증분 norm은 약 4.9e-7, 목적함수 차이는 약 3.5e-13, gradient norm은 약 3.1e-6이었다. 전역 최적성을 의미하지 않는다.
- [ ] tree의 선형 Gaussian 공분산과 loopy/nonlinear 근사 공분산을 구분한다.
- [x] `examples/SE(2)_example.py`에 사용자가 작성한 3개 prior 포즈·공분산, 두 range/bearing 관측 및 `0.1 * I2`를 유지했다. Bearing은 사용자가 확인한 observer body frame의 radian 각도이다.

## 5. 현재 상태와 다음 작업

- 현재 기하 구현은 `my_gbp/manifold/Euclidean.py`에 있다. 과거 다른 경로의 manifold 구현이나 검증 결과를 현재 구현의 완료 증거로 사용하지 않는다.
- `my_gbp/manifold/SE(2).py`도 구현했다. Exp의 이동은 V(theta) @ rho이며, sinc와 Taylor 전개로 영점 근처를 처리한다. Log는 principal rotation branch를 사용한다.
- `NdimGaussian`의 `rebase`를 variable·factor에 연결했다. Prior는 prior mean 기준에 고정하며, 팩터의 canonical parameters와 메시지는 해당 선형화 상태 주변의 증분을 나타낸다.
- `FactorGraph`는 모든 cavity를 먼저 준비한다. `VariableNode`는 공통 기준에서 정보를 합산하고 belief를 새 mean으로 옮긴다. SE2 NumPy/JAX 구현을 기본 연결 테스트에 사용했다.
- 연결 전후 toy example의 AD·Numerical·Analytic 결과를 비교했다. 세 방식 모두 14회 반복했다. 전체 mean·cov 이력의 최대 차이는 각각 AD 약 2.4e-7, Numerical 약 5.2e-10, Analytic 약 1.1e-14였다.
- `tests/test_euclidean_message_passing.py`의 5개 테스트가 통과했다. 선형 tree의 직접 해, rank-deficient 메시지, 팩터 순서 및 메시지 기준점 불변성, 세 Jacobian 방식과 history, tuple callback의 행렬 상태 형태를 검증한다.
- `tests/test_se2.py`의 7개 테스트를 추가했고, 두 파일의 전체 12개 테스트가 통과했다. 과거 구현의 109개 사례 결과를 재사용하지 않고 현재 소스로 실행했다.
- `examples/SE(2)_example.py`는 `x1 -> x2`, `x2 -> x3`의 range/bearing 측정을 가진다. SE2 상태는 3x3 pose 행렬이며 Numerical·Analytic·AD가 모두 36회에 수렴했다. AD는 float64 context에서 검증했다.
- 기존 첫 관측의 초기 bearing 오차가 정확히 180도이므로, 예제의 Numerical callback은 각도 차이를 wrap하는 중앙차분을 사용한다. 일반적인 실수 벡터 차분의 2pi 점프로 공분산이 잘못되는 문제를 피했다. Solver 전체에 임의의 angular measurement 처리를 추가한 것은 아니다.
- `tests/test_range_bearing_example.py`에서 몸체 좌표계·각도 wrap·원주 차분·analytic/AD Jacobian·월드 위치 공분산 변환을 확인했다. 현재 전체 16개 테스트가 통과했다.
- 기존 수렴 판정의 조기 종료 가능성과 미구현 `robustify_loss`는 알려진 별도 항목이다. 단순 기하 모듈 작성 과정에서 함께 수정하지 않는다.
- 이후 추가 SE2 실험의 모델·prior·측정·noise는 사용자와 합의한다. 기존 수렴 판정과 범용 manifold-valued measurement 인터페이스 보완은 별도 작업으로 남아 있다.

## 참고

- [GTSAM manifold 인터페이스](https://borglab.github.io/gtsam/manifold/)
- [A micro Lie theory for state estimation in robotics](https://arxiv.org/html/1812.01537v9)
- [A Robot Web for Distributed Many-Device Localisation, Section IV](https://arxiv.org/html/2202.03314v2#S4)
