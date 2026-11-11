# UTIAS 솔버 미분·선형화 검증

검증일: 2026-11-11. 현재 `UTIAS dataset.py`의 2 Hz, 5초 설정으로 실행했다.
변수 55개, factor 207개이며 prior/Q/R, 데이터 연결, 모델은 원본과 동일하다.
솔버 본체와 기존 `history.csv`는 이 검증에서 수정하지 않았다.

## 판단

이번 재현에서는 **해석적 Jacobian의 오류보다, 매 반복 재선형화와 감쇠 없는 전체 상태 갱신을 결합한 비선형 GBP 경로가 불안정성의 핵심**이라는 증거가 강하다.
재선형화를 끄면 기존 SE(2) 좌표계 이동·메시지 갱신 코드를 그대로 사용해도 안정화된다.
수정 전후 비교를 수행한 결과는 아니며, 특정 안정화 방식의 효과까지 검증한 것은 아니다.

## 실행 결과

| 검사 | 결과 |
|---|---|
| 기존 단위 테스트 | 36개 통과 |
| 모든 factor의 초기 해석적 Jacobian 대 중앙차분, 간격 1e-5 | odometry 최대 절대오차 5.86e-11, range-bearing 1.15e-10 |
| 비선형 GBP의 최저 Energy 상태에서 같은 검사 | odometry 5.84e-11, range-bearing 1.41e-10 |
| 중앙집중 GN의 최종 상태에서 같은 검사 | odometry 8.26e-11, range-bearing 1.75e-10 |
| 실제 Energy의 방향미분 대 조립한 gradient, 초기/중앙집중 최종 상태 각 5방향 | 최대 절대오차 각각 2.49e-6 / 2.18e-6; 차분 간격 1e-6 |
| 초기 선형계 | 최소 고유값 19.0, condition number 5.85e4 |
| 초기 선형화를 공통 좌표계에 고정한 Euclidean GBP | 직접 선형해와 상태 최대 절대오차 2.99e-14 |
| 고정 선형계의 precision이 안정화된 뒤 eta 반복행렬 | spectral radius 0.83345 < 1 |
| SE(2) 구현을 유지하고 `local_relin=False` | 100회 Energy 4392.638607, step norm 2.85e-9 |
| 동일 비선형 목적함수의 중앙집중 GN + backtracking | Energy 2231.375313, 최종 gradient 무한대 norm 1.75e-5 |

중앙차분 간격 1e-4, 1e-5, 1e-6을 모두 비교했다. 선형화 정보형의 부호도
`Lambda = J.T @ R^-1 @ J`, `eta = -J.T @ R^-1 @ residual`과 실제 Energy gradient를 대조했다.
각도 branch cut에서의 미분 가능성을 보장하는 검사는 아니다.

원본 비선형 GBP에서는 다음과 같이 감소 후 반복적인 Energy 폭증을 재현했다.

| 반복 | 실제 비선형 Energy | step norm |
|---:|---:|---:|
| 0 | 4695.224591 | — |
| 20 | 3366.481119 | 0.095934 |
| 40 | 3476.145611 | 0.270362 |
| 60 | 6561.545194 | 1.083102 |
| 91 | 167870.638444 | 3.808145 |
| 100 | 104094.933374 | 3.780956 |
| 131 | 2764.822326 | 0.656137 |
| 200 | 3458.256695 | 0.845227 |

200회 내내 무한히 증가한 것은 아니다. 크게 증가한 뒤 다시 감소하는 진동과 미수렴이 관찰됐다.
위 Energy 비교는 같은 비선형 목적함수를 사용한다. 별도 고정 Euclidean 선형계의
최종 Energy 3250.547315는 다른 근사 목적함수 값이므로 이 표와 직접 비교하지 않는다.
중앙집중 GN 결과도 전역 최적성의 보증은 아니다.

## 코드에서 확인한 원인 경로

1. `build_graph.py:44`의 `FactorGraph()`는 `beta=0`, `min_linear_iters=0`, `eta_damping=0`을 사용한다.
2. `my_gbp/gbp.py:230`은 인접 상태가 변한 factor를 바로 재선형화한다.
3. `my_gbp/gbp.py:241`은 매번 재선형화 → 메시지 한 번 갱신 → belief 갱신 순서로 진행한다.
   하나의 선형계에 대한 메시지가 충분히 안정화되기 전에 다음 선형계로 바뀔 수 있다.
4. `my_gbp/variable_node.py:57`은 계산한 증분 전체를 바로 retract한다.
   Energy 증가를 검사해 step을 줄이거나 거부하는 절차는 없다. `solve()`의 Energy는 모니터링 출력에만 쓰인다.

고정 선형계의 Schur 메시지 계산은 직접해와 일치하고, 재선형화를 끈 SE(2) 실행도 안정화됐다.
따라서 이번 증상에 대해 Jacobian 부호나 Schur 공식을 먼저 바꿀 근거는 발견하지 못했다.
다음 수정 후보는 **선형화를 유지하는 내부 메시지 반복과 외부 재선형화를 구분하고,
외부 상태 갱신에 실제 Energy를 이용한 step 수락·축소를 적용하는 것**이다.
반복 수나 damping 값은 이 검증에서 새 기본값으로 정하지 않았다.

## 별도로 재현한 두 문제

- `my_gbp/differentiation.py:34`: `check_jac()`가 `max(J - J_fd)`를 사용해 절대오차를 검사하지 않는다.
  실제 미분이 +1인 함수에 -1을 반환하는 Jacobian을 넣어도 `Passed!`가 출력됐다.
  이번 검증은 이 함수를 신뢰하지 않고 독립적인 중앙차분과 절대오차를 사용했다.
  현재 UTIAS의 solve 경로에서는 이 검증 함수를 호출하지 않으므로 직접적인 발산 원인은 아니다.
- `my_gbp/gbp.py:201,230`: 매번 재선형화하면 `iters_since_relin=0`, `factor.eta_damping=0`으로 초기화된다.
  `num_undamped_iters>0`이면 damping 활성화 조건에 도달하지 못할 수 있다.
  작은 비선형 예제에서 `eta_damping=0.8`, `num_undamped_iters=1`을 지정했지만 4회 모두 실제 damping은 0이었다.
  현재 UTIAS는 처음부터 damping=0이므로 이 설정 결함이 현재 기본 실행의 직접 원인은 아니다.

## 재현

저장 결과: `utias_solver_audit.json`.
실행 스크립트는 설정 파일을 읽고 메모리 안에서 대조 실험을 하며, `main()`이나 history 저장을 호출하지 않는다.
JSON에는 전체 Energy trace와 공분산, 검사 결과를 저장한다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools/validate_utias_solver.py --iters 200
```

검증용 스크립트와 이 보고서, 결과 JSON만 추가했다. 솔버 수정은 적용하지 않았다.
