# M1 EDA 산출물

이 디렉터리에는 UCI `Condition Based Maintenance of Naval Propulsion Plants` 릴리스의
집계 통계와 핵심 그림만 저장한다. 행 단위 분할 목록, 중간 파일과 임시 그림은 저장소에
포함하지 않는다.

## 재현 방법

```bash
uv sync --group eda
uv run --group eda python -m maritime_cbm.data.eda
```

CSV 표만 생성할 때는 `matplotlib` 없이 다음 명령을 사용할 수 있다.

```bash
uv run python -m maritime_cbm.data.eda --skip-figures
```

산출물은 다음과 같다.

- `summary_statistics.csv`: 전체 원본 입력·정답의 집계 통계
- `train_correlation_matrix.csv`: 기본 그룹 분할 학습 행의 pooled 상관행렬
- `train_speed_conditioned_correlations.csv`: 학습 행을 운항 속도별로 나눈 센서·정답 상관
- `train_between_speed_variance_ratio.csv`: 센서 분산 중 속도별 평균 차이가 차지하는 비율
- `target_grid_coverage.png`: 전체 열화 상태 격자의 운항 속도 관측 수
- `state_group_split.png`: 기본 그룹 분할의 상태 조합 배치
- `train_feature_target_correlations.png`: pooled 상관과 속도별 상관 절댓값 중앙값 비교

## 주요 관찰과 해석

센서·정답 관계 분석에는 기본 상태 그룹 분할의 학습 행만 사용했다. pooled Pearson 상관은
모든 센서와 `kMc`, `kMt` 조합에서 절댓값 0.048 이하이지만, 속도별로 계산하면 상관 절댓값의
최댓값은 0.992다. 대표적인 속도별 절댓값 중앙값은 `T2`–`kMc` 0.934,
`P2`–`kMt` 0.948, `GTT`–`kMc` 0.837이다.

센서 분산 중 속도별 평균 차이가 차지하는 비율은 `GTT` 0.999, `P2` 0.999, `mf` 0.996,
`T48` 0.977이다. 따라서 거의 회색인 pooled 상관 그림은 센서가 열화 상태와 무관하다는
뜻이 아니다. 운항 속도가 센서 변동을 지배해 속도별 열화 관계를 가리는 현상으로 해석한다.
M2에서는 원시 센서의 단순 선형 관계만 가정하지 않고 운항 조건을 반영하는 방법을 학습
데이터 안에서 검토한다. 이 값들은 모델 성능이 아니라 EDA 관찰 결과다.

## 출처와 라이선스

데이터 출처는 [UCI Machine Learning Repository](https://doi.org/10.24432/C5K31K)이며,
공식 페이지는 CC BY 4.0으로 표시한다. 원본 파일은 이 저장소에 포함하지 않는다.
현재 UCI 라이선스 표기와 배포 `README.txt`의 과거 상업적 이용 금지 문구가 충돌하므로,
상업적 이용 전에는 [데이터 카드](../../docs/DATASET.md)의 안내를 확인해야 한다.
