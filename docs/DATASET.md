# 데이터셋 안내

## 출처

- 이름: `Condition Based Maintenance of Naval Propulsion Plants`
- 제공처: UCI Machine Learning Repository
- 공식 페이지: <https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants>
- DOI: <https://doi.org/10.24432/C5K31K>
- 라이선스: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

권장 인용:

> Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014).
> Condition Based Maintenance of Naval Propulsion Plants [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5K31K.

## 다운로드

1. UCI 공식 페이지의 `Download` 링크에서 배포 파일을 내려받는다.
2. 압축을 해제하고 원본 파일을 `data/raw/uci_cbm/` 아래에 둔다.
3. `data.txt`, `README.txt`, `Features.txt`의 이름과 내용을 변경하지 않는다.
4. 최초 검증 시 사용한 원본 파일의 SHA-256 해시를 실험 기록에 남긴다.

`data/raw/`와 `data/processed/`는 Git에서 제외된다. 이 저장소에는 UCI 원본 데이터를
커밋하지 않는다.

예상 로컬 구조는 다음과 같다.

```text
data/raw/uci_cbm/
├── data.txt
├── Features.txt
└── README.txt
```

## 공식 설명 기준 검증값

- 행 수: 11,934
- 열 수: 입력 16개와 정답 2개를 합한 18개
- 운항 속도: 3~27 knots, 3 knots 간격, 고유값 9개
- `kMc`: 0.950~1.000, 0.001 간격, 고유값 51개
- `kMt`: 0.975~1.000, 0.001 간격, 고유값 26개
- 전체 격자: 9 × 51 × 26 = 11,934행

부동소수점 격자는 직접 equality 비교하지 않고 허용오차 또는 반올림을 적용해 검증한다.
실제 파일의 행·열 수, 변수 순서와 격자 구조가 공식 `README.txt`와 다르면 모델링을
중단하고 원인을 먼저 기록한다.

## 해석 제한

이 데이터는 타임스탬프가 없는 정상상태 시뮬레이션 데이터다. `kMc`, `kMt`는 실제 고장
발생을 관측한 라벨이 아니라 센서값 생성에 사용한 열화 상태 계수다. 따라서 이 프로젝트의
결과를 실제 고장 발생이나 시간에 따른 열화 진행 예측으로 해석하지 않는다.

데이터셋은 프로젝트 코드의 MIT License와 별도로 CC BY 4.0을 따른다. 데이터 사용·배포 시
원 저작자와 UCI 저장소를 표시해야 한다.
