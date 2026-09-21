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

배포 `README.txt`가 요구하는 관련 논문 인용:

> Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014).
> Machine Learning Approaches for Improving Condition-Based Maintenance of Naval
> Propulsion Plants. Proceedings of the Institution of Mechanical Engineers, Part M:
> Journal of Engineering for the Maritime Environment.
> https://doi.org/10.1177/1475090214540874.

## 다운로드

1. [UCI 공식 페이지](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants)의
   `Download` 링크 또는 [공식 배포 ZIP](https://archive.ics.uci.edu/static/public/316/condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip)을
   통해 파일을 내려받는다.
2. ZIP 내부의 `UCI CBM Dataset/` 폴더를 찾는다. `__MACOSX/`와 `.DS_Store`는 사용하지 않는다.
3. `data.txt`, `README.txt`, `Features.txt`를 `data/raw/uci_cbm/` 아래에 복사한다.
4. 세 파일의 이름과 내용을 변경하지 않는다.
5. 최초 검증 시 사용한 원본 파일의 SHA-256 해시를 이 문서의 `실제 검증 결과`에 남긴다.

`data/raw/`와 `data/processed/`는 Git에서 제외된다. 이 저장소에는 UCI 원본 데이터를
커밋하지 않는다.

예상 로컬 구조는 다음과 같다.

```text
data/raw/uci_cbm/
├── data.txt
├── Features.txt
└── README.txt
```

파일을 배치한 뒤 다음 명령을 실행한다.

```bash
uv run python -m maritime_cbm.data.validation
```

검증 성공 시 파일 크기와 SHA-256, 행·열 수, 격자 고유값 수, 상수·중복 열과 `lp`–`v`
관계가 JSON으로 출력된다. 필수 파일, 행·열 수, 숫자·유한값, 공식 격자 또는 전체 조합이
맞지 않으면 실패한다. 상수 열, 동일 열과 `lp`–`v` 1:1 관계는 실패가 아니라 후속 모델링
결정을 위한 관찰 결과로 기록한다.

`data.txt`에는 헤더가 없으므로 검증 모듈은 공식 스키마 순서대로 열 이름을 부여한다.
실제 변수 순서가 맞는지는 함께 배포된 `Features.txt`와 `README.txt`를 이 문서의 변수 목록과
별도로 대조해 확인한다.

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

## 실제 검증 결과

검증일은 2026-09-21이며, UCI 공식 배포 ZIP에서 추출한 세 파일을 다음 명령으로
검증했다.

```bash
uv run python -m maritime_cbm.data.validation
```

파일 메타데이터:

| 상대경로 | 크기(byte) | SHA-256 |
|---|---:|---|
| `data.txt` | 3,448,926 | `de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6` |
| `Features.txt` | 758 | `3318e98f507c3bba7ba674d341bcda679c7634027e691c1fe3a43fa3e18b17ae` |
| `README.txt` | 5,366 | `1ea823d918fed1225329563244e8b0d804912dea4af7a3dad7b0c6caa7356b6b` |

구조 및 격자 검증:

- 로컬 구조: `data/raw/uci_cbm/` 바로 아래에 필수 파일 3개 존재
- 실제 크기: 11,934행, 18열
- 결측값 및 비유한값: 0개
- 운항 속도: 3~27 knots, 고유값 9개
- `kMc`: 0.950~1.000, 고유값 51개
- `kMt`: 0.975~1.000, 고유값 26개
- 반올림한 `(v, kMc, kMt)` 조합마다 정확히 1행 존재
- `Features.txt` 및 `README.txt`의 순서가 문서화한 16개 입력과 압축기·터빈 정답 순서와 일치

모델링 전 관찰 결과:

- `T1`은 모든 행에서 `288.0`인 상수 열이다.
- `P1`은 모든 행에서 `0.998`인 상수 열이다.
- `Ts`와 `Tp`는 모든 행에서 값이 완전히 동일하다.
- `lp`와 `v`는 각각 9개 고유값이 1:1로 대응한다.
- 이 관찰은 검증 실패가 아니며, 특성 제거 여부는 분할 전에 별도로 결정한다.

배포 문서의 upstream 이상:

- `README.txt` 첫 제목은 실제 데이터와 무관한 `Human Activity Recognition Using Smartphones Dataset`으로 잘못 적혀 있다.
- 현재 UCI 페이지는 CC BY 4.0으로 표시하지만, 배포 `README.txt`에는 상업적 이용을 금지하는 과거 문구가 남아 있어 서로 충돌한다.
- 이 프로젝트는 원본 데이터를 재배포하지 않고 출처와 두 라이선스 표기를 모두 기록한다. 상업적 이용이 필요하면 권리자 또는 UCI에 별도 확인해야 한다.

## 해석 제한

이 데이터는 타임스탬프가 없는 정상상태 시뮬레이션 데이터다. `kMc`, `kMt`는 실제 고장
발생을 관측한 라벨이 아니라 센서값 생성에 사용한 열화 상태 계수다. 따라서 이 프로젝트의
결과를 실제 고장 발생이나 시간에 따른 열화 진행 예측으로 해석하지 않는다.

데이터셋은 프로젝트 코드의 MIT License와 별도로 CC BY 4.0을 따른다. 데이터 사용·배포 시
원 저작자와 UCI 저장소를 표시해야 한다. 다만 배포 `README.txt`의 과거 상업적 이용 금지
문구와 현재 UCI 라이선스 메타데이터가 충돌하므로 상업적 이용 전에는 별도 확인이 필요하다.
