# Maritime Anomaly Signal Diagnosis

스마트해운물류 × AI 미션 챌린지의 **이상신호 감지 기반 비정상 작동 진단** 문제를 기반으로 구성한 멀티클래스 센서 이상진단 프로젝트입니다.

## Competition Result

| 항목 | 내용 |
|---|---|
| 대회명 | 스마트해운물류 × AI 미션 챌린지 : 이상신호 감지 기반 비정상 작동 진단 |
| 주최 | 해양수산부 |
| 문제 유형 | 정형 데이터 기반 이상신호 감지, 20-class anomaly classification |
| 성과 | 예선 리더보드 2등 |
| 최종 점수 | 0.88727 |
| 대회 링크 | https://dacon.io/competitions/official/236590/leaderboard |

## 1. 프로젝트 개요

입력 데이터는 `X_01`부터 `X_52`까지의 비식별 센서 변수로 구성됩니다. 목표는 각 샘플을 20개의 anomaly class 중 하나로 분류하는 것입니다.

본 문제의 주요 난점은 다음과 같습니다.

- 센서 간 상관관계가 높고 일부 변수에 잡음과 이상치가 포함됨
- 클래스 간 경계가 모호해 기본 피처만으로는 분류가 어려움
- 시간축 및 도메인 메타데이터가 제한됨
- 모델 예측 확률이 특정 클래스에 쏠리거나 과신될 수 있음
## 2. 접근 전략

<img width="1121" height="508" alt="image" src="https://github.com/user-attachments/assets/712d48c0-f018-4d79-afd5-ffe561d59796" />

<br>

| 구분 | 내용 |
|---|---|
| 거리 기반 피처 | 클래스별 평균·공분산을 활용한 Mahalanobis distance feature |
| 상관쌍 파생 | 강한 상관을 갖는 센서쌍에 대해 합/차 피처 생성 |
| 방향 정규화 | 센서별 상승/하강 방향성을 통일해 임계 패턴 수치화 |
| 극단치 플래그 | train 기준 5/95 분위수 기반 상·하단 극단 여부 반영 |
| 판별 공간 투영 | LDA feature로 클래스 분리축 추가 |
| 클러스터 거리 | KMeans 중심까지의 거리로 국소 패턴 보강 |
| 모델 앙상블 | LightGBM + tabular neural network 계열 모델 확률 앙상블 |
| 후처리 | 스위치형 anomaly 규칙 보정 및 혼동군 균등화 |

## 3. 저장소 구조

<pre>
maritime-anomaly-signal-diagnosis/
├─ configs/
│  └─ default.yaml
├─ src/
│  └─ anomaly_diagnosis/
│     ├─ config.py
│     ├─ data.py
│     ├─ features.py
│     ├─ lgbm_model.py
│     ├─ torch_models.py
│     ├─ train_torch.py
│     ├─ ensemble.py
│     ├─ postprocess.py
│     └─ pipeline.py
├─ scripts/
│  ├─ train_pipeline.py
│  └─ make_submission.py
├─ docs/
├─ tests/
├─ reports/
├─ README.md
└─ requirements.txt
</pre>

## 4. 실행 방법

### 4.1 데이터 배치

대회 데이터는 저장소에 포함하지 않습니다. 아래 경로에 직접 배치합니다.

<pre>
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
</pre>

train.csv는 ID, target, X_01 ~ X_52 컬럼을 포함해야 합니다.

### 4.2 패키지 설치

<pre>
pip install -r requirements.txt
</pre>

### 4.3 전체 파이프라인 실행

<pre>
python scripts/train_pipeline.py --config configs/default.yaml
</pre>

실행 결과는 outputs/ 아래에 저장됩니다.

<pre>
outputs/
├─ oof_lgbm.npy
├─ test_lgbm.npy
├─ oof_torch.npy
├─ test_torch.npy
├─ oof_final.npy
├─ test_final.npy
├─ metrics.json
└─ submission.csv
</pre>

## 5. 후처리 규칙

### Rule 1. Switch-type anomaly correction

특정 센서의 고레벨/저레벨 plateau 전이가 target 2와 target 6에서 관찰되었습니다. 이를 반영해 다음 규칙을 적용합니다.

<pre>
X_16 >= 0.8 또는 X_18 >= 0.8 -> label 2
X_26 <= 0.3 또는 X_30 <= 0.3 -> label 6
</pre>

동시 트리거 시에는 모델 확률이 더 높은 클래스를 선택합니다.

### Rule 2. 0/15 혼동군 균등화

OOF 분석에서 특정 혼동군에 대해 클래스 0 과대, 클래스 15 과소 경향이 관찰될 수 있습니다. 따라서 모델 예측이 0 또는 15인 샘플에 한해 확률 마진이 작은 샘플부터 일부 라벨을 교체해 분포 편향을 완화합니다.

## 6. 설계 의도

이 프로젝트는 단일 모델 성능보다 다음을 중점적으로 보여주도록 구성했습니다.

- 센서 패턴 기반 EDA를 피처 엔지니어링으로 연결
- 거리 기반 모델과 신경망 계열 모델의 상보성 활용
- OOF 기반 일반화 성능 추정
- 확률 앙상블과 규칙 기반 후처리 분리
- 노트북 중심 코드를 재사용 가능한 모듈형 파이프라인으로 전환
