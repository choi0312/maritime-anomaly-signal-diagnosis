# Modeling Strategy

## 1. Feature Engineering

- Mahalanobis distance: 클래스별 평균과 공분산을 반영한 거리 기반 피처
- Correlation pair features: 강한 양/음 상관쌍에 대해 합/차 파생
- Clear-shift align: 센서별 방향성을 통일해 임계형 anomaly 탐지 보조
- Extreme flags: 5/95 분위수 기준 극단치 탐지
- LDA features: 선형 판별축을 추가해 클래스 간 분리도 강화
- KMeans distances: 클러스터 중심까지의 거리로 국소 패턴 보강

## 2. Models

- LightGBM DART: 임계형·규칙형·거리형 피처에 강한 트리 기반 모델
- Torch tabular models: 연속적인 센서 상호작용과 비선형 패턴 보완

## 3. Ensemble

LightGBM은 거리·임계 기반 신호에 강하고, 신경망 계열 모델은 연속적 상호작용과 국소 패턴에 민감하다. 두 모델군의 확률을 가중 평균하여 편향을 상쇄한다.

## 4. Postprocessing

후처리는 모델 학습과 분리해 적용한다. 스위치형 규칙은 명확한 센서 임계 패턴에만 사용하고, 혼동군 균등화는 모델 확률 마진이 작은 샘플에 제한적으로 적용한다.
