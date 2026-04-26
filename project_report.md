# 스마트 모빌리티를 위한 실내외 통합 노면 상태 감지 및 실시간 관제 시스템 구축 연구
## (Project Factory-Twin Guard: Integrated Perception & Monitoring System)

### 1. 서론

- 연구 배경: 스마트 물류 및 자율주행 모빌리티의 주행 안정성 확보와 장비 수명 연장을 위한 노면 환경 인지 기술의 필요성 증대.

- 연구 목적: IMU 센서 데이터를 활용한 범용 노면 분류 및 통계적 이상 탐지 알고리즘 개발과 실시간 웹 관제 시스템 구축.

---

### 2. 시스템 아키텍처

- 데이터 계층: 센서 원시 데이터(Quaternion, Accel)의 물리적 변수 변환 및 전처리.

- 추론 계층: XGBoost, RandomForest, DecisionTree 기반 재질 분류와 3-Sigma 기반 이상 탐지 하이브리드 엔진.

- 관제 계층: Flask 기반 API 서버 및 실시간 대시보드 인터페이스 구현.

---

### 3. 데이터 분석 및 통찰

#### 3.1 원시 데이터 물리적 특성 분석

<p align="left">
  <img src="./SSP/reports/figures/indoor_raw/01_raw_timeseries_per_surface.png" width="32.5%">
  <img src="./SSP/reports/figures/indoor_raw/02_raw_quaternion_corr.png" width="32.5%">
  <img src="./SSP/reports/figures/indoor_raw/03_raw_accel_distribution.png" width="32.5%">
</p>

[그림 3-1] 실내 원시 데이터 통합 분석 (시계열 / 쿼터니언 상관관계 / 가속도 분포)

- 데이터 패턴 및 상관관계: 시계열 분석 결과 노면별 진동 특성이 관찰되나, 쿼터니언 축 간 상관계수가 0.95를 상회하는 강한 다중공선성을 확인하여 오일러 각 변환의 필요성을 입증함.

- 분포 특성: 원시 가속도 데이터의 중첩이 모든 노면에서 심하게 나타나, 단일 시점의 값보다는 시간 영역의 통계적 특징 추출이 분류의 핵심임을 확인함.

#### 3.2 실외 이상 탐지 특성

<p align="left">
  <img src="./SSP/reports/figures/outdoor_eda/label_distribution.png" width="32.5%">
  <img src="./SSP/reports/figures/outdoor_eda/acc_z_max_distribution.png" width="32.5%">
  <img src="./SSP/reports/figures/outdoor_eda/acc_z_std_boxplot.png" width="32.5%">
</p>

[그림 3-2] 실외 결함 데이터 분포 및 충격량 특성

- 클래스 불균형: 정상 주행 대비 포트홀 발생 빈도가 극히 낮은 불균형 구조를 확인하여 재현율(Recall) 중심의 학습 전략을 수립함.

- 핵심 지표 식별: EDA 결과, 포트홀 데이터는 정상 주행 데이터 대비 Z축 가속도의 표준편차와 최댓값이 유의미하게 높게 나타남을 확인하여, 이를 이상 탐지 임계값의 근거로 활용함.

---

### 4. 분석 방법론 및 전처리

#### 4.1.1 실내 데이터 전처리 파이프라인

1. 차원 재구성: 쿼터니언 데이터를 Roll, Pitch, Yaw로 변환하여 다중공선성 해소 및 물리적 직관성 확보.

2. 특징 생성: 가속도/각속도 벡터 크기 및 변화량 변수를 생성하여 동적 주행 특성 강화.

3. 통계적 요약: 128-Step 윈도우 기반 8종 통계량 산출로 노면별 고유 진동 패턴 정량화.

#### 4.1.2 3-Sigma 기반 동적 임계값 산출

- 분석 방법론: 재질별 가속도 평균($\mu$)과 표준편차($\sigma$) 기반의 [$\mu \pm 3\sigma$] 범위를 정상 주행 구간으로 정의하여 실시간 대시보드와 연동.

#### 4.1.3 전처리 데이터 분석 및 모델 설계 근거

<p align="left">
  <img src="./SSP/reports/figures/indoor_eda/01_target_distribution.png" width="49%">
  <img src="./SSP/reports/figures/indoor_eda/02_feature_distributions.png" width="49%">
</p>

[그림 4-1] 클래스 분포 및 특징별 박스플롯 분석

- 클래스 불균형 분석: 노면별 샘플 분포 확인 결과, `hard_tiles` 및 `carpet` 클래스의 데이터 부족이 명확히 관찰됨. 이는 학습 시 다수 클래스 편향을 야기하므로 SMOTE 및 Class Weighting 전략 도입의 기술적 근거가 됨.

- 데이터 복잡성 분석: 박스플롯 결과 클래스 간 중앙값 차이는 존재하나 사분위 범위(IQR)의 중첩이 심하고 다수의 이상치가 관찰됨. 이는 단순 선형 분류의 한계를 의미하며, 이상치에 강인한 앙상블 모델의 필요성을 입증함.

<br>

<p align="left">
  <img src="./SSP/reports/figures/indoor_eda/03_correlation_heatmap.png" width="49%">
  <img src="./SSP/reports/figures/indoor_eda/04_pca_scatter.png" width="49%">
</p>

[그림 4-2] 특징 간 상관관계 및 PCA 차원 축소 분석

- 다중공선성 및 차원 축소: 특징 간 상관계수 0.99 기록 등 높은 변수 중복성을 확인하였으며, PCA 투영 시의 낮은 변별력(분산비 42.7%)과 클래스 중첩 현상은 비선형 기반 고차원 분류 모델 도입의 타당성을 뒷받침함.


#### 4.2.1 실외 데이터 전처리 파이프라인

1. 물리적 공간 라벨링: 
   Haversine 거리 공식을 적용하여 포트홀 반경 5m 이내 데이터를 정밀 라벨링.

2. 물리적 특성 기반 피처 생성: 
   Z축 가속도 편향 제거 및 속도 정규화(Normalized Accel) 변수를 생성하여 주행 속도 변화에 따른 충격 왜곡 해소.

3. 통계적 요약: 
   20-Step 슬라이딩 윈도우 기반 통계량(왜도, 첨도) 및 주파수 도메인 에너지(FFT) 산출로 불규칙 충격 패턴 정량화. 

4. 교차검증: 
   LOTO 방식을 적용하여 모델의 미학습 경로에 대한 일반화 성능을 객관적으로 검증


#### 4.2.2 전처리 데이터 분석 및 모델 설계 근거

<p align="center">
  <img src="./SSP/reports/figures/outdoor_eda/correlation_heatmap.png" width="50%">
</p>

[그림 4-3] 피처 간 상관관계 

- 다중공선성 및 피처 최적: 
  특징 간 상관계수 히트맵 분석 결과 높은 양의 상관관계가 확인됨. 이는 변수 간의 중복 정보를 의미하므로, 다중공선성 문제를 해소하고 모델의 연산 효율성을 높이기 위해 상관계수가 높은 피처를 선별적으로 제거하는 최적화 과정을 수행함.



<p align="center">
  <img src="./SSP/reports/figures/outdoor_eda/speed_vs_acc_z_std_scatter.png" width="50%">
</p>

[그림 4-4] 속도와 수직 가속도 간의 비선형성 및 이분산성 분석

- 데이터 비선형성 진단 및 클래스 중첩 해소를 위한 변수 정규화:
  속도와 수직 가속도 간의 산점도 분석 결과, 상관관계 히트맵 분석 결과 포착되지 않은 비선형적 종속 관계와 이분산성, 클래스 중첩 문제가 발견됨.

$$
\begin{aligned}
E &= \frac{1}{2}mv^2 \\
F \cdot \Delta t &= m \cdot \Delta v
\end{aligned}
$$

  물리 법칙을 근거로 속도 편향(Speed Bias)을 제거하기 위해 속도 정규화 작업을 수행함.

$$
\text{Normalized Accel} = \frac{\text{Accelerometer Z}}{\text{Mean Speed}}
$$


#### 4.2.3 데이터셋 최적화

<p align="center">
  <img src="./SSP/reports/figures/outdoor_eda/data_improvement.png" width="50%">
</p>

[그림 4-5] 성능 개선 추이

실외 주행 데이터의 비선형적 종속성 및 데이터 불균형 문제를 해소하기 위해 가중치 조절과 복잡한 패턴 학습에 능한 XGBoost 알고리즘 활용. 

희소 결함 탐지의 특성을 고려하여 Recall, F1-score, PR-AUC 지표들을 종합적으로 고려하여 D5 데이터셋을 최종 선택.


---

### 5. 모델 구축 및 성능 평가

#### 5.1 실내 노면 판별 모델 고도화 및 최적화 (Indoor Model Optimization)

1. 알고리즘 벤치마킹 및 비선형 모델군 선정:

| 알고리즘 기초 성능 비교 (시각화) | 모델별 상세 지표 (Baseline 데이터) |
| :--- | :--- |
| ![그림 5-1](./SSP/reports/figures/indoor_performance/loaded_model_metrics.png) | [XGBoost]<br>Accuracy: 0.864 / Precision: 0.872 / Recall: 0.866 / F1-Score: 0.866<br><br>[RandomForest]<br>Accuracy: 0.743 / Precision: 0.751 / Recall: 0.731 / F1-Score: 0.731<br><br>[DecisionTree]<br>Accuracy: 0.651 / Precision: 0.662 / Recall: 0.640 / F1-Score: 0.640<br><br>[대조군: LogisticReg / NaiveBayes / SVM]<br>F1-Score: 0.392 / 0.370 / 0.342 |

- 알고리즘 선정 근거: 초기 벤치마킹 결과 선형 모델 대비 모든 평가지표에서 압도적 우위를 점한 비선형 앙상블 3종을 고도화 후보군으로 선정함.

2. 모델별 최적 불균형 처리 전략 적용 및 후보군 압축:

| 불균형 해소 전략별 성능 분포 | 전략별 F1-Score 대조 및 최적 전략 |
| :--- | :--- |
| ![그림 5-2](./SSP/reports/figures/indoor_performance/imbalance_strategy_comparison.png) | [DecisionTree]<br>Baseline: 0.640 / ClassWeight: 0.681 / SMOTE: 0.667<br>→ 최적 전략: DecisionTree(ClassWeight)<br><br>[RandomForest]<br>Baseline: 0.731 / ClassWeight: 0.791 / SMOTE: 0.820<br>→ 최적 전략: RandomForest(SMOTE)<br><br>[XGBoost]<br>Baseline: 0.866 / ClassWeight: 0.857 / SMOTE: 0.862<br>→ 최적 전략: XGBoost(Baseline) |

- 전략 채택 결과: 클래스 불균형 해소 전략 적용 시 RandomForest는 SMOTE를 통해 약 9%의 성능 향상을 기록하였으며, 각 모델별 최적 조합을 확정함.

3. 하이퍼파라미터 최적화 및 최종 성능 비교:

| 최적화 단계별 최종 성능 추이 | 고도화 단계별 성능 및 최종 향상도 (Macro F1) |
| :--- | :--- |
| ![그림 5-3](./SSP/reports/figures/indoor_performance/final_3step_evolution.png) | [XGBoost]<br>1.Baseline: 0.866 / 2.Imbalance: 0.866 / 3.Tuned: 0.863<br>→ Improvement: -0.003<br><br>[RandomForest]<br>1.Baseline: 0.731 / 2.Imbalance: 0.820 / 3.Tuned: 0.831<br>→ Improvement: +0.100<br><br>[DecisionTree]<br>1.Baseline: 0.640 / 2.Imbalance: 0.681 / 3.Tuned: 0.692<br>→ Improvement: +0.052 |

- 최종 모델 확정: 3단계 고도화 결과 XGBoost 모델이 최종 Macro F1-Score 0.863으로 실시간 판별 엔진에 가장 적합한 성능을 보임을 입증함.

#### 5.2 실외 이상 탐지 모델 (구조 설계)

1. 특징 벡터 정의: 포트홀 발생 시 수직 충격량 탐지를 위한 피크 기반 특징 추출 전략 수립.

2. 최적화 방향: 주행 안전 확보를 위해 재현율(Recall) 향상 중심의 모델 튜닝 및 임계값 조정 수행 예정.

---

### 6. 실시간 웹 관제 시스템 통합

- 백엔드 구성: Flask 기반 app.py를 활용하여 수집 데이터를 최적화 모델 및 3-Sigma 임계값과 실시간으로 대조 추론.

- 프런트엔드 기능: Jinja2 대시보드 인터페이스로 노면 상태를 시각화하고 이상 감지 시 즉각적인 시각적 경보(Alarm) 발생.

---

### 7. 결론

- 연구 성과: 기계학습 분류와 통계적 감시 기법을 결합한 강건한 통합 노면 관제 솔루션 구현 완료.

- 기대 효과: 스마트팩토리 모빌리티의 주행 안정성 확보 및 장비 유지보수 신뢰도 제고에 기여.

- 향후 계획: 실외 환경 데이터 확충을 통한 전천후 통합 감제 시스템 고도화 추진.
