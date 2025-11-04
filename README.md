# 🔥 Melting Tank LSTM Baseline

**Project Goal**  
LSTM 기반의 용해탱크 공정 데이터를 활용하여 정상(OK) / 불량(NG) 상태를
예측하는 베이스라인 모델을 구축한다.


## 1. Project Overview

-   **데이터 주기:** 6초 간격으로 측정된 시계열 센서 데이터
-   **분석 목적:** 정상/불량 여부(TAG)를 분류
-   **입력 변수:**
    -   `MELT_TEMP`: 용해 온도(℃)\
    -   `MOTORSPEED`: 교반 모터 회전 속도(RPM)\
-   **출력 변수:**
    -   `TAG`: 품질 판정 결과 (OK=1, NG=0)


## 2. Model Architecture

-   **모델 유형:** LSTM (Long Short-Term Memory)
-   **입력 구조:** (window_size, feature)
-   **학습 방식:**
    -   MinMax 정규화\
    -   SMOTE를 통한 클래스 불균형 보정\
    -   EarlyStopping + ModelCheckpoint 콜백 사용\
    -   Adam Optimizer, Binary Crossentropy Loss


## 3. Training Configuration

| 항목 | 설정값 |
|------|--------|
| Epochs | 200 |
| Batch size | 50 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Metrics | Accuracy |
| Validation split | 30% |
| Callbacks | EarlyStopping, Checkpoint(all & best), ReduceLROnPlateau |


## 4. Evaluation Results

| Metric | Score | Interpretation |
|---------|--------|----------------|
| Precision | 0.996 | 오탐 최소화 (보수적 모델) |
| Recall | 0.806 | 정상 제품 중 약 80%를 탐지 |
| Accuracy | 0.805 | 전체 예측 정확도 약 80% |
| F1-score | 0.891 | 정밀도 중심의 균형 모델 |

**Confusion Matrix**

    [[ 2155   785]
     [47955 199655]]


## 5. Key Findings

- 모델은 **불량 검출에 매우 신중한(Precision 중심)** 경향을 보임
- Recall을 조금 높여 **정상 제품 분류의 재현율 개선**이 필요함
- 과적합 없이 안정적으로 수렴 (`val_loss` \< `train_loss`)


## 6. Environment Setup

### 🔹 Install dependencies

``` bash
pip install -r requirements.txt
```

### 🔹 Run notebook

``` bash
jupyter notebook Baseline-lstm.ipynb
```


## 7. File Structure

    MELTING-TANK-ML/
    ├─ checkpoints/              # Epoch별 모델 저장
    ├─ model/
    │  └─ best_model.keras       # 최고 성능 모델
    ├─ Baseline-lstm.ipynb       # 노트북 파일
    ├─ requirements.txt
    └─ README.md


## 8. Reference
Ministry of SMEs and Startups., and KAIST(Korea Advanced Institute of Science and
Technology). (2020, December 14). Melting tank AI dataset. Korea AI Manufacturing
Platform(KAMP). https://www.kamp-ai.kr/
