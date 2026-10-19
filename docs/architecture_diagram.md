
# HealthTwin AI — System Architecture

## 1. Architecture Overview

```mermaid
flowchart TD
    A["Synthetic Historical Health Record<br/>Age, Diabetes History, HbA1c"]
    B["Simulated Wearable Data<br/>Glucose, Heart Rate, HRV, Sleep, Steps"]
    C["Data Preparation and Validation"]
    D["Feature Assembly"]
    E["Random Forest Machine Learning Model"]
    F["Estimated Elevated-Glucose Event Risk"]
    G["Streamlit Dashboard"]
    H["Patient Profile, KPI Cards, Trend Charts<br/>and Risk Indicator"]
    I["Model Evaluation<br/>Accuracy, Precision, Recall, F1, ROC-AUC"]

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    E --> I

    classDef input fill:#e8f2ff,stroke:#2878b5,color:#123;
    classDef processing fill:#fff0d9,stroke:#d68b00,color:#321;
    classDef model fill:#e8e4ff,stroke:#7656c8,color:#213;
    classDef output fill:#ddf7e7,stroke:#238653,color:#123;

    class A,B input;
    class C,D processing;
    class E,I model;
    class F,G,H output;
```

## 2. Main Components

- **Synthetic health record:** Contains historical patient attributes.
- **Wearable data simulator:** Generates sample readings for the dashboard.
- **Data preparation:** Organizes the inputs for model prediction.
- **Feature assembly:** Combines the relevant model inputs.
- **Machine learning model:** Uses a Random Forest classifier to estimate event risk.
- **Streamlit dashboard:** Displays patient information, trends, and the estimated risk.
- **Model evaluation:** Reports performance on a held-out synthetic test set.

## 3. Data Flow

Historical record + simulated wearable readings → Data preparation → Feature assembly → Random Forest model → Risk estimate → Streamlit dashboard.

## 4. Limitations and Safety

This project is an educational proof of concept. All patient records and wearable readings are synthetic. The model has not been clinically validated and must not be used for medical diagnosis, treatment, or real-world clinical decisions.
