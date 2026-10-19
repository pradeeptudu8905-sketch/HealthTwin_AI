
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

from .data import make_synthetic_data

FEATURES = [
    "age",
    "prior_diabetes",
    "hba1c",
    "resting_hr",
    "glucose",
    "sleep_hours",
    "steps",
    "hrv",
]


def train_demo_model(seed=42):
    # Generate synthetic patient records.
    data = make_synthetic_data(seed=seed)

    X = data[FEATURES]
    y = data["event_next_2h"]

    # Keep a separate test set for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # Train the model only on training data.
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
    )

    model.fit(X_train, y_train)

    # Evaluate predictions on unseen test data.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    test_positive = int(y_test.sum())
    test_negative = int(len(y_test) - test_positive)

    print("Test-set size:", len(y_test))
    print("Actual positive events:", test_positive)
    print("Actual negative events:", test_negative)
    print("Predicted positive events:", int(y_pred.sum()))




    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ),
    }

    return model, metrics, data


def estimate_risk(model, patient):
    import pandas as pd

    row = pd.DataFrame([
        {feature: patient[feature] for feature in FEATURES}
    ])

    return float(model.predict_proba(row)[0, 1])
