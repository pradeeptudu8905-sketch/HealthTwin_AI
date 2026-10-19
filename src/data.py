import numpy as np
import pandas as pd

def make_synthetic_data(n=1800, seed=42):
    """Create fully synthetic EHR + wearable examples for a demo only."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 86, n)
    prior_diabetes = rng.binomial(1, np.clip((age - 25) / 100, 0.05, 0.65), n)
    hba1c = np.clip(rng.normal(5.6 + 0.9 * prior_diabetes, 0.8, n), 4.2, 11.5)
    resting_hr = np.clip(rng.normal(72 + 3 * prior_diabetes, 11, n), 42, 135)
    glucose = np.clip(rng.normal(105 + 30 * prior_diabetes + 9 * (hba1c - 5.5), 28, n), 55, 320)
    sleep_hours = np.clip(rng.normal(7.1 - 0.45 * prior_diabetes, 1.25, n), 2, 12)
    steps = np.clip(rng.normal(6500 - 900 * prior_diabetes, 2600, n), 100, 22000)
    hrv = np.clip(rng.normal(48 - 4 * prior_diabetes - 0.12 * (resting_hr - 70), 14, n), 8, 120)
    # Transparent synthetic rule: not a clinically validated label.
    logit = (-5.0 + 0.035 * (glucose - 110) + 0.65 * prior_diabetes
             + 0.45 * (hba1c - 5.5) + 0.018 * (resting_hr - 70)
             + 0.16 * (7 - sleep_hours) - 0.000035 * (steps - 5000)
             - 0.008 * (hrv - 45))
    probability = 1 / (1 + np.exp(-logit))
    event = rng.binomial(1, probability)
    return pd.DataFrame({
        "age": age,
        "prior_diabetes": prior_diabetes,
        "hba1c": np.round(hba1c, 1),
        "resting_hr": np.round(resting_hr, 1),
        "glucose": np.round(glucose, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "steps": steps.astype(int),
        "hrv": np.round(hrv, 1),
        "event_next_2h": event
    })

def make_patient_timeline(patient, hours=24, seed=7):
    """Create a simulated time series around a chosen patient's current values."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=hours, freq="h")
    glucose = np.clip(rng.normal(patient["glucose"], 12, hours) + np.linspace(-5, 8, hours), 55, 320)
    heart_rate = np.clip(rng.normal(patient["resting_hr"], 5, hours), 42, 150)
    hrv = np.clip(rng.normal(patient["hrv"], 7, hours), 8, 120)
    return pd.DataFrame({"time": times, "glucose": glucose, "heart_rate": heart_rate, "hrv": hrv})
