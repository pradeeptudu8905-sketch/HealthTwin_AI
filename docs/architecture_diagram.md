# HealthTwin AI Architecture

Paste the Mermaid diagram below into a Mermaid-compatible editor and export it as PDF or include it in your presentation.

```mermaid
flowchart TD
    A[ Synthetic EHR: age, history, HbA1c ] --> C[ Data preparation ]
    B[ Simulated wearable stream: glucose, HR, HRV, sleep, steps ] --> C
    C --> D[ Feature assembly ]
    D --> E[ Random Forest demo model ]
    E --> F[ Estimated event risk ]
    F --> G[ Streamlit doctor-facing dashboard ]
    G --> H[ Trend charts, virtual patient profile, risk alert ]
```

All data is synthetic. The model is a demonstrator and is not clinically validated.
