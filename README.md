# Diamond Price Predictor 💎

XGBoost-powered Streamlit app — R² 98.1%

## Project structure

```
diamond_app/
├── train.py          ← run once to train & save the model
├── app.py            ← Streamlit UI (loads .pkl files)
├── requirements.txt
├── diamonds.csv      ← put your dataset here (for training only)
└── model/            ← auto-created by train.py
    ├── model.pkl
    ├── encoder.pkl
    └── scaler.pkl
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — train (run once)

Put `diamonds.csv` next to `train.py`, then:

```bash
python train.py
```

This saves `model/model.pkl`, `model/encoder.pkl`, `model/scaler.pkl`.

## Step 2 — run the app

```bash
streamlit run app.py
```

> The CSV is only needed for training.
> Deploy the app with just `app.py`, `requirements.txt`, and the `model/` folder.
