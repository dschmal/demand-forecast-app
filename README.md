# Demand Forecast Visualizer (Streamlit)

This is a **deploy-safe** Streamlit demand forecasting app built as a **single file** (`app.py`) to avoid import/path issues on Streamlit Cloud / GitLab deployments.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data format

Upload a CSV with:
- a date column (default: `date`)
- a demand/target column (default: `demand`)

## Deployment notes

- Keep `app.py` and `requirements.txt` at the repository root.
- Prophet is commented out by default in `requirements.txt` because it can fail to build on some environments.
  If you want Prophet, uncomment it and redeploy.

## Docker (optional)

```bash
docker build -t demand-forecast .
docker run -p 8501:8501 demand-forecast
```
