# Toronto Collision Streamlit Deploy

Lightweight deployment repo for the Toronto Collision risk dashboard.

## Contents
- `app.py` — Streamlit entrypoint
- `dashboard_utils_openmeteo_live.py` — dashboard helper functions
- `models/stacking_4model_dashboard_bundle.pkl` — trained deployment model
- `data/data_raw/Neighbourhoods.geojson` — neighborhood boundaries
- `data/processed/supervised_hood_3h_multiclass.xlsx` — historical display data
- `.streamlit/config.toml` — theme config
- `requirements.txt` — deployment dependencies
- `.gitignore` — excludes local envs and unnecessary files

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Push as a new GitHub repo
```bash
cd toronto-collision-streamlit-deploy-clean
git init
git add .
git commit -m "Initial Streamlit deployment repo"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

## Streamlit Community Cloud
- Repository: your new GitHub repo
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`
