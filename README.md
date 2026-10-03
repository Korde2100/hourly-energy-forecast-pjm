# Hourly Energy Consumption Forecast (PJM West Region)

## Project Overview
This repository contains the deliverables for the capstone project **P-702 / P679** – *Hourly Energy Consumption Forecast* for the PJM West transmission zone. The goal is to predict hourly electricity demand (MW) to aid grid balancing, scheduling, and risk mitigation.

### Team
- **Mentor:** K. Dilavar Bassha
- **Team Members:**
  - Palem Soma Sekhar
  - Seetha Reddiar
  - Mallela Somanath
  - Vishal Parmar
  - Ailaveni Praveen
  - Richa Amardas Deshmukh
  - Dhanashree Korde

## Key Features
- **Data:** 143,202 hourly observations (Apr 2002 – Aug 2018) from PJM.
- **Feature Engineering:** 16 temporal features – calendar indicators, US federal holidays, lag variables (1h, 2h, 3h, 24h, 48h, 168h), and rolling statistics (24‑h/168‑h moving averages & std dev).
- **Model:** XGBoost Regressor (best performance).
  - **MAE:** 52.09 MW
  - **RMSE:** 70.28 MW
  - **MAPE:** 0.91 %
  - **R²:** 0.9950
- **Deployment:** Streamlit web app with recursive 30‑day (720‑hour) forecasting, interactive charts, daily aggregations, and CSV download.

## Repository Structure
```
├── app.py                         # Streamlit deployment script
├── PJMW_Hourly_Energy_Consumption.ipynb (or .py)  # Data cleaning, EDA, feature engineering, model training
├── PJMW_MW_Hourly.xlsx            # Raw historical dataset
├── PJMW_Final_30_Day_Forecast.csv # 720‑hour forecast (hourly)
├── PJMW_Final_30_Day_Daily_Summary.csv # Daily aggregated metrics
├── Deployment_Screenshot.pdf      # Screenshot of the Streamlit app
├── Hourly_Energy_Consumption_Forecast_25_Slide_PPT.pptx # Project presentation deck (25 slides)
├── Requirement documenct.docx     # Requirement specification document
└── README.md                      # This file
```

## Setup & Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your‑username>/<repo‑name>.git
   cd <repo‑name>
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   .\\venv\\Scripts\\activate   # Windows PowerShell
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # (Create this file if needed)
   ```
   The main packages are:
   - `pandas`, `numpy`
   - `scikit‑learn`
   - `xgboost`
   - `streamlit`
   - `openpyxl` (for reading the Excel dataset)
4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   The dashboard will be available at `http://localhost:8501`.

## Usage
- **Exploratory analysis** can be reviewed in the Jupyter notebook `PJMW_Hourly_Energy_Consumption.ipynb`.
- **Model training** is performed inside the notebook; the trained model (`model.xgb` or similar) is loaded by `app.py`.
- **Forecasting**: The Streamlit UI lets you generate a 30‑day forecast, view interactive plots, and download the results as CSV.

## Demo Video
A walkthrough of the project, including EDA, feature construction, model benchmarking, and the live Streamlit dashboard, is available here:
[Project Demo Video](https://drive.google.com/drive/folders/1knlSg_PHwI06y4oUOPM43y5-6KIL-Vy2?usp=sharing)

## License & Citation
This work is for academic purposes and may be shared for educational use with proper attribution to the team and mentor.

---
*Generated automatically by Antigravity AI assistant.*
