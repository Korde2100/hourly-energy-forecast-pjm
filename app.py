"""
P-702 / P679 – Hourly Energy Consumption Forecast
Streamlit deployment application.

This deployment follows the final notebook:
- Target: PJMW_MW
- Frequency: Hourly
- Train/Test split: last 1 year as test set
- Final model: XGBoost Regressor
- Features: calendar, holiday, lag and rolling features
- Forecast horizon: 30 days / 720 hours
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import holidays
except ImportError:
    holidays = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PJM Hourly Energy Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ PJM Hourly Energy Consumption Forecast")
st.markdown(
    """
    **P-702 / P679 Forecasting Project**

    This application forecasts hourly PJM electricity demand using the
    **XGBoost model selected in the project notebook**.
    """
)

# -------------------------------------------------------------------
# Constants from the notebook
# -------------------------------------------------------------------
FEATURE_COLS = [
    "Hour",
    "Day",
    "DayOfWeek",
    "Month",
    "Year",
    "IsWeekend",
    "IsHoliday",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_std_24",
]

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data
def read_dataset(source):
    """Read a CSV/Excel PJM dataset."""
    if hasattr(source, "name"):
        filename = source.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(source)
        else:
            try:
                df = pd.read_excel(source, sheet_name="PJMW_hourly")
            except Exception:
                df = pd.read_excel(source)
    else:
        path = str(source)
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            try:
                df = pd.read_excel(path, sheet_name="PJMW_hourly")
            except Exception:
                df = pd.read_excel(path)

    required = {"Datetime", "PJMW_MW"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[["Datetime", "PJMW_MW"]].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["PJMW_MW"] = pd.to_numeric(df["PJMW_MW"], errors="coerce")
    df = df.dropna(subset=["Datetime", "PJMW_MW"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    return df


# -------------------------------------------------------------------
# Notebook-aligned feature engineering
# -------------------------------------------------------------------
def add_features(df):
    """Create the same feature families used in the notebook."""
    if holidays is None:
        raise ImportError(
            "The 'holidays' package is required. Install it with: pip install holidays"
        )

    data = df.copy().sort_values("Datetime").reset_index(drop=True)

    # US holiday calendar, matching the notebook.
    start_year = int(data["Datetime"].dt.year.min())
    end_year = int(data["Datetime"].dt.year.max())
    us_holidays = holidays.US(years=range(start_year, end_year + 1))
    holiday_dates = set(us_holidays.keys())

    # Calendar features
    data["IsHoliday"] = (
        data["Datetime"].dt.date.isin(holiday_dates).astype(int)
    )
    data["Hour"] = data["Datetime"].dt.hour
    data["Day"] = data["Datetime"].dt.day
    data["DayOfWeek"] = data["Datetime"].dt.dayofweek
    data["Month"] = data["Datetime"].dt.month
    data["Year"] = data["Datetime"].dt.year
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

    # Lag features – exactly as used in the notebook.
    data["lag_1"] = data["PJMW_MW"].shift(1)
    data["lag_2"] = data["PJMW_MW"].shift(2)
    data["lag_3"] = data["PJMW_MW"].shift(3)
    data["lag_24"] = data["PJMW_MW"].shift(24)
    data["lag_48"] = data["PJMW_MW"].shift(48)
    data["lag_168"] = data["PJMW_MW"].shift(168)

    # Rolling features – shifted first to avoid using the current target.
    data["rolling_mean_24"] = (
        data["PJMW_MW"].shift(1).rolling(24).mean()
    )
    data["rolling_mean_168"] = (
        data["PJMW_MW"].shift(1).rolling(168).mean()
    )
    data["rolling_std_24"] = (
        data["PJMW_MW"].shift(1).rolling(24).std()
    )

    # Notebook drops rows made incomplete by lag/rolling calculations.
    data = data.dropna(subset=FEATURE_COLS + ["PJMW_MW"]).reset_index(drop=True)

    return data, holiday_dates


# -------------------------------------------------------------------
# Train/test split – last 1 year as test
# -------------------------------------------------------------------
def split_last_year(data):
    last_date = data["Datetime"].max()
    test_start = last_date - pd.DateOffset(years=1)

    train_df = data[data["Datetime"] < test_start].copy()
    test_df = data[data["Datetime"] >= test_start].copy()

    return train_df, test_df


# -------------------------------------------------------------------
# Model training
# -------------------------------------------------------------------
@st.cache_resource
def train_xgb(train_x, train_y):
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Install it with: pip install xgboost"
        )

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(train_x, train_y)
    return model


# -------------------------------------------------------------------
# Recursive 30-day forecasting
# -------------------------------------------------------------------
def create_future_features(history, holiday_dates):
    """Create one future feature row using the notebook's logic."""
    current_time = (
        history["Datetime"].iloc[-1] + pd.Timedelta(hours=1)
    )

    row = {
        "Hour": current_time.hour,
        "Day": current_time.day,
        "DayOfWeek": current_time.dayofweek,
        "Month": current_time.month,
        "Year": current_time.year,
        "IsWeekend": int(current_time.dayofweek in [5, 6]),
        "IsHoliday": int(current_time.date() in holiday_dates),
        "lag_1": history["PJMW_MW"].iloc[-1],
        "lag_2": history["PJMW_MW"].iloc[-2],
        "lag_3": history["PJMW_MW"].iloc[-3],
        "lag_24": history["PJMW_MW"].iloc[-24],
        "lag_48": history["PJMW_MW"].iloc[-48],
        "lag_168": history["PJMW_MW"].iloc[-168],
        "rolling_mean_24": history["PJMW_MW"].iloc[-24:].mean(),
        "rolling_mean_168": history["PJMW_MW"].iloc[-168:].mean(),
        "rolling_std_24": history["PJMW_MW"].iloc[-24:].std(),
    }

    return pd.DataFrame([row]), current_time


def recursive_forecast(model, history_df, holiday_dates, hours=720):
    """Generate a recursive hourly forecast."""
    future_history = history_df[["Datetime", "PJMW_MW"]].copy()
    future_predictions = []

    progress = st.progress(0, text="Generating 30-day forecast...")

    for i in range(hours):
        X_future, future_time = create_future_features(
            future_history, holiday_dates
        )

        prediction = float(model.predict(X_future[FEATURE_COLS])[0])

        future_predictions.append(
            {
                "Datetime": future_time,
                "Predicted_MW": prediction,
            }
        )

        future_history = pd.concat(
            [
                future_history,
                pd.DataFrame(
                    {
                        "Datetime": [future_time],
                        "PJMW_MW": [prediction],
                    }
                ),
            ],
            ignore_index=True,
        )

        if i % 10 == 0 or i == hours - 1:
            progress.progress(
                (i + 1) / hours,
                text=f"Generating forecast: {i + 1}/{hours} hours",
            )

    progress.empty()
    return pd.DataFrame(future_predictions)


# -------------------------------------------------------------------
# Optional local default dataset
# -------------------------------------------------------------------
def find_local_dataset():
    candidates = [
        "PJMW_MW_Hourly.xlsx",
        "PJMW_hourly.xlsx",
        "PJMW_hourly.csv",
    ]

    for name in candidates:
        if Path(name).exists():
            return name

    return None


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.header("Forecast Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload PJM dataset",
    type=["csv", "xlsx"],
)

forecast_days = st.sidebar.number_input(
    "Forecast horizon (days)",
    min_value=1,
    max_value=30,
    value=30,
    step=1,
)

st.sidebar.info(
    "Final notebook model: XGBoost\n\n"
    "Forecast horizon: 30 days / 720 hours"
)

# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
try:
    if uploaded_file is not None:
        raw_df = read_dataset(uploaded_file)
    else:
        local_path = find_local_dataset()
        if local_path is None:
            st.info(
                "Please upload PJMW_MW_Hourly.xlsx or PJMW_hourly.csv "
                "from the sidebar to start the forecast."
            )
            st.stop()
        raw_df = read_dataset(local_path)

except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------
st.subheader("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Raw observations", f"{len(raw_df):,}")

with c2:
    st.metric(
        "Start",
        raw_df["Datetime"].min().strftime("%Y-%m-%d"),
    )

with c3:
    st.metric(
        "End",
        raw_df["Datetime"].max().strftime("%Y-%m-%d"),
    )

with c4:
    st.metric(
        "Average Load",
        f"{raw_df['PJMW_MW'].mean():,.2f} MW",
    )

st.caption(
    "The notebook dataset contains the columns Datetime and PJMW_MW. "
    "Feature engineering follows the final notebook workflow."
)

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
try:
    with st.spinner("Preparing time-series and forecasting features..."):
        model_df, holiday_dates = add_features(raw_df)

except Exception as exc:
    st.error(f"Feature engineering failed: {exc}")
    st.stop()

st.subheader("2. Feature Engineering")

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.metric("Modeling observations", f"{len(model_df):,}")

with f2:
    st.metric("Forecast features", len(FEATURE_COLS))

with f3:
    st.metric("US holiday dates", f"{len(holiday_dates):,}")

with f4:
    st.metric("Frequency", "Hourly")

with st.expander("View feature list"):
    st.write(FEATURE_COLS)

# -------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------
train_df, test_df = split_last_year(model_df)

if len(train_df) == 0 or len(test_df) == 0:
    st.error("The dataset is not long enough to create the notebook's 1-year test split.")
    st.stop()

st.subheader("3. Train / Test Split")

s1, s2 = st.columns(2)

with s1:
    st.write(
        f"**Training:** {train_df['Datetime'].min():%Y-%m-%d} "
        f"to {train_df['Datetime'].max():%Y-%m-%d}"
    )
    st.write(f"Training observations: **{len(train_df):,}**")

with s2:
    st.write(
        f"**Testing:** {test_df['Datetime'].min():%Y-%m-%d} "
        f"to {test_df['Datetime'].max():%Y-%m-%d}"
    )
    st.write(f"Testing observations: **{len(test_df):,}**")

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
st.subheader("4. Final Forecasting Model")

st.success(
    "XGBoost Regressor — selected as the final model in the notebook "
    "because it provided the lowest overall prediction errors."
)

st.code(
    """XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)""",
    language="python",
)

# Train model using the same train/test setup.
try:
    with st.spinner("Training XGBoost model..."):
        X_train = train_df[FEATURE_COLS]
        y_train = train_df["PJMW_MW"]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df["PJMW_MW"]

        model = train_xgb(X_train, y_train)

except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

# -------------------------------------------------------------------
# Test-set evaluation
# -------------------------------------------------------------------
test_pred = model.predict(X_test)

mae = float(np.mean(np.abs(y_test.values - test_pred)))
rmse = float(np.sqrt(np.mean((y_test.values - test_pred) ** 2)))
mape = float(
    np.mean(np.abs((y_test.values - test_pred) / y_test.values)) * 100
)
ss_res = float(np.sum((y_test.values - test_pred) ** 2))
ss_tot = float(np.sum((y_test.values - y_test.values.mean()) ** 2))
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

st.subheader("5. Model Performance on Test Set")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("MAE", f"{mae:,.2f} MW")

with m2:
    st.metric("RMSE", f"{rmse:,.2f} MW")

with m3:
    st.metric("MAPE", f"{mape:.2f}%")

with m4:
    st.metric("R²", f"{r2:.4f}")

# -------------------------------------------------------------------
# Actual vs predicted chart
# -------------------------------------------------------------------
st.subheader("6. Actual vs XGBoost Prediction")

comparison_plot = pd.DataFrame(
    {
        "Actual": y_test.values,
        "XGBoost Prediction": test_pred,
    },
    index=test_df["Datetime"],
)

# Show the most recent 14 days to keep the Streamlit chart readable.
st.line_chart(comparison_plot.tail(24 * 14))

# -------------------------------------------------------------------
# Feature importance
# -------------------------------------------------------------------
st.subheader("7. XGBoost Feature Importance")

importance = pd.DataFrame(
    {
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_,
    }
).sort_values("Importance", ascending=False)

st.dataframe(
    importance.round(4),
    use_container_width=True,
    hide_index=True,
)

# -------------------------------------------------------------------
# Forecast generation
# -------------------------------------------------------------------
st.subheader("8. 30-Day Hourly Forecast")

if st.button("Generate Forecast", type="primary"):
    forecast_hours = int(forecast_days) * 24

    try:
        forecast = recursive_forecast(
            model=model,
            history_df=raw_df,
            holiday_dates=holiday_dates,
            hours=forecast_hours,
        )
    except Exception as exc:
        st.error(f"Forecast generation failed: {exc}")
        st.stop()

    st.success(
        f"Forecast generated successfully for {forecast_hours:,} hourly observations."
    )

    # Forecast summary
    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric(
            "Average Forecast",
            f"{forecast['Predicted_MW'].mean():,.2f} MW",
        )

    with a2:
        st.metric(
            "Peak Forecast",
            f"{forecast['Predicted_MW'].max():,.2f} MW",
        )

    with a3:
        st.metric(
            "Minimum Forecast",
            f"{forecast['Predicted_MW'].min():,.2f} MW",
        )

    # Forecast chart
    st.subheader("9. Forecast Visualization")

    forecast_chart = forecast.set_index("Datetime")
    st.line_chart(forecast_chart["Predicted_MW"])

    # Hourly table
    st.subheader("10. Hourly Forecast Table")

    display_forecast = forecast.copy()
    display_forecast["Predicted_MW"] = display_forecast["Predicted_MW"].round(2)

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True,
    )

    # Daily summary
    daily = forecast.copy()
    daily["Forecast_Day"] = (
        np.arange(len(daily)) // 24
    ) + 1

    daily_summary = (
        daily.groupby("Forecast_Day")
        .agg(
            Start_Datetime=("Datetime", "min"),
            End_Datetime=("Datetime", "max"),
            Average_Forecast_MW=("Predicted_MW", "mean"),
            Minimum_Forecast_MW=("Predicted_MW", "min"),
            Maximum_Forecast_MW=("Predicted_MW", "max"),
        )
        .reset_index()
    )

    for col in [
        "Average_Forecast_MW",
        "Minimum_Forecast_MW",
        "Maximum_Forecast_MW",
    ]:
        daily_summary[col] = daily_summary[col].round(2)

    st.subheader("11. Daily Forecast Summary")

    st.dataframe(
        daily_summary,
        use_container_width=True,
        hide_index=True,
    )

    # Downloads
    hourly_csv = forecast.to_csv(index=False).encode("utf-8")
    daily_csv = daily_summary.to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download Hourly Forecast CSV",
            data=hourly_csv,
            file_name="PJMW_Final_30_Day_Forecast.csv",
            mime="text/csv",
        )

    with d2:
        st.download_button(
            "Download Daily Summary CSV",
            data=daily_csv,
            file_name="PJMW_Final_30_Day_Daily_Summary.csv",
            mime="text/csv",
        )

else:
    st.info(
        "Click **Generate Forecast** to create the selected number of "
        "future hourly predictions."
    )

st.markdown("---")
st.caption(
    "P-702 / P679 Hourly Energy Consumption Forecast | "
    "Final notebook model: XGBoost | 30-day recursive hourly forecasting"
)
