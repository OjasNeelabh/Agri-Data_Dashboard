import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import os

# --- Page Config ---
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# --- AUTO-LOAD LOGIC ---
# This looks for the file in the same directory as the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'Air_Pollution.csv')

@st.cache_data
def load_data():
    # If the file exists, load it. If not, show a clean error.
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        return df
    return None

df_raw = load_data()

if df_raw is None:
    st.error(f"❌ **Data File Missing!** Make sure 'Air_Pollution.csv' is uploaded to your GitHub repository in the same folder as this script.")
    st.stop()

# --- PREPROCESSING (Same as your IPYNB) ---
# Dropping NAs and fixing column names for Statsmodels compatibility
df = df_raw.dropna(subset=['pollutant_avg', 'latitude', 'longitude']).copy()
df = pd.get_dummies(df, columns=['pollutant_id'], drop_first=True)
df.columns = [c.replace('.', '_') for c in df.columns] 

# Feature Engineering
scaler = StandardScaler()
df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude', 'longitude']])
df['latitude2'] = df['latitude']**2
df['longitude2'] = df['longitude']**2
df['lat_long_interaction'] = df['latitude'] * df['longitude']

# --- DASHBOARD UI ---
st.title("🌍 Air Pollution Data Dashboard")

tabs = st.tabs(["📊 Analysis Results", "🤖 Machine Learning"])

with tabs[0]:
    st.subheader("Statistical Summary (OLS)")
    # Replicating your notebook's exact regression formula
    formula = 'pollutant_avg ~ latitude + longitude + latitude2 + longitude2 + lat_long_interaction + pollutant_id_NH3 + pollutant_id_NO2 + pollutant_id_OZONE + pollutant_id_PM10 + pollutant_id_PM2_5 + pollutant_id_SO2'
    model_ols = smf.ols(formula, data=df).fit()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Model Parameters:**")
        st.write(f"- R-Squared: {model_ols.rsquared:.4f}")
        st.write(f"- Adj. R-Squared: {model_ols.rsquared_adj:.4f}")
    with col2:
        st.text(str(model_ols.summary().tables[1])) # Show the coefficients table

with tabs[1]:
    st.subheader("Model Performance (10-Fold CV)")
    X = df[['latitude', 'longitude', 'latitude2', 'longitude2', 'lat_long_interaction', 
            'pollutant_id_NH3', 'pollutant_id_NO2', 'pollutant_id_OZONE', 
            'pollutant_id_PM10', 'pollutant_id_PM2_5', 'pollutant_id_SO2']]
    y = df['pollutant_avg']
    
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(np.mean(-scores))
    
    st.metric("Model RMSE", f"{rmse:.2f}")
    st.info("The RMSE tells your professor how many units off our predictions are on average.")