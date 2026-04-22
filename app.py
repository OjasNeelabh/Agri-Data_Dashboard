import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Agri-Data Econometrics", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# --- 2. DATA CACHING & PROCESSING ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ankitaS11/Crop-Yield-Prediction-in-India-using-ML/main/crop_production.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df['Season'] = df['Season'].str.strip()
    df['Yield'] = df['Production'] / df['Area']
    df = df[np.isfinite(df['Yield'])]
    return df

with st.spinner('Fetching National Agricultural Database...'):
    raw_df = load_data()

# --- 3. SIDEBAR: CUSTOMIZATION & FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/305/305100.png", width=100) # Optional icon
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Use these filters to customize the analysis.")

# Filter: State
all_states = sorted(raw_df['State_Name'].unique())
selected_states = st.sidebar.multiselect("Select State(s):", all_states, default=["Maharashtra", "Punjab", "Uttar Pradesh", "Madhya Pradesh"])

# Filter: Crop
all_crops = sorted(raw_df['Crop'].unique())
selected_crops = st.sidebar.multiselect("Select Crop(s):", all_crops, default=["Wheat", "Rice", "Maize", "Cotton(lint)"])

# Apply Filters
if not selected_states or not selected_crops:
    st.warning("Please select at least one State and one Crop from the sidebar to continue.")
    st.stop()

filtered_df = raw_df[(raw_df['State_Name'].isin(selected_states)) & (raw_df['Crop'].isin(selected_crops))]

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate Analysis", ["1. Executive Summary", "2. Statistical Inference (T-Test)", "3. Predictive Econometrics"])

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "1. Executive Summary":
    st.title("🌾 Agricultural Econometrics: Executive Summary")
    st.markdown("Welcome to the interactive analysis tool for Indian agricultural production. This dashboard applies statistical structures to understand crop yields across different regions and seasons.")
    
    # Top-level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records Analyzed", f"{len(filtered_df):,}")
    col2.metric("Total Area Harvested (Hectares)", f"{filtered_df['Area'].sum():,.0f}")
    col3.metric("Total Production (Units)", f"{filtered_df['Production'].sum():,.0f}")
    
    st.markdown("### Production Distribution by State")
    # Interactive Plotly Chart
    fig = px.pie(filtered_df, values='Production', names='State_Name', hole=0.4, 
                 title="Share of Total Production by State (Filtered)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Guide: How to read this data"):
        st.write("""
        * **Area:** The total land size (in hectares) dedicated to farming.
        * **Production:** The total output generated.
        * **Yield:** Our primary metric of efficiency (Production / Area). A higher yield means the land is being used more efficiently.
        """)

    st.markdown("### Raw Data Viewer")
    st.dataframe(filtered_df[['State_Name', 'Crop_Year', 'Season', 'Crop', 'Area', 'Production', 'Yield']].head(100), use_container_width=True)

# --- PAGE 2: HYPOTHESIS TESTING ---
elif page == "2. Statistical Inference (T-Test)":
    st.title("⚖️ Statistical Inference: Seasonal Yields")
    st.markdown("Does the farming season statistically impact the efficiency (yield) of our crops?")
    
    # Allow user to choose which seasons to test against each other
    available_seasons = list(filtered_df['Season'].unique())
    if len(available_seasons) < 2:
        st.error("Not enough seasonal data for the selected crops/states. Please select more options in the sidebar.")
        st.stop()
        
    st.markdown("### Configure Your Test")
    colA, colB = st.columns(2)
    season_1 = colA.selectbox("Select Season A:", available_seasons, index=0)
    season_2 = colB.selectbox("Select Season B:", available_seasons, index=1 if len(available_seasons)>1 else 0)
    
    data_1 = filtered_df[filtered_df['Season'] == season_1]['Yield']
    data_2 = filtered_df[filtered_df['Season'] == season_2]['Yield']
    
    st.markdown("---")
    # Run Statistics
    stat_var, p_var = stats.levene(data_1, data_2)
    t_stat, p_value = stats.ttest_ind(data_1, data_2, equal_var=False) # Welch's T-test
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Yield (Season A)", f"{data_1.mean():.2f}")
    col2.metric("Mean Yield (Season B)", f"{data_2.mean():.2f}")
    col3.metric("P-Value", f"{p_value:.4e}")
    
    # Explainer Box for viewers
    if p_value < 0.05:
        st.success(f"**Conclusion:** Reject the Null Hypothesis. There is a statistically significant difference between {season_1} and {season_2} yields.")
    else:
        st.warning(f"**Conclusion:** Fail to reject the Null Hypothesis. We cannot definitively say there is a difference between these seasons based on this data.")
        
    with st.expander("🧠 What does this mean? (Statistical Explainer)"):
        st.write("""
        * **Null Hypothesis (H0):** Assumes there is zero difference in average yield between the two chosen seasons.
        * **P-Value:** The probability of seeing this data if the Null Hypothesis were true. If it is less than 0.05 (5%), we say the difference is "statistically significant" and not just due to random chance.
        * **Welch's T-Test:** We use this specific version of the t-test because Levene's test proved our two seasons have unequal variance (spread) in their data.
        """)

    # Interactive Boxplot
    st.markdown("### Yield Distribution")
    test_df = filtered_df[filtered_df['Season'].isin([season_1, season_2])]
    fig2 = px.box(test_df, x="Season", y="Yield", color="Season", points="all", 
                  title=f"Spread of Yield Data: {season_1} vs {season_2}")
    st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 3: PREDICTIVE ECONOMETRICS ---
elif page == "3. Predictive Econometrics":
    st.title("📈 Predictive Econometrics: Regression Models")
    st.markdown("Using machine learning and generalized linear models to forecast agricultural output.")
    
    # Model Configuration
    st.markdown("### Model Configuration")
    use_dummy_vars = st.checkbox("Include 'Season' as a dummy variable to improve accuracy?", value=True)
    
    # Prepare Data
    if use_dummy_vars:
        model_data = pd.get_dummies(filtered_df[['Area', 'Season', 'Production']], columns=['Season'], drop_first=True)
    else:
        model_data = filtered_df[['Area', 'Production']]
        
    X = model_data.drop('Production', axis=1)
    y = model_data['Production']
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Models
    lr = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=10.0).fit(X_train, y_train)
    
    lr_r2 = r2_score(y_test, lr.predict(X_test))
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))
    
    # Metrics display
    col1, col2 = st.columns(2)
    col1.metric("Ordinary Least Squares (OLS) R²", f"{lr_r2:.4f}")
    col2.metric("Ridge Regression (L2) R²", f"{ridge_r2:.4f}")
    
    with st.expander("🔍 What is R-Squared and Ridge Regression?"):
        st.write("""
        * **R-Squared (R²):** Represents the percentage of variance in crop production that our model can explain. An R² of 0.85 means our model explains 85% of the crop output based purely on the land area and season!
        * **Ridge Regression:** When we add many variables (like multiple seasons), standard linear regression can become unstable. Ridge adds a "penalty" to keep the math balanced, making it more reliable for real-world predictions.
        """)

    # Interactive Simulator
    st.markdown("### 🚜 Interactive Output Simulator")
    st.markdown("Adjust the farm size below to see how much production the model predicts.")
    
    sim_col1, sim_col2 = st.columns([2, 1])
    with sim_col1:
        input_area = st.slider("Total Farm Area (Hectares):", min_value=10, max_value=200000, value=50000, step=1000)
    
    # Create input dataframe matching model features
    input_df = pd.DataFrame({'Area': [input_area]})
    if use_dummy_vars:
        for col in X.columns:
            if col != 'Area':
                input_df[col] = 0 # Default to base season
                
    predicted_val = lr.predict(input_df)[0]
    
    with sim_col2:
        st.info("### Estimated Output")
        st.title(f"{predicted_val:,.0f}")
        st.markdown("*Units of Production*")

    # Diagnostic Plot
    st.markdown("### Model Diagnostics: Residual Plot")
    st.markdown("A good statistical model should have randomly scattered errors (residuals). If there is a pattern, we are missing a variable.")
    
    residuals = y_test - lr.predict(X_test)
    diag_df = pd.DataFrame({'Predicted': lr.predict(X_test), 'Residuals': residuals})
    
    fig3 = px.scatter(diag_df, x="Predicted", y="Residuals", opacity=0.5,
                      title="Residuals vs. Fitted Values (Checking Model Validity)")
    fig3.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)