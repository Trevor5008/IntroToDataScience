import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide",
                   page_title="Dashboard")

st.title("Biscayne Bay Water Quality EDA")

st.header("1. Data Loading")
df = pd.read_csv("datasets/biscayne_bay_water_quality2.csv")

with st.expander("View Raw Data"):
    st.dataframe(df)

st.header("2. Basic Information")
col1, col2 = st.columns(2)
with col1:
    st.write("**Features**")
    st.write(df.columns)

    st.write("Sum of All Null Values")
    st.write(df.isnull().sum())

    st.write("Sum of All Duplicatd Rows")
    st.write(df.duplicated().sum())

with col2:
    st.write("**Descriptive Statistics**")
    st.dataframe(df.describe())

numerical_cols = ["Total Water Column (m)",
                  "Salinity (ppt)",
                  "Temperature (c)",
                  "pH",
                  "ODO mg/L"]

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in Total Water Column, Temperature, Salinity, pH, and ODO
outlier_dict = {}
for col in numerical_cols:
    outliers, lower, upper = detect_outliers(df, col)
    outlier_dict[col] = {
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Number of Outliers": len(outliers),
        "Outlier Percentage": len(outliers) / len(df) * 100,
    }

# Convert outlier data to a DataFrame and display
st.header("3. Outlier Summary")
outlier_df = pd.DataFrame.from_dict(outlier_dict, orient="index")
st.dataframe(outlier_df)

st.header("4. Box Plots")
boxplots = st.checkbox("Show Box Plots (Raw Data)")
if boxplots:
    for col in numerical_cols:
        fig = px.box(df,
                     x=col,
                     title=f"Box plot of {col}")
        st.plotly_chart(fig)

# Remove Outliers
df_no_outliers = df.copy()
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]

st.write(f"The data size after removing outliers: {df_no_outliers.shape}.")

boxplots = st.checkbox("Show Box Plots (After Outliers Treatment)")
if boxplots:
    for col in numerical_cols:
        fig = px.box(df_no_outliers,
                     x=col,
                     title=f"Box plot of {col} (After outliers treatment)")
        st.plotly_chart(fig)

# Skewness
skewness_values = {}
for col in numerical_cols:
    gp = 3 * (df_no_outliers[col].mean() - df_no_outliers[col].median()) / df_no_outliers[col].std()
    skewness_values[col]=gp
    print(f"Pearson median skew ({col}):", gp)

skew_df = pd.DataFrame({"Skewness":skewness_values})
st.dataframe(skew_df)