import streamlit as st
import pandas as pd


df = pd.read_csv("datasets/oct25-2024.csv")

st.title("Data Visualization")
st.subheader("Water Quality Data in Biscayne Bay")

st.dataframe(df)