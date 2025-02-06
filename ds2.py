import pandas as pd
import streamlit as st

df = pd.read_csv("datasets/biscayne_bay_water_quality.csv")

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.describe()) # Descriptive Statistics

filtered_df = df[['TWC', 'Speed', 'Salinity',
       'Temp C', 'pH', 'ODO']]
print(filtered_df.corr())

st.title("Water Quality Data Analysis")
st.dataframe(df)

import plotly.express as px


fig1 = px.box(df,y="pH")
st.plotly_chart(fig1)

fig2 = px.box(df,y="Salinity")
st.plotly_chart(fig2)

Q1_salinity = df["Salinity"].quantile(0.25)
st.write(f"25% quantile: {Q1_salinity}")

Q3_salinity = df["Salinity"].quantile(0.75)
st.write(f"75% quantile: {Q3_salinity}")

# The interquartile range
IQR_salinity = Q3_salinity - Q1_salinity
st.write(f"IQR: {IQR_salinity:.2f}")

# Let's calculate the lower and upper bounds:
lower_bound_salinity = Q1_salinity - 1.5*IQR_salinity
upper_bound_salinity = Q3_salinity + 1.5*IQR_salinity

st.write(f"Lower bound: {lower_bound_salinity:.2f}")
st.write(f"Upper bound: {upper_bound_salinity:.2f}")

# So, let's identify the potential outliers
salinity_outliers = df[
       (df["Salinity"]<lower_bound_salinity)
       |
       (df["Salinity"]>upper_bound_salinity)
]

st.dataframe(salinity_outliers)

df_cleaned = df[
       (df["Salinity"] > lower_bound_salinity)
       &
       (df["Salinity"] < upper_bound_salinity)
]

print(df_cleaned.shape)

fig3 = px.box(df_cleaned,x="Salinity")
st.plotly_chart(fig3)

st.dataframe(df_cleaned.describe())

fig4 = px.line(df_cleaned,x="Time",y="Temp C",title="Line Chart for Temperature")
st.plotly_chart(fig4)

fig5 = px.line(df_cleaned,x="Time",y="ODO",title="Line Chart for ODO")
st.plotly_chart(fig5)

fig6 = px.scatter(df_cleaned,x="TWC",y="Speed",color="Temp C")
st.plotly_chart(fig6)

# This is a dataset that contains DEPTH, LAT, LONG, what cool visualizations
# can I do with this? Math students?

fig7 = px.scatter_3d(df_cleaned,
                     x = "Longitude",
                     y = "Latitude",
                     z = "TWC",
                     color = "Temp C")
fig7.update_scenes(zaxis_autorange="reversed")
st.plotly_chart(fig7)

fig8 = px.scatter_mapbox(df_cleaned,
                         lat="Latitude",
                         lon="Longitude",
                         mapbox_style="open-street-map",
                         zoom=17,
                         hover_data=df_cleaned,
                         color="Temp C")
st.plotly_chart(fig8)