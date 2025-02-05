import streamlit as st
import datetime


st.title("Introduction to Data Science")
st.header("Gregory Murad Reis")
st.subheader("Example 1")

first_name = st.text_input("Enter your first name")
last_name = st.text_input("Enter your last name")
yob = st.number_input("Enter your year of birth")
start_year_at_FIU = st.number_input("Enter the year you started at"
"FIU")

current_year = datetime.date.today().year
age = current_year - yob
years_at_fiu = current_year - start_year_at_FIU

if first_name and last_name and yob and start_year_at_FIU:
    st.success(f"{first_name} {last_name} is {age} years old,"
    f"and has been working at FIU for {years_at_fiu} years.")
    st.write("This is a test")
    st.warning("The model is imperfect")
    st.info("You need to submit your dataset as csv format")
    st.error("No models could be generated")