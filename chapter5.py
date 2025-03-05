import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

st.title("Machine Learning - Example 1")

iris = load_iris()

st.subheader("1. The iris dataset")
cols = st.columns([4,1],vertical_alignment='top',border=True)
with cols[0]:
    st.dataframe(iris.data)
    st.caption("Features: columns are sepal length, sepal width, petal length, petal width")
with cols[1]:
    st.dataframe(iris.target)
    st.caption("labels: 0, 1, 2 correspond to the three iris species")

st.subheader("2. Splitting into training and testing sets")
st.write("Usually: 80% train, 20% test")
X = iris.data
y = iris.target

with st.form("Building a k-NN Classifier"):
    test_size=st.slider("Testing size",min_value=0.05,max_value=0.4,step=0.05,value=0.2)
    st.success(f"Selected: {(1-test_size)*100}% train, {test_size*100}% test")
    random_state=st.slider("Random state",min_value=0,max_value=100,step=1,value=42)
    st.success(f"Selected random state: {random_state}")
    with st.popover("Random state"):
        st.text("Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_neighbors=st.number_input("Neighbors", min_value=1, max_value=10, step=1, value=3)
    submit=st.form_submit_button("submit")

if submit:
    st.subheader("3. Initialize a k-NN classifier and train it")

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)  # learn from the training data
    st.write(model)

    st.subheader("4. Make predictions on the test set")
    y_pred = model.predict(X_test)
    st.write(y_pred)

    # 5. Evaluate the accuracy of the model on the test set
    accuracy = (y_pred == y_test).mean()
    print("Test accuracy:", accuracy)
