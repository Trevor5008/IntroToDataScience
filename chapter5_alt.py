import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import plotly.express as px

st.title("Supervised Learning: Classification with Iris Dataset")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Introduction
st.subheader("1. Iris Dataset")
st.write("The Iris dataset consists of 150 samples with 4 features each, labeled into 3 classes (species).")
df = pd.DataFrame(X, columns=feature_names)
df_labels = pd.DataFrame({'species': [target_names[label] for label in y]})
cols = st.columns([0.7, 0.3])

with cols[0]:
    st.dataframe(df.head(10))
    st.caption("First 10 samples of features (in cm)")
with cols[1]:
    st.dataframe(pd.DataFrame(df).head(10))
    st.caption("Labels (species)")

# Split dataset interactively
st.subheader("2. Data Splitting")
test_size = st.slider("Select Testing Data Size (%)", min_value=5, max_value=50, step=5, value=20)
random_state = st.number_input("Random State (for reproducibility)", min_value=0, max_value=100, value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

st.write(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# Train k-NN classifier interactively
st.subheader("3. k-NN Classifier Training")
n_neighbors = st.slider("Select number of neighbors (k)", min_value=1, max_value=15, value=3, step=1)

if st.button("Train Model"):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Accuracy of model on test set: {accuracy:.2f}")

    # Show predictions
    predictions_df = pd.DataFrame({
        "Predicted": [target_names[i] for i in y_pred],
        "Actual": [target_names[i] for i in y_test]
    })
    st.subheader("Sample Predictions vs Actual")
    st.dataframe(predictions_df.head(10))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted Label',
                      yaxis_title='True Label')
    st.plotly_chart(fig)

    # Scatter plot visualization to show clusters
    scatter_df = pd.DataFrame(X_test, columns=feature_names)
    scatter_df['Predicted'] = [target_names[label] for label in y_pred]
    scatter_df['Actual'] = [target_names[label] for label in y_test]

    fig_scatter = px.scatter(
        scatter_df,
        x=feature_names[0],
        y=feature_names[2],
        color='Actual',
        symbol='Predicted',
        title='Scatter Plot of Test Data Predictions',
        labels={feature_names[0]: feature_names[0], feature_names[2]: feature_names[2]},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
