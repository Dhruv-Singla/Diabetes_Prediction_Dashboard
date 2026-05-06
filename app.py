import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    layout="wide"
)

st.title("Diabetes Prediction Dashboard")

st.subheader(
    "GUI-Based Machine Learning Platform for Healthcare Classification"
)

df = pd.read_csv("diabetes.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset",
    "Visualizations",
    "Training",
    "Results",
    "Compare Models"
])

with tab1:

    st.header("Dataset Overview")

    st.dataframe(df.head())

    rows, cols = df.shape

    col1, col2 = st.columns(2)

    col1.metric("Rows", rows)
    col2.metric("Columns", cols)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Target Distribution")

    fig_target = px.histogram(
        df,
        x="Outcome",
        color="Outcome",
        title="Diabetes Distribution"
    )

    st.plotly_chart(fig_target)

with tab2:

    st.header("Data Visualizations")

    st.subheader("Correlation Heatmap")

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax_corr
    )

    st.pyplot(fig_corr)

    st.write(
        "Heatmap shows correlation between dataset features."
    )

    features = df.columns[:-1].tolist()

    st.subheader("Histogram")

    selected_hist = st.selectbox(
        "Select Feature for Histogram",
        features
    )

    fig_hist = px.histogram(
        df,
        x=selected_hist,
        color="Outcome",
        title=f"{selected_hist} Distribution"
    )

    st.plotly_chart(fig_hist)

    st.subheader("Box Plot")

    selected_box = st.selectbox(
        "Select Feature for Box Plot",
        features
    )

    fig_box, ax_box = plt.subplots()

    sns.boxplot(
        data=df,
        x="Outcome",
        y=selected_box,
        ax=ax_box
    )

    st.pyplot(fig_box)

    st.subheader("Scatter Plot")

    col3, col4 = st.columns(2)

    feat_x = col3.selectbox(
        "X Axis",
        features,
        index=1
    )

    feat_y = col4.selectbox(
        "Y Axis",
        features,
        index=5
    )

    fig_scatter = px.scatter(
        df,
        x=feat_x,
        y=feat_y,
        color=df["Outcome"].astype(str),
        title=f"{feat_x} vs {feat_y}"
    )

    st.plotly_chart(fig_scatter)

st.sidebar.title("ML Dashboard")

st.sidebar.header("Model Configuration")

algorithm = st.sidebar.selectbox(
    "Algorithm",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM",
        "Naive Bayes"
    ]
)

st.sidebar.subheader("Hyperparameters")

test_size = st.sidebar.slider(
    "Test Size (%)",
    10,
    40,
    20
)

random_state = st.sidebar.slider(
    "Random State",
    1,
    100,
    42
)

n_estimators = st.sidebar.slider(
    "Random Forest Estimators",
    10,
    300,
    100
)

max_depth = st.sidebar.slider(
    "Max Depth",
    1,
    20,
    5
)

neighbors = st.sidebar.slider(
    "KNN Neighbors",
    1,
    15,
    5
)

c_value = st.sidebar.slider(
    "C Value",
    0.1,
    10.0,
    1.0
)

kernel = st.sidebar.selectbox(
    "SVM Kernel",
    ["linear", "rbf", "poly"]
)

model_info = {
    "Logistic Regression":
    "Used for binary classification using probability.",

    "Decision Tree":
    "Tree-based classification model.",

    "Random Forest":
    "Ensemble model using multiple decision trees.",

    "KNN":
    "Classification using nearest neighbors.",

    "SVM":
    "Separates classes using hyperplanes.",

    "Naive Bayes":
    "Probability-based classification algorithm."
}

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=test_size / 100,
    random_state=random_state
)

if algorithm == "Logistic Regression":

    model = LogisticRegression(C=c_value)

elif algorithm == "Decision Tree":

    model = DecisionTreeClassifier(
        max_depth=max_depth
    )

elif algorithm == "Random Forest":

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

elif algorithm == "KNN":

    model = KNeighborsClassifier(
        n_neighbors=neighbors
    )

elif algorithm == "SVM":

    model = SVC(
        C=c_value,
        kernel=kernel,
        probability=True
    )

elif algorithm == "Naive Bayes":

    model = GaussianNB()

with tab3:

    st.header("Model Training")

    st.info(model_info[algorithm])

    if st.button("Train Model"):

        with st.spinner("Training model..."):

            model.fit(X_train, y_train)

            st.success(
                "Model trained successfully"
            )

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

with tab4:

    st.header("Performance Analysis")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    col6.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )

    col7.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )

    col8.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )

    st.subheader("Classification Report")

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            output_dict=True
        )
    ).transpose()

    st.dataframe(report_df)

    st.subheader("Confusion Matrix")

    st.write(
        "Confusion matrix shows correct and incorrect predictions."
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig_cm, ax_cm = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax_cm
    )

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")

    st.pyplot(fig_cm)

    st.subheader("ROC Curve")

    st.write(
        "ROC curve shows classification performance."
    )

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probs
        )

        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()

        ax_roc.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.2f}"
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            linestyle='--'
        )

        ax_roc.set_xlabel(
            "False Positive Rate"
        )

        ax_roc.set_ylabel(
            "True Positive Rate"
        )

        ax_roc.set_title(
            "ROC Curve"
        )

        ax_roc.legend()

        st.pyplot(fig_roc)

        st.subheader(
            "Prediction Probability Distribution"
        )

        prob_df = pd.DataFrame({
            "Probability": probs,
            "Actual": y_test.values
        })

        fig_prob = px.histogram(
            prob_df,
            x="Probability",
            color=prob_df["Actual"].astype(str),
            barmode="overlay",
            nbins=30,
            title="Prediction Probability Distribution"
        )

        st.plotly_chart(fig_prob)

    if hasattr(model, "feature_importances_"):

        st.subheader("Feature Importance")

        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        fig_imp, ax_imp = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            data=imp_df,
            x="Importance",
            y="Feature",
            ax=ax_imp
        )

        ax_imp.set_title(
            "Feature Importance"
        )

        st.pyplot(fig_imp)

        top_feat = imp_df.iloc[0]["Feature"]

        st.subheader(
            f"Violin Plot - {top_feat}"
        )

        fig_vio, ax_vio = plt.subplots()

        sns.violinplot(
            data=df,
            x="Outcome",
            y=top_feat,
            ax=ax_vio
        )

        st.pyplot(fig_vio)

with tab5:

    st.header("Model Comparison")

    all_models = {

        "Logistic Regression":
        LogisticRegression(),

        "Decision Tree":
        DecisionTreeClassifier(),

        "Random Forest":
        RandomForestClassifier(),

        "KNN":
        KNeighborsClassifier(),

        "SVM":
        SVC(),

        "Naive Bayes":
        GaussianNB()

    }

    results = {}

    for name, m in all_models.items():

        scores = cross_val_score(
            m,
            X_scaled,
            y,
            cv=5
        )

        results[name] = round(
            scores.mean(),
            4
        )

    result_df = pd.DataFrame({

        "Model": results.keys(),

        "Cross Validation Accuracy": results.values()

    }).sort_values(
        "Cross Validation Accuracy",
        ascending=False
    )

    st.dataframe(result_df)

    fig_compare, ax_compare = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=result_df,
        x="Model",
        y="Cross Validation Accuracy",
        ax=ax_compare
    )

    plt.xticks(rotation=20)

    ax_compare.set_title(
        "Model Comparison"
    )

    st.pyplot(fig_compare)

    best_model = result_df.iloc[0]

    st.success(
        f"Best Model: {best_model['Model']} | Accuracy: {best_model['Cross Validation Accuracy']}"
    )
