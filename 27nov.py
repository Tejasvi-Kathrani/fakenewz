import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VeriFact Advanced Analysis",
    layout="wide"
)

st.title(" VeriFact: Advanced News Analysis")
st.caption("Temporal Trends | Research Report | External Model Evaluation")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("verifact_model.pkl")
    vectorizer = joblib.load("verifact_tfidf.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------- DATASET UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload test.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your combined dataset to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

required_columns = ["title", "text", "subject", "date", "real"]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset must contain: title, text, subject, date, real")
    st.stop()

# data cleaning
# ---------------- DATA CLEANING ----------------

df = df.copy()

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)
df["subject"] = df["subject"].fillna("Unknown").astype(str)

# Clean labels
df["real"] = pd.to_numeric(df["real"], errors="coerce")

# Clean dates
df["date"] = pd.to_datetime(
    df["date"].astype(str).str.strip(),
    errors="coerce",
    format="mixed"
)

# Remove invalid records
df = df.dropna(subset=["date", "real"])

df = df[df["real"].isin([0, 1])]

df = df[
    (df["date"].dt.year >= 2000) &
    (df["date"].dt.year <= 2026)
].copy()

df = df[df["title"].str.strip() != ""]

# Convert labels
df["real"] = df["real"].astype(int)

# Create analysis columns
df["News Type"] = df["real"].map({
    0: "Fake",
    1: "Real"
})

df["Year"] = df["date"].dt.year.astype(int)
df["Month"] = df["date"].dt.to_period("M").astype(str)

if df.empty:
    st.error("No valid records found after cleaning.")
    st.stop()


st.success(f"Dataset loaded successfully: {len(df):,} valid records")

# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Select Task",
    [
        "Temporal Trend Analysis",
        "Final Findings Report",
        "External Dataset Evaluation"
    ]
)

# TASK 1: TEMPORAL TREND ANALYSIS


if page == "Temporal Trend Analysis":

    st.title("VeriFact - Temporal Trend Analysis")

    # ---------------- DATA CLEANING ----------------

    df = df.copy()

    required_columns = ["title", "text", "subject", "date", "real"]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["subject"] = (
        df["subject"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df["real"] = pd.to_numeric(df["real"], errors="coerce")

    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        errors="coerce",
        format="mixed"
    )

    # Remove invalid records
    df = df.dropna(subset=["date", "real"])

    df = df[df["real"].isin([0, 1])]

    df = df[
        (df["date"].dt.year >= 2000) &
        (df["date"].dt.year <= 2026)
    ].copy()

    df = df[df["title"].str.strip() != ""]

    df["real"] = df["real"].astype(int)

    # Create analysis columns
    df["Year"] = df["date"].dt.year.astype(int)
    df["Month"] = df["date"].dt.to_period("M").astype(str)

    df["News Type"] = df["real"].map({
        0: "Fake",
        1: "Real"
    })

    if df.empty:
        st.error("No valid records found after cleaning.")
        st.stop()

    # ---------------- SUMMARY ----------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Total News Records", len(df))
    col2.metric("Fake News", int((df["real"] == 0).sum()))
    col3.metric("Real News", int((df["real"] == 1).sum()))

    st.divider()

    # ---------------- 1. YEAR-WISE DISTRIBUTION ----------------

    st.subheader("1. Year-wise Real vs Fake News")

    yearly_data = (
        df.groupby(["Year", "News Type"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        yearly_data,
        x="Year",
        y="Count",
        color="News Type",
        barmode="group",
        title="Year-wise Real vs Fake News Distribution"
    )

    fig.update_xaxes(type="category", title="Year")
    fig.update_yaxes(title="Number of News Articles")

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False}
    )

    st.divider()

    # ---------------- 2. MONTHLY TREND ----------------

    st.subheader("2. Monthly Real vs Fake News Trend")

    monthly_data = (
        df.groupby(["Month", "News Type"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.line(
        monthly_data,
        x="Month",
        y="Count",
        color="News Type",
        title="Monthly Real vs Fake News Trend"
    )

    # Remove points from monthly lines
    fig.update_traces(mode="lines")

    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Number of News Articles")

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False}
    )

    st.divider()

    # ---------------- 3. FAKE NEWS PERCENTAGE ----------------

    st.subheader("3. Year-wise Fake News Percentage")

    fake_rate = (
        df.groupby("Year")["real"]
        .apply(lambda x: (x == 0).mean() * 100)
        .reset_index(name="Fake News Percentage")
    )

    fig = px.bar(
        fake_rate,
        x="Year",
        y="Fake News Percentage",
        title="Fake News Percentage by Year",
        text_auto=".2f"
    )

    fig.update_xaxes(type="category", title="Year")
    fig.update_yaxes(
        title="Fake News Percentage (%)",
        range=[0, 100]
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False}
    )

    st.divider()

    # ---------------- 4. SUBJECT ANALYSIS ----------------

    st.subheader("4. Fake News Distribution by Subject")

    fake_subjects = (
        df[df["real"] == 0]
        .groupby("subject")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    if fake_subjects.empty:

        st.warning("No fake news records found for subject analysis.")

    else:

        fig = px.bar(
            fake_subjects,
            x="subject",
            y="Count",
            title="Fake News Distribution by Subject",
            text_auto=True
        )

        fig.update_xaxes(title="Subject")
        fig.update_yaxes(title="Number of Fake News Articles")

        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False}
        )

    st.divider()

    st.success("Task 1: Temporal Trend Analysis completed.")
# TASK 2: FINAL FINDINGS REPORT

elif page == "Final Findings Report":

    st.header("VeriFact - Final Research Findings")

    total = len(df)
    fake = int((df["real"] == 0).sum())
    real = int((df["real"] == 1).sum())

    fake_percentage = fake / total * 100
    real_percentage = real / total * 100

    first_year = int(df["Year"].min())
    last_year = int(df["Year"].max())

    yearly_fake = (
        df[df["real"] == 0]
        .groupby("Year")
        .size()
    )

    highest_fake_year = (
        int(yearly_fake.idxmax()) if not yearly_fake.empty else "N/A"
    )

    popular_fake_subject = (
        df[df["real"] == 0]["subject"].mode().iloc[0]
        if fake > 0 else "N/A"
    )

    st.subheader("1. Project Overview")

    st.write("""
    VeriFact is a machine learning-based fake news detection project.
    It uses Natural Language Processing and machine learning techniques
    to distinguish between real and fake news.
    """)

    st.subheader("2. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Articles", f"{total:,}")
    col2.metric("Fake Articles", f"{fake:,}")
    col3.metric("Real Articles", f"{real:,}")

    st.write(f"**Dataset period:** {first_year} - {last_year}")

    st.subheader("3. Dataset Distribution")

    distribution = pd.DataFrame({
        "News Type": ["Fake", "Real"],
        "Count": [fake, real],
        "Percentage": [fake_percentage, real_percentage]
    })

    st.dataframe(distribution, width="stretch")

    fig = px.pie(
        distribution,
        names="News Type",
        values="Count",
        title="Real vs Fake News Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("4. Major Findings")

    st.write(f"""
    - Fake news represents {fake_percentage:.2f}% of the dataset.
    - Real news represents {real_percentage:.2f}% of the dataset.
    - The year with the highest number of fake articles is {highest_fake_year}.
    - The most frequent fake-news subject is {popular_fake_subject}.
    - The dataset covers the period from {first_year} to {last_year}.
    """)

    st.subheader("5. Recommendations")

    st.write("""
    1. Improve public awareness about misinformation.
    2. Verify news through reliable sources before sharing.
    3. Continuously update machine learning models with new data.
    4. Monitor changes in fake-news topics and language patterns.
    5. Evaluate models using independent datasets.
    """)

    st.subheader("6. Limitations")

    st.write("""
    - Results depend on the quality and representativeness of the dataset.
    - Historical trends do not automatically indicate real-world prevalence.
    - Machine learning predictions may contain false positives and false negatives.
    - Changes in news language can affect model performance.
    """)

    st.subheader("7. Conclusion")

    st.success("""
    VeriFact demonstrates how machine learning and text analysis can
    support fake news detection. Temporal analysis helps identify
    historical patterns, while external evaluation helps assess
    model generalizability.
    """)

# TASK 3: EXTERNAL DATASET EVALUATION

elif page == "External Dataset Evaluation":

    st.header(" External Dataset Model Evaluation")

    st.write("""
    This task evaluates the trained VeriFact model on an independent
    dataset to examine its ability to classify news from different sources.
    """)

    st.warning(
        "Ensure this dataset was not used to train or tune the existing model."
    )

    if st.button("Run External Dataset Evaluation"):

        with st.spinner("Testing model on external dataset..."):

            try:
                X = vectorizer.transform(df["title"])
                y_true = df["real"]

                y_pred = model.predict(X)

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(
                    y_true, y_pred, zero_division=0
                )
                recall = recall_score(
                    y_true, y_pred, zero_division=0
                )
                f1 = f1_score(
                    y_true, y_pred, zero_division=0
                )

                st.success("External evaluation completed!")

                st.subheader("1. Evaluation Metrics")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                col2.metric("Precision", f"{precision * 100:.2f}%")
                col3.metric("Recall", f"{recall * 100:.2f}%")
                col4.metric("F1 Score", f"{f1 * 100:.2f}%")

                st.subheader("2. Confusion Matrix")

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Fake", "Actual Real"],
                    columns=["Predicted Fake", "Predicted Real"]
                )

                fig = px.imshow(
                    cm_df,
                    text_auto=True,
                    title="External Dataset Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual")
                )

                st.plotly_chart(fig, width="stretch")

                st.subheader("3. Classification Report")

                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=["Fake News", "Real News"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose(),
                    width="stretch"
                )

                st.subheader("4. Prediction Analysis")

                results = df[["title", "date", "real"]].copy()

                results["Actual"] = results["real"].map({
                    0: "Fake",
                    1: "Real"
                })

                results["Predicted"] = pd.Series(y_pred).map({
                    0: "Fake",
                    1: "Real"
                }).values

                results["Correct"] = (
                    results["Actual"] == results["Predicted"]
                )

                st.dataframe(
                    results.head(100),
                    width="stretch"
                )

                correct = int(results["Correct"].sum())
                incorrect = len(results) - correct

                st.write(f"Correct predictions: {correct:,}")
                st.write(f"Incorrect predictions: {incorrect:,}")

                st.subheader("5. Evaluation Conclusion")

                st.write(f"""
                The VeriFact model achieved an accuracy of
                {accuracy * 100:.2f}% on the external dataset.

                Precision was {precision * 100:.2f}%,
                recall was {recall * 100:.2f}%,
                and the F1 score was {f1 * 100:.2f}%.
                """)

                if f1 >= 0.8:
                    st.success(
                        "The model demonstrated strong performance "
                        "on this external dataset."
                    )
                elif f1 >= 0.6:
                    st.warning(
                        "The model demonstrated moderate performance. "
                        "Further improvement may be required."
                    )
                else:
                    st.error(
                        "The model struggled with this external dataset. "
                        "Further investigation and retraining may be needed."
                    )

                st.info(
                    "These results measure performance on this particular "
                    "dataset and do not establish generalizability to every "
                    "news source or time period."
                )

            except Exception as e:
                st.error(f"Evaluation failed: {e}")