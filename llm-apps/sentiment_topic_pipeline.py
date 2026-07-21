import streamlit as st
import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
from io import BytesIO

# Function to analyze each review
def classify_sentiment_and_topic(api_key, df):
    openai.api_key = api_key

    sentiment_list = []
    topic_list = []

    for review in df["review"]:
        prompt = f"""
        Analyze the following customer review:
        "{review}"
        
        Return a JSON object with:
        1. sentiment: one of ["Positive", "Negative", "Neutral"]
        2. topics: a list of 1–3 key topics or keywords

        Example output:
        {{
            "sentiment": "Positive",
            "topics": ["delivery", "price"]
        }}
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            sentiment = parsed.get("sentiment", "Unknown")
            topics = ", ".join(parsed.get("topics", []))
        except Exception as e:
            sentiment = "Error"
            topics = "Parsing failed"

        sentiment_list.append(sentiment)
        topic_list.append(topics)

    df["sentiment"] = sentiment_list
    df["topics"] = topic_list
    return df


# Streamlit app
def main():
    st.set_page_config(page_title="Sentiment Analysis AI Assistant", layout="wide")
    st.title("🧠 Sentiment Analysis AI Assistant")
    st.write("Upload customer review data and get AI-generated sentiment insights, topic extraction, filtering, and visual dashboards.")

    # API Key Input
    api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

    # File upload
    uploaded_file = st.file_uploader("📄 Upload CSV or Excel file (must include 'review' column, optional 'rating')", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return

        df.columns = df.columns.str.strip().str.lower()

        if "review" not in df.columns:
            st.error("The file must contain a column named 'review'.")
            return

        if st.button("🔍 Analyze Reviews"):
            if not api_key:
                st.error("Please enter your OpenAI API Key.")
                return

            with st.spinner("Analyzing reviews using GPT..."):
                df = classify_sentiment_and_topic(api_key, df)

            st.success("✅ Analysis complete!")

            # Filters
            st.subheader("🔎 Filter Reviews")
            sentiments = df["sentiment"].unique().tolist()
            sentiment_filter = st.multiselect("Select Sentiments", options=sentiments, default=sentiments)

            all_topics = df["topics"].dropna().str.split(", ").explode().unique().tolist()
            topic_filter = st.multiselect("Select Topics", options=sorted(all_topics))

            filtered_df = df[df["sentiment"].isin(sentiment_filter)]
            if topic_filter:
                filtered_df = filtered_df[filtered_df["topics"].str.contains('|'.join(topic_filter), na=False)]

            st.subheader("🗃️ Filtered Review Results")
            st.dataframe(filtered_df[["review", "sentiment", "topics"]])

            # Download Button
            csv_download = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Results as CSV", csv_download, file_name="sentiment_results.csv", mime="text/csv")

            # 📊 Dashboard
            st.subheader("📊 Dashboard")

            col1, col2 = st.columns(2)

            with col1:
                if "rating" in df.columns:
                    st.markdown("#### Rating Frequency")
                    fig, ax = plt.subplots()
                    sns.histplot(df["rating"], bins=10, kde=False, ax=ax)
                    ax.set_xlabel("Rating")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                else:
                    st.info("No 'rating' column found. Skipping rating histogram.")

            with col2:
                st.markdown("#### Sentiment Distribution")
                sentiment_counts = df["sentiment"].value_counts()
                st.bar_chart(sentiment_counts)

            # Optional: Word Cloud of Topics
            st.subheader("☁️ Topic Word Cloud")
            all_topics_text = " ".join(df["topics"].dropna().astype(str).tolist())
            if all_topics_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_topics_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No topics available to generate word cloud.")

# Run the app
if __name__ == "__main__":
    main()
