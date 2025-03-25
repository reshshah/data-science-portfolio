import streamlit as st
import openai

# Function to generate sentiment analysis write-up
def generate_sentiment_analysis_writeup(api_key, text_data, task_instruction):
    openai.api_key = api_key

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data scientist. Your job is to analyze sentiment from customer reviews and generate insights."},
            {"role": "user", "content": f"Here is the text data for analysis:\n{text_data}\n\nTask: {task_instruction}"}
        ]
    )
    return response.choices[0].message.content.strip()

# Streamlit app
def main():
    st.title("Sentiment Analysis AI Assistant")
    st.write("Enter customer review data and specify what insights you need.")

    # API Key Input
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # User Inputs
    text_data = st.text_area("Paste customer reviews here")
    task_instruction = st.text_input("Enter task instruction (e.g., 'Perform sentiment analysis and summarize key themes')", 
                                     value="Perform sentiment analysis and summarize key themes.")

    # Button to trigger analysis
    if st.button("Generate Analysis"):
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
        elif not text_data.strip():
            st.error("Please enter text data for analysis.")
        else:
            with st.spinner("Generating sentiment analysis write-up..."):
                sentiment_writeup = generate_sentiment_analysis_writeup(api_key, text_data, task_instruction)
                st.subheader("Sentiment Analysis Write-up:")
                st.write(sentiment_writeup)

# Run the app
if __name__ == "__main__":
    main()
