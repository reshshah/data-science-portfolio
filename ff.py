import streamlit as st

# Define the questions and answers
questions = [
    {
        "question": "Name a Bollywood movie every 80s kid remembers.",
        "options": ["Sholay", "DDLJ", "Hum Aapke Hain Koun", "Jo Jeeta Wohi Sikandar", "Lagaan"],
        "points": [40, 35, 25, 15, 10]
    },
    {
        "question": "What’s a must-have at an Indian wedding?",
        "options": ["Food", "Dhol", "Sangeet", "Baraat", "Saree"],
        "points": [40, 30, 20, 15, 10]
    },
    # Add more questions here (total of 20)
]

st.title("🎉 Family Feud: Desi Edition 🎉")

# Initialize session state
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "score" not in st.session_state:
    st.session_state.score = 0

# Get current question
q_index = st.session_state.question_index
if q_index < len(questions):
    question = questions[q_index]
    st.header(f"Q{q_index + 1}: {question['question']}")

    selected = st.radio("Select your answer:", question["options"], key=f"q{q_index}")

    if st.button("Lock in Answer"):
        points = question["points"][question["options"].index(selected)]
        st.session_state.score += points
        st.success(f"✅ You earned {points} points!")

    if st.button("Next Question"):
        st.session_state.question_index += 1
        st.experimental_rerun()
else:
    st.header("🎉 Game Over!")
    st.subheader(f"Your final score: {st.session_state.score} points")

    if st.button("Restart Game"):
        st.session_state.question_index = 0
        st.session_state.score = 0
        st.experimental_rerun()
