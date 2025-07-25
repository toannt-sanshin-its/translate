import streamlit as st
import requests

# Configure the page
st.set_page_config(page_title="Japanese→English Chat", page_icon="💬", layout="centered")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("💬 Japanese→English Translate AI")
st.write("Enter a Japanese sentence and get an English translation in real time.")

# Input form
with st.form(key="translate_form", clear_on_submit=True):
    user_input = st.text_input("Your Japanese text:")
    submit_button = st.form_submit_button(label="Translate")

if submit_button and user_input:
    # Call the FastAPI endpoint
    try:
        response = requests.get(
            "http://localhost:8000/translate",
            params={"text": user_input},
            timeout=100
        )
        response.raise_for_status()
        data = response.json()
        translated = data.get("translation", "(no translation)")
        score = data.get("score")
    except Exception as e:
        st.error(f"Error calling translation API: {e}")
    else:
        # Append to chat history
        st.session_state.history.append({
            "input": user_input,
            "output": translated,
            "score": score
        })

# Display chat history
if st.session_state.history:
    for msg in st.session_state.history:
        st.markdown(f"**You**: {msg['input']}")
        st.markdown(f"**Bot**: {msg['output']} \t  _Score: {msg['score']}_")

# Footer
st.markdown("---")
st.markdown("Built with [FastAPI] and [Streamlit]")


# running:
# pip install streamlit requests
# streamlit run translate_streamlist.py
