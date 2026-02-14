import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="RAG Chat", layout="centered")

st.title("📄 RAG Question Answering")
st.caption("Powered by FastAPI + LlamaIndex")

# Input box
question = st.text_input("Ask a question from your documents:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=120
            )

        if response.status_code == 200:
            answer = response.json()["answer"]
            st.success("Answer:")
            st.write(answer)
        else:
            st.error(f"Error: {response.status_code}")
            st.text(response.text)
