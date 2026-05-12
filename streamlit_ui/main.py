import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(
    page_title="Healthcare Assistant",
    layout="wide"
)

st.title("🩺 Healthcare Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


query = st.chat_input(
    "Ask your medical question..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    payload = {
        "query": query,
        "mode": "general"
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    data = response.json()

    answer = data["answer"]

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander(
            "View Retrieved Sources"
        ):

            for source in data["sources"]:

                st.write(source[:500])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
