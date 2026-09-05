"""
Streamlit frontend for PolicyPilot.
"""

import streamlit as st
import requests

API_URL = "https://policypilot-api-c3yw.onrender.com"

st.set_page_config(page_title="PolicyPilot", page_icon="📄")
st.title("PolicyPilot")
st.caption("Ask questions about policy documents — every answer is grounded and cited.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None  # None = using the static default corpus
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Sidebar: upload your own document ---
with st.sidebar:
    st.subheader("Upload your own document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process this document"):
            with st.spinner("Uploading and processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    response.raise_for_status()
                    data = response.json()

                    st.session_state.session_id = data["session_id"]
                    st.session_state.uploaded_filename = data["filename"]
                    st.session_state.messages = []  # fresh chat for the new document

                    st.success(f"Processed '{data['filename']}' into {data['chunks_stored']} chunks.")
                except requests.exceptions.ConnectionError:
                    st.error("Couldn't reach the backend. Is uvicorn running?")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # Show which document is currently active, and let the user switch back
    if st.session_state.session_id:
        st.info(f"Currently asking about: **{st.session_state.uploaded_filename}**")
        if st.button("Switch back to default policy corpus"):
            st.session_state.session_id = None
            st.session_state.uploaded_filename = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Currently using the default pre-loaded policy corpus.")

# --- Main chat area ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src['source']}`, page {src['page']}")

user_question = st.chat_input("Ask about your policy documents...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            try:
                payload = {"question": user_question}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id

                response = requests.post(f"{API_URL}/ask", json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

            except requests.exceptions.ConnectionError:
                answer = "Couldn't reach the backend API. Make sure uvicorn is running."
                sources = []
            except requests.exceptions.HTTPError as e:
                answer = f"The backend returned an error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"- `{src['source']}`, page {src['page']}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})