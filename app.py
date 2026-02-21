import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .title { text-align: center; color: #00d4ff; font-size: 42px; font-weight: bold; padding: 20px; }
    .subtitle { text-align: center; color: #888; font-size: 16px; margin-bottom: 30px; }
    .user-bubble { background-color: #00d4ff; color: black; padding: 10px 15px; border-radius: 15px 15px 0px 15px; margin: 5px 0; max-width: 70%; float: right; clear: both; }
    .ai-bubble { background-color: #1e1e2e; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 0px; margin: 5px 0; max-width: 70%; float: left; clear: both; }
    .chat-container { overflow: hidden; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📄 AI Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">PDF upload karo aur 3 powerful AI features use karo!</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    full_text = " ".join([doc.page_content for doc in documents])
    return vectorstore.as_retriever(), full_text

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

uploaded_file = st.file_uploader("📂 PDF file upload karo", type="pdf")

if uploaded_file is not None:
    with st.spinner("⏳ Document process ho raha hai..."):
        retriever, full_text = process_pdf(uploaded_file.read())

    st.success("✅ Document ready hai! Feature chunno:")

    col1, col2, col3 = st.columns(3)
    with col1:
        btn_qa = st.button("🔍 Search & Q&A", use_container_width=True)
    with col2:
        btn_summary = st.button("📝 Summary", use_container_width=True)
    with col3:
        btn_translate = st.button("🌐 Translate", use_container_width=True)

    if "active_feature" not in st.session_state:
        st.session_state.active_feature = None

    if btn_qa:
        st.session_state.active_feature = "qa"
    if btn_summary:
        st.session_state.active_feature = "summary"
    if btn_translate:
        st.session_state.active_feature = "translate"

    # Q&A Feature
    if st.session_state.active_feature == "qa":
        st.markdown("### 🔍 Document se kuch bhi poochho!")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        question = st.text_input("Apna sawaal likho:", key="qa_input")
        if question:
            with st.spinner("Jawab dhundh raha hoon..."):
                docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])
                prompt = f"""Context ke basis pe jawab do:\n\nContext:\n{context}\n\nSawaal: {question}\n\nJawab:"""
                response = llm.invoke(prompt)
                st.session_state.chat_history.append({"q": question, "a": response.content})

        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="chat-container"><div class="user-bubble">🧑 {chat["q"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-container"><div class="ai-bubble">🤖 {chat["a"]}</div></div>', unsafe_allow_html=True)

    # Summary Feature
    elif st.session_state.active_feature == "summary":
        st.markdown("### 📝 Document Summary")
        with st.spinner("Summary ban rahi hai..."):
            prompt = f"Is document ka aik acha aur detailed summary banao:\n\n{full_text[:4000]}"
            response = llm.invoke(prompt)
            st.markdown(f'<div class="chat-container"><div class="ai-bubble">🤖 {response.content}</div></div>', unsafe_allow_html=True)

    # Translate Feature
    elif st.session_state.active_feature == "translate":
        st.markdown("### 🌐 Document Translate karo")
        language = st.selectbox("Language chunno:", ["Urdu", "Arabic", "French", "Spanish", "Chinese", "German"])
        if st.button("Translate karo!"):
            with st.spinner(f"{language} mein translate ho raha hai..."):
                prompt = f"Is document ko {language} mein translate karo:\n\n{full_text[:4000]}"
                response = llm.invoke(prompt)
                st.markdown(f'<div class="chat-container"><div class="ai-bubble">🤖 {response.content}</div></div>', unsafe_allow_html=True)