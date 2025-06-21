import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document  # ✅ Needed for summary
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv(dotenv_path="C:\\assistant\\.env.example")

# API Key confirmation (optional)
print("API KEY:", os.getenv("OPENAI_API_KEY"))


# ✅ Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# ✅ Split into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# ✅ Build vector database
def get_vectorstore(text_chunks):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# ✅ Setup Conversational QA chain
def get_conversation_chain(vectorstore):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )


# ✅ Summary logic with Document wrapper
def generate_summary(text):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff")
    docs = [Document(page_content=text)]
    return chain.run(docs)


# ✅ Generate logic-based questions (placeholder)
def generate_challenge_questions(vectorstore):
    return [
        "What is the objective of the uploaded document?",
        "List one functional requirement mentioned.",
        "What does the assistant do in Challenge Me mode?"
    ]


# ✅ Evaluate user's answer with context
def evaluate_answer(question, user_answer, vectorstore):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    llm = ChatOpenAI()
    prompt = f"""You are a document evaluator.
Question: {question}
User's Answer: {user_answer}
Document Context:
{context}

Evaluate the user’s answer. Say whether it’s correct and explain why based on the document."""
    return llm.predict(prompt)


# ✅ Handle chat messages
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


# ✅ Streamlit App
def main():
    st.set_page_config(page_title="Smart PDF Assistant", page_icon="📄")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("📑 Smart Assistant for Research Summarization")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📤 Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)

                # ✅ Auto Summary
                try:
                    summary = generate_summary(raw_text)
                    st.subheader("📝 Document Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Summary failed: {e}")

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

        # ✅ Challenge Me Mode
        if st.session_state.vectorstore:
            st.subheader("🎯 Challenge Me")
            if st.button("Start Quiz"):
                st.session_state.questions = generate_challenge_questions(st.session_state.vectorstore)

            if "questions" in st.session_state:
                for i, q in enumerate(st.session_state.questions):
                    st.markdown(f"**Q{i+1}: {q}**")
                    user_ans = st.text_input(f"Your answer to Q{i+1}:", key=f"ans_{i}")
                    if user_ans:
                        feedback = evaluate_answer(q, user_ans, st.session_state.vectorstore)
                        st.success(feedback)


if __name__ == '__main__':
    main()
