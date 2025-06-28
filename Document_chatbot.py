# 필요한 라이브러리 임포트
import streamlit as st
import tiktoken
import torch
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback


# UI + 전체 챗봇 로직
def main():
    st.set_page_config(
        page_title="📄 DocuGenie - Document QA Chatbot",
        page_icon="🧞‍♂️",
        layout="wide"
    )

    # 상단 헤더 영역
    st.markdown(
        """
        <div style='text-align: center; margin-top: 30px; margin-bottom: 40px;'>
            <h1 style='font-size: 3rem; color: #4F8BF9;'>🧞‍♂️ DocuGenie</h1>
            <p style='font-size: 1.2rem; color: #555;'>PDF, DOCX, PPTX 문서를 업로드하고 궁금한 점을 물어보세요.</p>
        </div>
        """, unsafe_allow_html=True
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요! 문서에 대해 궁금한 점을 물어보세요."
        }]

    # 사이드바
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        uploaded_files = st.file_uploader("📎 파일 업로드 (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API Key", key="chatbot_api_key", type="password")
        button = st.button("🚀 챗봇 시작")

        st.markdown("---")
        st.info("DocuGenie는 업로드된 문서에서 정보를 추출하여 대화형으로 답변합니다.")

        if button:
            if not openai_api_key:
                st.warning("API 키를 입력해주세요.")
                st.stop()
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    # 채팅 메시지 표시
    for message in st.session_state.messages:
        avatar_icon = "🧞‍♂️" if message["role"] == "assistant" else "🧑🏻"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    # 입력 + 응답
    if st.session_state.conversation is not None:
        if query := st.chat_input("질문을 입력하세요...", key="chat_input_1"):
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user", avatar="🧑🏻"):
                st.markdown(query)

            with st.chat_message("assistant", avatar="🧞"):
                chain = st.session_state.conversation
                with st.spinner("🧠 생각 중입니다..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result["chat_history"]

                    response = result["answer"]
                    source_docs = result["source_documents"]

                    st.markdown(response)

                    with st.expander("📚 참고 문서 보기"):
                        for i, doc in enumerate(source_docs[:3]):
                            st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
                            st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": response})


# 🔧 텍스트 토큰 길이 측정 함수
def tiktken_len(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# 🔧 문서 파싱 함수
def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
            logger.info(f"File {file_name} saved successfully.")

        if '.pdf' in file_name:
            loader = PyPDFLoader(file_name)
        elif '.docx' in file_name:
            loader = Docx2txtLoader(file_name)
        elif '.pptx' in file_name:
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)

    return doc_list


# 🔧 문서 분할
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktken_len
    )
    return text_splitter.split_documents(text)


# 🔧 벡터스토어 생성
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.from_documents(text_chunks, embeddings)


# 🔧 Conversational Chain 설정
def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
