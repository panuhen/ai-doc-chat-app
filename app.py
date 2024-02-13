import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain_community.llms.huggingface_hub import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
            
            


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":eyes:")
    st.title("Chat with your PDFs :eyes:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        st.info("Please submit and process your PDF documents")
        


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                
        st.divider()
        st.subheader("⚠️DISCLAIMER")
        st.write("Please be advised that this website utilizes the OpenAI API to provide certain functionalities.")
        st.write("Users are cautioned against uploading or inputting any business-sensitive or personally sensitive material through this platform.")

    # Show chat input only if conversation has been initialized
    if st.session_state.conversation is not None:
        user_question = st.chat_input()
    # Trigger AI bot to give a welcome message when chat input is initiated
        if st.session_state.chat_history is None:
            with st.chat_message("AI"):
                st.write("Hello! I am your document assistant bot. How may I help you?")

        if user_question:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()