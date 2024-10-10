import streamlit as st
import os
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_retrieval_chain , create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory


from dotenv import load_dotenv
load_dotenv()


# os.environ['HF_TOKEN']=os.getenv("HF__TOKEN")

# HF__TOKEN = os.getenv("HF__TOKEN")
# if HF__TOKEN is None:
#     print("Hugging Face token not found! Please check your .env file.")
# else:
#     os.environ['HF_TOKEN'] = HF__TOKEN

os.environ['HF_TOKEN'] = "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"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    
# embeddings = HuggingFaceEmbeddings(model_name="all-MinLM-L6-v2")

#streamlit app
st.title("Conversational AI for Finance: Document Retrieval and Analysis with Chat History and PDF Integration")
st.write("Upload pdf ")

#input the groq api key
api_key=st.text_input("Enter your groq api key :" , type="password")

#check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key , model_name="Gemma2-9b-It")
    
    session_id=st.text_input("Session ID" , value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store={}
        
    uploaded_files=st.file_uploader("choose a pdf file", type="pdf",accept_multiple_files=True)
    #process uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
                
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            
    #split and create embeddings for the documnets
        text_splitter =RecursiveCharacterTextSplitter(chunk_size=5000 , chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore= Chroma.from_documents(documents=splits , embedding=embeddings ,  persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()
        
        
        contextualize_q_system_prompt=(
            "given a chat history and the latest user question"
            "which might refrence context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. do not answer the question"
            "just reformulate it if needed and otherwise return it as it is."
        )
    
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                
                ]
            )
        
        history_aware_retriever = create_history_aware_retriever(llm , retriever , contextualize_q_prompt)
        
        #answer question prompt 
        
        system_prompt=(
            "you are an assistant for question answering tasks"
            "use the following pieces of retrieved context to answer"
            "the question . if you dont know the answer say that you"
            "dont know . use three senetences maximum and keep the "
            "answer concise"
            "/n/n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain= create_stuff_documents_chain(llm , qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever , question_answer_chain)
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain= RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input" , 
            history_messages_key="chat_history",
            output_messages_key="answer" 
            
        )
        
        user_input=st.text_input("your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                
                
                config={
                    "configurable":{"session_id":session_id}
                }, #constructs a key "abc123" in store
            )
            
            st.write(st.session_state.store)
            st.success(f"assistant:{response['answer']}")
            st.write("chat_history" , session_history.messages)
else:
    st.warning("please enter the groq api key")