import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import json, torch, os, re
import streamlit as st
from setproctitle import setproctitle
setproctitle("streamlit_openai")

from deep_translator import GoogleTranslator
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

import streamlit as st
import time


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

embedding_model = HuggingFaceEmbeddings(
                model_name='jhgan/ko-sroberta-multitask',
                model_kwargs={'device': "cuda:0"},
                encode_kwargs={'normalize_embeddings' : True}
            )


# 크로마 DB 접속, 도큐먼트의 컨텐츠와 메타 정보 메모리에 불러오기
chroma_host = '호스트를 입력'
chroma_port = '8000'
chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)    
collection_name = "ai_nlp_langchain_kr_v1.3"

# @st.cache_data(show_spinner=False, allow_output_mutation=True, ttl=3600)
def translate(corpus):
    rst = GoogleTranslator(source='auto', target='ko').translate(text=corpus) 
    return rst
    
# @st.cache_data(show_spinner=True, ttl=3600, hash_funcs={chromadb.HttpClient: id})    
def get_collection_documents():
    collection = chroma_client.get_collection(collection_name)
    data = collection.get()
    documents = data['documents']
    metadatas = data['metadatas']
    return documents, metadatas

class Document_create:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def trans_data():
    documents, metadatas = get_collection_documents()
    
    output_documents_representation = [{
        'page_content': doc,
        'metadata': {'doc': meta['doc'], 'page': meta['page'], 'origin_cont': meta['origin_cont']}
    } for doc, meta in zip(documents, metadatas)]
    
    docs = [Document_create(page_content=doc['page_content'], metadata=doc['metadata']) for doc in output_documents_representation]

    return docs

# 불러온 메모리에서 검색하기
@tool('retriever')
def retriever(query):
    weights = [0.6, 0.4]

    chroma_docs = trans_data()
    
    bm25_retriever = BM25Retriever.from_documents(chroma_docs)
    bm25_retriever.k = 3
    
    chroma_retriever = Chroma.from_documents(chroma_docs, embedding_model)
    chroma_retriever = chroma_retriever.as_retriever(search_kwargs={'k':3})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=weights
    )

    response = ensemble_retriever.invoke(query)
    docs = [response[0].page_content, response[1].page_content, response[2].page_content]
    meta = [response[0].metadata, response[1].metadata, response[2].metadata]
    return docs, meta

tools = [retriever]

@st.cache_resource
def load_model():
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key='openai_api_key', streaming=True)
    
llm = load_model()
memory_key = "history" #대화 내역을 기록할 메모리 키입니다.

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

#chat gpt에게 줄 prompt입니다
system_message = SystemMessage(
    content=(
        "You are a nice customer service agent."
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
        "If you don't know the answer, just say you don't know. Don't try to make up an answer."
        "Make sure to answer in Korean"
        "반드시 한국어로 대답하세요"
        "줄글로 설명하세요"
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

# 프롬프트에 따라 답변을 생성하도록 agent를 생성 및 실행합니다
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools, # 여기서의 tool은 데코레이터로 지정하고 tools 리스트로 더 지정할 수 있습니다.
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

####대화 기록 관리 관련 함수 ####
def load_chat_history():
    '''
    저장된 대화 기록이 있는 경우 time, prompt, response를 요소로 하는 튜플 리스트, 
    기록이 없으면 빈 리스트를 반환
    '''
    try:
        with open("chat_history.json", "r") as file:
            chat_history = json.load(file)
    except FileNotFoundError:
        return []

    return chat_history

def save_chat_history(chat_history):
    '''대화 기록을 JSON 파일에 저장'''
    with open("chat_history.json", "w") as file:
        json.dump(chat_history, file)


def process_message(prompt, response):
    '''대화 기록 업데이트 및 저장'''
    if not st.session_state.messages:
        chat_history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prompt, response))
        save_chat_history(chat_history)


def main():
    st.header('💬 사내 챗봇')
    st.sidebar.title("Chat History")
    user_messages = [message['content'] for message in st.session_state.get("messages", []) if message["role"] == "user"]

    for idx, msg in enumerate(user_messages, start=1):
        st.sidebar.markdown(f"**질문 {idx}.** {msg}")
        
    llm = load_model()
    
    chat_history = load_chat_history()

        
    if st.sidebar.button("새로운 채팅 시작"):
        st.session_state.messages = []   
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}] #기본 메시지
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])
    
    if prompt := st.chat_input():
        # 사용자 인풋을 prompt 변수로 받습니다
        st.session_state.messages.append({"role": "user", "content": prompt})
        # session state의 메시지에, role과 content로 구성된 dictionary로 받아온 유저 프롬프트를 저장합니다.
        with st.chat_message("user"):
            st.markdown(prompt)
        prompt = translate(prompt)
        # 프롬프트를 한국어로 바꿔 리트리버 시스템으로 넘깁니다
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            retrived_output = retriever(translate(prompt))  
            meta = retrived_output[1]
            # 검색 결과를 받아옵니다 (문서, 메타 데이터로 구성된 튜플로, 문서는 문자열이며 메타데이터는 딕셔너리입니다. 
            # 키값은 doc, page, origin_cont로 doc은 문서의 제목(문자열), page는 청크의 문서에서의 페이지(정수), origin_doc은 원문 문서 내용(문자열)입니다)           
            refer = f"아래는 검색된 문서 관련 정보입니다. \n\n1. '{meta[0]['doc']}'의 {meta[0]['page']}페이지 입니다.\n\n2. '{meta[1]['doc']}'의 {meta[1]['page']}페이지 입니다. \n\n3. '{meta[2]['doc']}'의 {meta[2]['page']}페이지 입니다.\n\n"
            # 검색 결과를 동기로 출력합니다
            for chunk in (refer).split(' '):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            # gpt api의 추론 결과를 받아와 동기로 출력합니다    
            result = agent_executor({"input": prompt})
            for chunk in (result['output']).split(' '):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response}) # 챗봇의 대답도 session state의 메시지로 추가합니다


if __name__ == "__main__":
    clear_cache()
    main()


