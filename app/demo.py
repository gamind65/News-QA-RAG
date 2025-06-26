import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import streamlit as st
import os
import json
import getpass

#---------------------
# Set up streamlit interface layout
st.set_page_config(layout="wide")

# Ollama base URL (points to Docker container)
base_url="http://ollama-container:11434"
ollama_client = ollama.Client(host=base_url)

# Load Gemini Key in JSON file
with open("key.json") as f:
    key = json.load(f)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = key["gemini_key"]

#---------------------
# Load cache data
@st.cache_resource()
def load_cache_resource(llm_name="qwen3:1.7b", embed_name="nomic-embed-text"):
    # Load LLM
    ollama_client.pull(model=llm_name)
    ollama_client.pull(model=embed_name)

    llm = ChatOllama(model=llm_name, 
                     base_url=base_url,
                     enable_thinking=True, 
                     Temperature=0.6, 
                     TopP=0.95, 
                     TopK=20, 
                     MinP=0) # Hit setting as best practice listed on Qwen3 HuggingFace 
    
    # Load embedding model
    embeddings = OllamaEmbeddings(model=embed_name,
                                  base_url=base_url
                                  )
    
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
        )

    return llm, llm_gemini, embeddings

llm, llm_gemini, embeddings = load_cache_resource()

#---------------------
# Load resource to initialize compoments
# Query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    Bạn là một trợ lý với nhiệm vụ giúp đỡ người dùng trả lời những câu hỏi liên quan đến bài báo.
    Bạn sẽ thực hiện điều này bằng cách tạo ra thêm 2 câu hỏi khác tương đồng với câu hỏi gốc để 
    tím kiếm những thông tin liên quan trong Vector Database.
    Bằng cách tạo ra thêm những góc mới tương đồng với câu hỏi của người dùng, mục tiêu của bạn là 
    giúp người dùng vượt qua rào cản của việc tìm kiếm thông tin dựa trên khoảng cách tương đối về 
    mặt ngữ nghĩa của chúng. 
    Hãy cố gắng để không trả lời những thông tin tồn tại trong bài báo.
    Nếu bạn không biết câu trả lời, hãy cứ trả lời là không biết.
    Hãy đưa ra câu trả lời trên một dòng mới và đừng lặp lại câu hỏi của người dùng. 
    Hãy đưa ra câu trả lời một cách ngắn gọn.
    Câu hỏi: {question}""",
)

# RAG prompt template
template = """Hãy trả lời câu hỏi mà CHỈ dựa vào thông tin được cho như sau:
{context}
Câu hỏi: {question}"""

#---------------------
# Data chunking
def data_chunking(article, size=512, overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text=article)
    return chunks

# Add created chunks into vector database 
def create_vectordb(text_chunks, embeddings):
    vectordb = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectordb

# Set up retriever
def retriever(llm, vector_db, QUERY_PROMPT):
    retriever_agent = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT,
    )
    return retriever_agent

# Data processing and generate answer
def generate_ans(article, question, llm_choice):
    chunks = data_chunking(article)                                     # Create chunks
    prompt = ChatPromptTemplate.from_template(template)                 # Create prompt template
    vector_db = create_vectordb(chunks, embeddings)                     # Create vector database for embedded chunks
    
    if llm_choice == "Gemini-2.0-flash":
        # Create retriever
        retriever_gemini = retriever(llm_gemini, vector_db, QUERY_PROMPT)   
        
        # Create chain
        chain = (
            {"context": retriever_gemini, "question": RunnablePassthrough()}
            | prompt
            | llm_gemini
            | StrOutputParser()
        )

        # Generate answer
        ans = chain.invoke(question)

    else:
        # Create retriever
        retriever_agent = retriever(llm, vector_db, QUERY_PROMPT)   

        # Create chain
        chain = (
            {"context": retriever_agent, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate answer
        ans = re.sub('<think>(?s:.)*?</think>', '', chain.invoke(question))

    return ans

#---------------------
def main():
    # Default setup
    # initiate 2 columns in streamlit UI
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        # Title
        st.subheader("Article section")

    with col2:
        # Title
        st.header("Vietnamese news article QnA chatbot with RAG")
        st.subheader("Chatbot section")

        
    article = None
    with col1:
        # Textbox for article content
        llm_choice = st.selectbox(label="Choose what model you want to use:",
                     options=("Gemini-2.0-flash", "Qwen3-1.7b"))
        st.write("You selected:", llm_choice)

        article = st.text_input("Article content", None, placeholder="Paste your news article content here...")
        if article is not None:
            st.write("Your article content:")
            st.write(article)
        
    with col2:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Display user input
        container = st.container()
        if query := st.chat_input("What is up?"):
            # Display user message in chat message container
            with container:
                with st.chat_message("user"):
                    st.markdown(query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})

            # Respond
            if article is None: 
                with container:
                    with st.chat_message("assistant"):
                        response = "Please add news article content in the left section first."
                        st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            else:
                response = generate_ans(article, query, llm_choice)
                # Display assistant response in chat message container
                with container:
                    with st.chat_message("assistant"):
                        st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    return 

if __name__ == '__main__':
    main()




