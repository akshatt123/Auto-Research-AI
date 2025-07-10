import arxiv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

def fetch_arxiv_papers(topic, max_results=5):
    search = arxiv.Search(query=topic, max_results=max_results)
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return papers

def build_vectorstore(papers):
    abstracts = [paper['summary'] for paper in papers]
    docs = [Document(page_content=abs) for abs in abstracts]
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
