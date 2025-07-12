import arxiv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from crewai import LLM
import os
from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# llm=LLM(model="gemini/gemini-2.0-flash",
#                            temperature=0.5,
#                            api_key=os.getenv("GOOGLE_API_KEY"),
#                            stream=True)


def fetch_arxiv_papers(topic, max_results=5):
    papers = []
    client = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=max_results)
    results = client.results(search)
    for result in results:
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
    retrivalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    return retrivalQA
