# 🤖 AutoResearcher.ai — Agentic RAG Research Assistant

AutoResearcher.ai is an intelligent multi-agent research system that autonomously performs:
- Topic planning
- Retrieval-augmented research
- Report generation with citations

Built using CrewAI, LangChain, FAISS, and OpenAI.

---

## 🚀 Features

- 🧠 **LLM Planning Agent** – breaks the research topic into subtopics  
- 🔍 **RAG Research Agent** – retrieves ArXiv papers & generates insights  
- 📝 **Markdown Writer Agent** – formats a structured research report  
- 💬 **Streamlit UI (optional)** – enter a topic and download the final report  
- 📁 **Markdown to PDF ready**  

---

## 🛠️ Tech Stack

- [CrewAI](https://github.com/joaomdmoura/crewai)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI GPT-4 / 3.5](https://platform.openai.com/)
- [ArXiv API](https://arxiv.org/help/api/index)
- [Streamlit (Optional UI)](https://streamlit.io/)

---

## 🧩 Project Structure

```
AutoResearcher.ai/
├── main.py                    
├── .env                      
├── agents/
│   ├── planner.py             
│   ├── researcher.py         
│   └── writer.py              
├── tools/
│   └── rag_tool.py            
├── streamlit_app/
│   └── app.py                 
├── README.md
├── .gitignore
```

---

## 🧪 Setup Instructions

1. **Clone the repo**  
```bash
git clone https://github.com/akshatt123/autoresearcher_ai.git
cd autoresearcher_ai
```

2. **Create virtual environment**  
```bash  
conda conda create --name agentic_rag python=3.12
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Add API Key**  
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
# or
GROQ_API_KEY=your_groq_key
```

5. **Run CLI version**  
```bash
python main.py
```

6. **Run Streamlit UI (Optional)**  
```bash
streamlit run streamlit_app/app.py
```
---