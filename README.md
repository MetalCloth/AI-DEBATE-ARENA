# 🧠 AI Debate Arena

> **LangGraph-powered multi-agent debate system featuring Claude, tools, and real-time Streamlit UI.**

<img width="2076" height="1099" alt="mermaid-diagram-2025-07-14-222150" src="https://github.com/user-attachments/assets/ae1722fb-2075-41ee-be6c-001ce7c2a794" />
      
---

## 🎯 Overview

**AI Debate Arena** is a multi-agent system powered by [LangGraph](https://www.langchain.com/langgraph) that simulates dynamic debates between intelligent agents:

- 🤖 **ProAgent** defends a claim with conviction and tools  
- 🤖 **ConAgent** refutes it with ruthless logic  
- 🧑‍⚖️ **CriticAgent** evaluates each round, scores both sides, and offers feedback  
- 🔁 Infinite looping rounds until the user ends the debate  
- 📜 On-demand final summary with markdown table and verdict  

Built with **Claude 3.5 Haiku**, **Tavily Search**, **LangChain**, and a clean **Streamlit UI**.

---

## 🚀 Features

✅ LangGraph orchestration with parallel agents  
✅ Claude-based Pro/Con agents with ReAct-style prompting  
✅ Integrated tools: direct reasoning + Tavily web search  
✅ Structured critic scoring via Pydantic  
✅ Real-time UI with Streamlit and session state  
✅ Memory across rounds for smarter arguments  
✅ Final markdown-style summary and verdict generation  



---

## 🧩 Tech Stack

| Component        | Technology                     |
|------------------|--------------------------------|
| UI               | Streamlit                      |
| Debate Agents    | Claude 3.5 (via LangChain)     |
| Tools            | Custom tools + Tavily Search   |
| Orchestration    | LangGraph                      |
| Evaluation       | Claude + PydanticOutputParser  |
| Prompting        | Custom ReAct format            |

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-debate-arena.git
cd ai-debate-arena
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your environment variables

Create a `.env` file in the root folder:

```env
ANTHROPIC_API_KEY=your_claude_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🎥 Walkthrough 

https://github.com/user-attachments/assets/bde7679c-fbce-480a-b2f5-dd9d67b939b3

---

## 🧠 How It Works

The app uses LangGraph to define a graph of state transitions:

- A round begins with **ProAgent** and **ConAgent** running in parallel
- **CriticAgent** then scores both arguments, gives feedback + strategy
- A memory step stores round summaries
- Loop continues until user clicks **End Debate**
- You can trigger a **final markdown summary** at the end

---

## 🤝 Acknowledgements

- [LangGraph by LangChain](https://www.langchain.com/langgraph)
- [Anthropic Claude](https://www.anthropic.com)
- [Tavily Search API](https://www.tavily.com)
- [Streamlit](https://streamlit.io)

---



