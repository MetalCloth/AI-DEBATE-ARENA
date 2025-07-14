🗣️ AI Debate Arena
AI Debate Arena is an advanced multi-agent debate system powered by LangGraph, Anthropic Claude 3.5, and Streamlit. It simulates formal Oxford-style debates between AI agents — ProAgent, ConAgent, and a CriticAgent — with stepwise reasoning, memory, and dynamic strategy refinement.

📌 Core Idea: You input a debate topic. The system runs infinite rounds of structured arguments until you hit "End Debate". Every round is judged and critiqued, with arguments updated based on feedback.

🚀 Features
🔁 Infinite Debate Loop: Pro and Con agents continue debating endlessly until manually stopped.

🧠 ReAct Agents: Both sides use LangChain’s ReAct paradigm with tools (Tavily, direct response).

🧑‍⚖️ Critic Agent: Evaluates rounds, gives scores, feedback, and strategy tips.

🧾 Round Memory: Each round is stored, including score, feedback, and argument history.

📊 Dynamic Strategy Update: Agents adapt based on feedback to improve future arguments.

🧵 Live Streamlit UI: Interactive web app to monitor, control, and analyze the debate in real time.

📜 Markdown Transcript + Summary: Generates full debate summary in markdown format.

📸 Demo


UI includes:

Round-by-round collapsible arguments

On-demand toggles for Pro, Con, Critic

Debate summary button

Debug mode for inspecting state

🧠 Agents
ProAgent: Defends the claim fiercely, uses facts, logic, and bold rhetoric.

ConAgent: Refutes the claim with cold analysis, sarcasm, and intellectual takedowns.

CriticAgent: Scores both arguments, provides tactical feedback, and declares round winner.

⚙️ Technologies Used
🔁 LangGraph

🤖 LangChain

🧠 Anthropic Claude 3.5 Haiku

🌐 TavilySearch

🧪 Streamlit

🧊 Python 3.10+

