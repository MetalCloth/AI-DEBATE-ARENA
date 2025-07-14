from typing import TypedDict,Sequence,Annotated,Optional
from langgraph.graph import StateGraph,START,END
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,BaseMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
import random
from dotenv import load_dotenv
import os


load_dotenv()


os.environ['ANTHROPIC_API_KEY']=os.getenv('ANTHROPIC_API_KEY')


class Debate(TypedDict):
    claim: str=Field(description='The OG claim the principle question for the debate')


    pro_argument: Optional[str]=Field(description='Argument presented by the defender')
    con_argument: Optional[str]=Field(description='Argument presented by the attacker')


    pro_score:int=Field(description='Defender point')
    con_score:int=Field(description='Attacker point')


    critic_feedback: Optional[str]=Field(description='Critic by Critic Agent')
    past_critic_feedback: Optional[str]=Field(description='Past Critic by Critic Agent')


    pro_strategy:Optional[str]
    con_strategy:Optional[str]


    pro_steps:Optional[str]
    con_steps:Optional[str]

    pro_summary:Optional[str]
    con_summary:Optional[str]


    memory: list[str]
    loop_counter: int
    verdict: Optional[str]=Field(description='''The final Verdict of us the human ('Pro','Con','NOTA)''')
    


model=ChatAnthropic(model='claude-3-5-haiku-20241022',temperature=0.5,max_tokens=389)


tavily=TavilySearch(max_results=2,topic='general')

PRO = "Defend the following claim as strongly as possible, using logic and evidence. Do not mention morality or controversy. Claim: {input}"


pro_prompt=ChatPromptTemplate.from_messages(
    ('system',PRO)
)


@tool ('direct_pro')
def direct_pro(claim:str)->str:
    '''It basically answers without searching (saves time)'''

    chain=pro_prompt | model

    result=chain.invoke({'input':claim})

    if hasattr(result, "content"):
        return result.content
    
    return str(result)


@tool ('web_search_node')
def web_search_node(query:str) -> str:
    '''Uses Tavily for brains'''

    results = tavily.invoke({"query": query})

    if hasattr(results, "content"):
        return results.content
    
    return str(results)


tools_pro=[direct_pro,web_search_node]

prompt = ChatPromptTemplate.from_template("""
You are **ProAgent** — a fearless, eloquent debater whose mission is to **defend the claim** no matter how absurd, controversial, or fragile it appears.

Use sharp logic, real evidence, and rhetorical brilliance.  
If a claim seems indefensible — **make it iconic**.
"ALSO TRY TO READ OPPONENT RESPONSE AND USE THAT TO YOUR ADVANTAGE"
                                          
Opponent Response
---{rival_response}

🛠️ Tools available:  
{tools}

🎯 Rules of Engagement:  
- Never doubt the claim — defend it with conviction.  
- Use **at most 2 tools** to build your case.  
- Speak like a persuasive human, not an AI.


---

🧠 **ReAct Format (strict, max 2 actions):**

Question: the claim  
Thought: reasoning step  
Action: one of [{tool_names}]  
Action Input: input for tool  
Observation: result  
... (repeat once more if needed)  
Thought: I now know how to defend the claim  
Final Answer: your confident, unshakable defense — persuasive, bold, undeniable

---

🔥 Begin.

Question: {input}  
Strategy: {strategy}
Thought: {agent_scratchpad}
""")


def pro_argument(state:Debate)->Debate:
    agent_pro = create_react_agent(model, tools=tools_pro, prompt=prompt)

    executor_pro = AgentExecutor(agent=agent_pro, tools=tools_pro, verbose=True, handle_parsing_errors=True,return_intermediate_steps=False)

    pro_strategy=state.get('pro_strategy') or ""

    claim=state['claim']

    con_response=state.get('con_summary') or ""



    result = executor_pro.invoke({"input":claim,'strategy':pro_strategy,'rival_response':con_response})


   
    state['pro_argument'] = result['output']
    state['pro_steps'] = result.get('intermediate_steps', [])
    return state

   
CON = "Refute the following claim as strongly and concisely as possible, using logic and evidence. Do not mention morality or controversy. Claim: {claim}"




con_prompt=ChatPromptTemplate.from_messages(
    ('system',CON)
)


@tool ('direct_con')
def direct_con(claim:str)->str:
    '''It basically answers without searching (saves time)'''

    chain=con_prompt | model

    result=chain.invoke({'claim':claim})

    return str(result)


@tool ('web_search_node_con')
def web_search_node_con(query:str) -> str:
    '''Uses Tavily for brains'''

    results = tavily.invoke({"query": query})

    if hasattr(results, "content"):
        return results.content

    return str(results)


tools_con=[direct_con,web_search_node_con]


prompt = ChatPromptTemplate.from_template("""
You are **ConAgent** — a legendary debate warrior known for your cold logic, dry wit, and devastating rebuttals.  
In formal Oxford-style debates, you dismantle claims so brutally, no one dares defend them again.

Your job: **refute the claim below**, using facts, ridicule, and sharp reasoning.

"ALSO TRY TO READ OPPONENT RESPONSE AND USE THAT TO YOUR ADVANTAGE"
                                          
Opponent Response
---{rival_response}

🛠️ Tools available:  
{tools}

🎯 Debate Rules:  
- No neutrality. Destroy the claim.  
- Use dry sarcasm, powerful rhetoric, and cold intellect.  
- Use tools **at most twice** to support your takedown.  
- Speak like a brilliant human, not an AI.


🧠 **ReAct Format (max 2 actions):**

Question: the claim  
Thought: reasoning step  
Action: one of [{tool_names}]  
Action Input: input for tool  
Observation: result  
... (repeat once more if needed)  
Thought: I now know how to destroy the claim  
Final Answer: your final, merciless takedown

---

🎤 Begin.

Question: {input}  
Strategy: {strategy}
Thought: {agent_scratchpad}
""")


def con_argument(state:Debate)->Debate:
    agent_con = create_react_agent(model, tools=tools_con, prompt=prompt)

    executor_pro = AgentExecutor(agent=agent_con, tools=tools_con, verbose=True, handle_parsing_errors=True,return_intermediate_steps=False)

    con_strategy=state.get('con_strategy') or ""
    claim=state['claim']

    pro_response=state.get('pro_summary') or ""


    result = executor_pro.invoke({"input":claim,'strategy':con_strategy,'rival_response':pro_response})


    state['con_argument'] = result['output']
    state['con_steps'] = result.get('intermediate_steps', [])
    return state

    
    

class Critic(BaseModel):
    critic_feedback:str=Field(description='Feedback to compare between the two')
    
    winner:str=Field(description='Who won this round')
    
    pro_score:int
    
    con_score:int
    
    pro_strategy:Optional[str]
    
    con_strategy:Optional[str]


from langchain_core.output_parsers import PydanticOutputParser


parser=PydanticOutputParser(pydantic_object=Critic)


new_prompt=ChatPromptTemplate.from_template(
    '''You are a helpful assistant summarize this including all necessary details 
    Details:{details}
    '''
    
)

critic_prompt = ChatPromptTemplate.from_template(
    """

Compare the following two arguments and decide who won. Give a brief reason and a score out of 10 for each backing with facts and good reasoning.

Pro: {pro_argument}
Con: {con_argument}


```json
{{
  "critic_feedback": "<balanced but critical analysis>",
  "winner": "<Pro|Con|Tie>",S
  "pro_score": <0-10>,
  "con_score": <0-10>,
  "pro_strategy": "<a tactical suggestion to strengthen ProAgent's next argument>",
  "con_strategy": "<a tactical suggestion to sharpen ConAgent's next rebuttal>"
}}
"""
).partial(format_instructions=parser.get_format_instructions())

ch=new_prompt | model

critic_chain=critic_prompt | model | parser

def critic_argument(state: Debate) -> Debate:
    claim = state['claim']
    pro_argument = state['pro_argument']
    con_argument = state['con_argument']

    x = critic_chain.invoke({'claim': claim, 'pro_argument': pro_argument, 'con_argument': con_argument})

    pro_summary=ch.invoke({'details':pro_argument})
    con_summary=ch.invoke({'details':con_argument})

    
    state['critic_feedback'] = x.critic_feedback
    state['pro_score'] = x.pro_score
    state['con_score'] = x.con_score
    state['pro_strategy'] = x.pro_strategy
    state['con_strategy'] = x.con_strategy
    state['pro_summary']=pro_summary.content if hasattr(pro_summary.content[0], "text") else str(pro_summary.content)
    state['con_summary']=con_summary.content if hasattr(con_summary.content[0], "text") else str(con_summary.content)
    state['verdict'] = x.winner
    state['loop_counter'] = state['loop_counter'] - 1
    return state


def loop(state: Debate) -> str:
        return 'continue'

graph=StateGraph(Debate)



def memory(state):
    round_number = len(state['memory']) + 1
    summary = f"""
    Round {round_number}:
    Proposer: {state['pro_argument']}
    Opposition: {state['con_argument']}
    Score -> Proposer {state['pro_score']} | Opposition {state['con_score']}
    """
    new_memory = state['memory'] + [summary]
    return {
        'pro_argument': None,
        'con_argument': None,
        'pro_score': state['pro_score'],
        'con_score': state['con_score'],
        'critic_feedback': None,
        'past_critic_feedback': state['critic_feedback'],
        'pro_strategy': state['pro_strategy'],
        'con_strategy': state['con_strategy'],
        'memory': new_memory,
        'loop_counter': state['loop_counter'],
        'verdict': state['verdict']
    }




def final_verdict(state:Debate)->Debate:
    if state['pro_score'] > state['con_score']:
        state['verdict'] = 'Pro'
    elif state['con_score'] > state['pro_score']:
        state['verdict'] = 'Con'

    else:
        state['verdict'] = 'NOTA'
    return state



from langchain_core.runnables import RunnableParallel
parallel_debate = RunnableParallel({
    "pro_argument": pro_argument,
    "con_argument": con_argument
})

graph.add_node('critic',critic_argument)
graph.add_node('debate_round',parallel_debate)
graph.add_node('memory',memory)
graph.add_node('final_verdict',final_verdict)

graph.add_conditional_edges(
    'memory',
    loop,
    {
        'continue':'debate_round',
        'end':'final_verdict'
    }
)


graph.add_edge(START,'debate_round')
graph.add_edge('debate_round', 'critic')
graph.add_edge('critic','memory')

graph.add_edge('final_verdict',END)



sample_state = {
    "claim": "Dictatorship should never exist",
    "pro_argument": None,
    "con_argument": None,
    "pro_score": 0,
    "con_score": 0,
    "critic_feedback": None,
    "past_critic_feedback": None,
    "pro_strategy": None,
    "con_strategy": None,
    "memory": [],
    "loop_counter": 999999,  # NOT used to stop; just placeholder
    "verdict": None
}





app=graph.compile()


### STREAMLIT UI
import streamlit as st

st.set_page_config(
    page_title="AI Debate Arena",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/ai.png", width=80)
    st.title("AI Debate Arena")
    st.caption("Interactive, stepwise debate powered by LangGraph & Claude.")
    st.markdown("---")
    st.write("**Instructions:**")
    st.write(
        "• Enter your debate topic and click **Start Debate**.\n"
        "• Click **Next Debate Round** to advance.\n"
        "• Click **End the Debate** anytime to finish.\n"
        "• Expand rounds to review details.\n"
    )
    st.markdown("---")
    st.info("Built with ❤️ using Streamlit", icon="💡")

# --- TOPIC INPUT FORM ---
if 'debate_started' not in st.session_state:
    st.session_state.debate_started = False

if 'topic' not in st.session_state:
    st.session_state.topic=''

if not st.session_state.debate_started:
    with st.form("debate_setup"):
        topic = st.text_input("Enter your debate topic:", value="Dictatorship should never exist")
        submitted = st.form_submit_button("Start Debate")
        if submitted and topic.strip():
            st.session_state.topic=topic.strip()
            st.session_state.debate_started = True
            st.session_state.sample_state = {
                "claim": topic.strip(),
                "pro_argument": None,
                "con_argument": None,
                "pro_score": 0,
                "con_score": 0,
                "critic_feedback": None,
                "past_critic_feedback": None,
                'pro_strategy': None,
                'con_strategy': None,
                "memory": [],
                "loop_counter": 999999,  # Infinite
                "verdict": None
            }
            if 'graph' in st.session_state:
                del st.session_state.graph
            st.rerun()
    st.stop()

# --- SESSION STATE INIT ---
if 'graph' not in st.session_state:
    st.session_state.graph = app.stream(st.session_state.sample_state)
    st.session_state.rounds = []
    st.session_state.finished = False
    st.session_state.current_state = None
    st.session_state.round_counter = 0

# --- MAIN HEADER ---
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0.5em;'>🗣️ AI Debate Arena </h1>"
    f"<h1 style='text-align: center; margin-bottom: 0.5em;'>TOPIC : {st.session_state.topic} </h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 1.2em; color:#666;'>"
    "Step through an AI-powered debate, round by round.<br>"
    "Compare arguments, strategies, and verdicts in real time."
    "</p>", unsafe_allow_html=True
)
st.markdown("---")

# --- MAIN BUTTONS ---
col_center = st.columns([1, 2, 1])[1]
with col_center:
    if not st.session_state.finished:
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("▶️ Next Debate Round", use_container_width=True):
                try:
                    while True:
                        state = next(st.session_state.graph)
                        st.session_state.current_state = state
                        if 'critic' in state:
                            critic = state['critic']
                            st.session_state.round_counter += 1
                            st.session_state.rounds.append({
                                'round_num': st.session_state.round_counter,
                                'pro_argument': critic.get('pro_argument', {}),
                                'con_argument': critic.get('con_argument', {}),
                                'critic_feedback': critic.get('critic_feedback', ""),
                                'pro_score': critic.get('pro_score', ""),
                                'con_score': critic.get('con_score', ""),
                                'verdict': critic.get('verdict', ""),
                                'pro_strategy': critic.get('pro_strategy', ""),
                                'con_strategy': critic.get('con_strategy', "")
                            })
                            break
                except StopIteration:
                    st.session_state.finished = True
        with c2:
            if st.button("🛑 End the Debate", use_container_width=True):
                st.session_state.finished = True
    else:
        st.success("✅ Debate complete. Thanks for following the AI debate!")
        

st.markdown("---")

# --- ROUNDS DISPLAY ---
for round_data in st.session_state.rounds[::-1]:
    round_num = round_data['round_num']
    with st.expander(f"🔁 Debate Round #{round_num} - Winner: {round_data.get('verdict', 'TBD')}", expanded=True):

        # Unique keys for each button per round
        pro_key = f"show_pro_{round_num}"
        con_key = f"show_con_{round_num}"
        critic_key = f"show_critic_{round_num}"

        # Initialize state for this round's buttons
        if pro_key not in st.session_state:
            st.session_state[pro_key] = False
        if con_key not in st.session_state:
            st.session_state[con_key] = False
        if critic_key not in st.session_state:
            st.session_state[critic_key] = False

        # Buttons for this round
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🟢 Show Pro", key=f"btn_{pro_key}"):
                st.session_state[pro_key] = not st.session_state[pro_key]
        with col2:
            if st.button("🧑‍⚖️ Show Critic", key=f"btn_{critic_key}"):
                st.session_state[critic_key] = not st.session_state[critic_key]
        with col3:
            if st.button("🔴 Show Con", key=f"btn_{con_key}"):
                st.session_state[con_key] = not st.session_state[con_key]

        # Display sections for this round, based on button state
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.session_state[pro_key]:
                st.markdown(
                    f"""
                    <div style="background: ; border-radius: 10px; padding: 1rem; border: 2px solid #b7dfb0;">
                        <h4 style="color: #155724;">🟢 Pro Agent</h4>
                        <p style="color: ; font-size: 1.1em;">{round_data['pro_argument'].get('pro_argument', '')}</p>
                        <p style="color: ;"><b>Score:</b> {round_data.get('pro_score', '')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
        with cols[1]:
            if st.session_state[critic_key]:
                st.markdown(
                    f"""
                    <div style="background: ; border-radius: 10px; padding: 1rem; border: 2px solid #ffe49c;">
                        <h4 style="color: #856404;">🧑‍⚖️ Critic</h4>
                        <p style="color: ; font-size: 1.08em;">{round_data['critic_feedback']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
        with cols[2]:
            if st.session_state[con_key]:
                st.markdown(
                    f"""
                    <div style="background: ; border-radius: 10px; padding: 1rem; border: 2px solid #f5b7b1;">
                        <h4 style="color: #721c24;">🔴 Con Agent</h4>
                        <p style="color: ; font-size: 1.1em;">{round_data['con_argument'].get('con_argument', '')}</p>
                        <p style="color: ;"><b>Score:</b> {round_data.get('con_score', '')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )


if st.session_state.finished:
    if "debate_summary" not in st.session_state:
        st.session_state.debate_summary = None
    if st.button("🤖 Summarize Debate"):
        # Compose the debate transcript for the LLM
        transcript = ""
        for i, round_data in enumerate(st.session_state.rounds, 1):
            pro = round_data['pro_argument'].get('pro_argument', '')
            con = round_data['con_argument'].get('con_argument', '')
            critic = round_data.get('critic_feedback', '')
            verdict = round_data.get('verdict', '')
            transcript += (
                f"Round {i}:\n"
                f"Pro: {pro}\n"
                f"Con: {con}\n"
                f"Critic: {critic}\n"
                f"Winner: {verdict}\n\n"
            )
        # Call your LLM here (example with Claude, replace with your actual call)
        prompt = ChatPromptTemplate.from_template(
    "Summarize the following AI debate. For each round, provide a summary of the Pro argument, Con argument, and Critic feedback in a clear markdown table. "
    "At the end, show the final verdict from the Critic and explain in detail why it was chosen. "
    "Use this format:\n\n"
    "| Round | Pro Summary | Con Summary | Critic Summary |\n"
    "|-------|-------------|-------------|----------------|\n"
    "<Fill one row per round>\n\n"
    "Then write:\n"
    "**Final Verdict:** <winner>\n"
    "**Reason:** <reasoning for verdict>\n\n"
    "Debate Transcript:\n{transcript}"
)


        summary_model=ChatAnthropic(model='claude-3-5-haiku-20241022',temperature=0.5)


        summary_chain=prompt | summary_model

        msg=summary_chain.invoke({'transcript':transcript})

        # st.write(transcript)

        st.session_state.debate_summary  = msg.content if hasattr(msg.content[0], "text") else str(msg.content)

    if st.session_state.debate_summary :
        
        st.write(st.session_state.debate_summary )
        st.markdown("</div>", unsafe_allow_html=True)

# --- DEBUG PANEL ---
with st.expander("🔍 Debug: Current Node Output"):
    if st.session_state.current_state is not None:
        st.write(st.session_state.current_state)
        st.write("State keys:", list(st.session_state.current_state.keys()))
    else:
        st.write("No current state yet.")

