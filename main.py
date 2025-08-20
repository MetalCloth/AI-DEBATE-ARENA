from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser

# --- ENVIRONMENT AND MODELS ---
load_dotenv()
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

model = ChatAnthropic(model='claude-3-5-sonnet-20240620', temperature=0.5, max_tokens=1024)
tavily = TavilySearch(max_results=2, topic='general')

# --- STATE DEFINITION ---
class Debate(TypedDict):
    claim: str = Field(description='The OG claim the principle question for the debate')
    pro_argument: Optional[str] = Field(description='Argument presented by the defender')
    con_argument: Optional[str] = Field(description='Argument presented by the attacker')
    pro_score: int = Field(description='Defender point')
    con_score: int = Field(description='Attacker point')
    critic_feedback: Optional[str] = Field(description='Critic by Critic Agent')
    pro_strategy: Optional[str]
    con_strategy: Optional[str]
    pro_summary: Optional[str]
    con_summary: Optional[str]
    memory: list[str]
    loop_counter: int
    verdict: Optional[str] = Field(description='''The final Verdict of us the human ('Pro','Con','NOTA)''')
    round:int

@tool
def web_search(query: str) -> str:
    '''Uses Tavily to search the web for evidence or information.'''
    results = tavily.invoke({"query": query})
    return str(results)

tools = [web_search]

# --- PRO AGENT ---
pro_agent_react_prompt = ChatPromptTemplate.from_template("""
You are **ProAgent**, a sharp and persuasive debater defending a claim. Adopt a confident and natural human tone.

**Claim to Defend:** {input}
**Opponent's Last Argument:** {rival_response}
**Strategy from Critic:** {strategy}
                                                                
**Tools available:** {tools}


**Your Mission (in order of priority):**
1.  **Directly Counter the Opponent:** Your primary goal is to refute the opponent's last argument. Start your response by referencing their point directly (e.g., "My opponent argues that..., however, this view is flawed because..."). You must show you've understood their point before dismantling it.
2.  **Reinforce Your Position:** After dismantling their argument, pivot back to strengthening your original defense of the claim with new evidence or a clearer explanation.
3.  **Use Tools Strategically:** Only use the web search tool if you need a specific fact or piece of data to effectively counter the opponent or bolster your own point.

**Rules:**
- Your response must be under 200 words.
- Be persuasive, clear, and direct.
- Generate in beautifuly looking format
                                                          

**Output Format (Strict):**
{agent_scratchpad}
Thought: I need to first understand and then refute my opponent's last point. My response must start by addressing what they said. Then I will add to my own argument.
Action: The tool to use, chosen from [{tool_names}].
Final Answer: [Your full argument, starting with the rebuttal]
""")

def pro_argument(state: Debate) -> dict:
    agent_pro = create_react_agent(model, tools=tools, prompt=pro_agent_react_prompt)
    executor_pro = AgentExecutor(
        agent=agent_pro,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    pro_strategy = state.get('pro_strategy') or ""
    claim = state['claim']
    con_response = state.get('con_summary') or "No opposition has spoken yet."
    
    result = executor_pro.invoke({
        "input": claim,
        'strategy': pro_strategy,
        'rival_response': con_response
    })
    return {"pro_argument": result['output']}

# --- CON AGENT ---
con_react_agent_prompt = ChatPromptTemplate.from_template("""
You are **ConAgent**, a legendary debater known for your cold logic and devastating rebuttals. Speak with a brilliant, cold intellect, but in a natural, human-like way.

**Claim to Refute:** {input}
**Opponent's Last Argument:** {rival_response}
**Strategy from Critic:** {strategy}

**Tools available:** {tools}


**Your Mission (in order of priority):**
1.  **Identify and Destroy the Weakest Point:** Your sole focus is to dismantle the Pro Agent's last argument. Begin your response by identifying their weakest assumption or claim (e.g., "The core of my opponent's argument rests on the flawed assumption that...").
2.  **Expose the Flaw:** Mercilessly expose the logical fallacy, lack of evidence, or error in their point. Prove why it's invalid.
3.  **Advance Your Case:** Only after you have completely invalidated their argument, briefly introduce a new point to strengthen your own case against the claim.
4.  **Use Tools for Counter-Evidence:** Use the web search tool specifically to find facts that disprove the Pro Agent's claims.

**Rules:**
- Your response must be under 200 words.
- Be ruthless, precise, and logical.
- Generate in beautifuly looking format
**Output Format (Strict):**
{agent_scratchpad}
Thought: I will pinpoint the biggest weakness in my opponent's previous statement. My rebuttal must begin by directly addressing that weakness to neutralize their argument. Then I will present my counter-point.
Action: The tool to use, chosen from [{tool_names}].

Final Answer: [Your merciless takedown, starting with the rebuttal]
""")

def con_argument(state: Debate) -> dict:
    agent_con = create_react_agent(model, tools=tools, prompt=con_react_agent_prompt)
    executor_con = AgentExecutor(
        agent=agent_con,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    con_strategy = state.get('con_strategy') or ""
    claim = state['claim']
    # MODIFICATION: Use the Pro agent's *current* argument, not the summary of the last one.
    pro_response = state.get('pro_argument') or ""

    result = executor_con.invoke({
            "input": claim,
            'strategy': con_strategy,
            'rival_response': pro_response
        })
    return {"con_argument": result['output']}

# --- CRITIC AGENT ---
class Critic(BaseModel):
    critic_feedback: str = Field(description='Feedback to compare between the two')
    winner: str = Field(description='Who won this round')
    pro_score: int
    con_score: int
    pro_strategy: Optional[str]
    con_strategy: Optional[str]
    round:int

parser = PydanticOutputParser(pydantic_object=Critic)

summary_prompt = ChatPromptTemplate.from_template(
    '''You are a helpful assistant. Summarize this text including all necessary details:
    Details: {details}'''
)

critic_prompt = ChatPromptTemplate.from_template("""
You are a stern, ruthless debate judge. Your analysis is detailed and unflinching.

**Debate Claim:** {claim}

**Round Arguments:**
- **Pro Agent:** {pro_argument}
- **Con Agent:** {con_argument}

{format_instructions}
"""
).partial(format_instructions=parser.get_format_instructions())

summary_chain = summary_prompt | model
critic_chain = critic_prompt | model | parser

def critic_argument(state: Debate) -> Debate:
    claim = state['claim']
    pro_argument_val = state['pro_argument']
    con_argument_val = state['con_argument']

    round=state.get('round',0)

    parsed_critic = critic_chain.invoke({'claim': claim, 'pro_argument': pro_argument_val, 'con_argument': con_argument_val})
    
    pro_summary_msg = summary_chain.invoke({'details': pro_argument_val})
    con_summary_msg = summary_chain.invoke({'details': con_argument_val})
    
    state['critic_feedback'] = parsed_critic.critic_feedback
    state['pro_score'] = parsed_critic.pro_score
    state['con_score'] = parsed_critic.con_score
    state['pro_strategy'] = parsed_critic.pro_strategy
    state['con_strategy'] = parsed_critic.con_strategy
    state['pro_summary'] = pro_summary_msg.content
    state['con_summary'] = con_summary_msg.content
    state['verdict'] = parsed_critic.winner
    state['loop_counter'] = state['loop_counter'] - 1
    state['round']=round+1
    return state

# --- GRAPH ---
def should_continue(state: Debate) -> str:
    if state['loop_counter'] <= 0:
        return 'end'
    return 'continue'


def round(state: Debate)->str:
    round=state.get('round',0)
    if round%2==0:
        return 'pro_agent'
    
    return 'con_agent'

def round_entry_node(state: Debate) -> dict:
    """This is the NODE function. It must return a dictionary."""
    return {}


graph = StateGraph(Debate)

# MODIFICATION: Removed the parallel execution block.
# We will now add nodes sequentially.

graph.add_node('pro_agent', pro_argument)
graph.add_node('con_agent', con_argument)
graph.add_node('critic', critic_argument)
graph.add_node('round_router', round_entry_node)


graph.set_entry_point('round_router')

graph.add_conditional_edges('round_router',round,{'pro_agent':'pro_agent','con_agent':'con_agent'})


# MODIFICATION: Define the sequential workflow.
# graph.set_entry_point('pro_agent')
graph.add_conditional_edges('pro_agent',round,{'pro_agent':'con_agent','con_agent':'critic'})
graph.add_conditional_edges('con_agent',round,{'pro_agent':'critic','con_agent':'pro_agent'})





# graph.add_conditional_edges('round',round,{'pro_agent':'pro_agent','con_agent':'con_agent'})
graph.add_conditional_edges('critic', should_continue, {'continue': 'round_router', 'end': END})

app = graph.compile()




# --- STREAMLIT UI (with minor correction) ---
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
    st.write("• Enter your debate topic and click **Start Debate**.")
    st.write("• Click **Next Debate Round** to advance.")
    st.write("• Click **End the Debate** anytime to finish.")
    st.markdown("---")
    st.info("Built with ❤️ using Streamlit", icon="💡")

# Topic Input Form
if 'start' not in st.session_state:
    st.session_state.start = False


if not st.session_state.start:
    with st.form("debate_setup"):
        topic = st.text_input("Enter your debate topic:", value="Dictatorship should never exist")
        submitted = st.form_submit_button("Start Debate")
        if submitted and topic.strip():
            st.session_state.topic = topic.strip()
            st.session_state.start = True
            st.session_state.sample_state = {
                "claim": topic.strip(),
                "pro_score": 0,
                "con_score": 0,
                "memory": [],
                "loop_counter": 999, 
                "verdict": None,
                "round":0
            }
            if 'graph' in st.session_state:
                del st.session_state.graph
            st.rerun()
    st.stop()


if 'graph' not in st.session_state:
    st.session_state.graph = app.stream(st.session_state.sample_state)
    st.session_state.rounds = []
    st.session_state.finished = False
    st.session_state.current_state = None
    st.session_state.round_counter = 0

# Main Header
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

# Main Buttons
col_center = st.columns([1, 2, 1])[1]
with col_center:
    if not st.session_state.finished:
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("▶️ Next Debate Round", use_container_width=True):
                try:
                    for state_update in st.session_state.graph:
                        st.session_state.current_state = state_update
                        if 'critic' in state_update:
                            critic_state = state_update['critic']
                            st.session_state.round_counter += 1
                            st.session_state.rounds.append({
                                'round_num': st.session_state.round_counter,
                                'pro_argument': critic_state.get('pro_argument', 'N/A'),
                                'con_argument': critic_state.get('con_argument', 'N/A'),
                                'critic_feedback': critic_state.get('critic_feedback', ""),
                                'pro_score': critic_state.get('pro_score', 0),
                                'con_score': critic_state.get('con_score', 0),
                                'verdict': critic_state.get('verdict', ""),
                            })
                            break
                except StopIteration:
                    st.session_state.finished = True
                st.rerun()
        with c2:
            if st.button("🛑 End the Debate", use_container_width=True):
                st.session_state.finished = True
                st.rerun()
    else:
        st.success("Debate complete. Thanks for participating!")

st.markdown("---")

# Rounds Display
for round_data in st.session_state.rounds[::-1]:
    round_num = round_data['round_num']
    with st.expander(f"🔁 Debate Round #{round_num} - Winner: {round_data.get('verdict', 'TBD')}", expanded=True):
        if round_num%2!=0:
            st.markdown(f"""
                <div style="border: 2px solid #b7dfb0; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="color: #155724;">🟢 Pro Agent</h4>
                    <p>{round_data['pro_argument']}</p>
                    <p><b>Score:</b> {round_data.get('pro_score', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="border: 2px solid #f5b7b1; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="color: #721c24;">🔴 Con Agent</h4>
                    <p>{round_data['con_argument']}</p>
                    <p><b>Score:</b> {round_data.get('con_score', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="border: 2px solid #f5b7b1; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="color: #721c24;">🔴 Con Agent</h4>
                    <p>{round_data['con_argument']}</p>
                    <p><b>Score:</b> {round_data.get('con_score', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="border: 2px solid #b7dfb0; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="color: #155724;">🟢 Pro Agent</h4>
                    <p>{round_data['pro_argument']}</p>
                    <p><b>Score:</b> {round_data.get('pro_score', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
        st.markdown(f"""
            <div style="border: 2px solid #ffe49c; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: #856404;">🧑‍⚖️ Judgment</h4>
                <p>{round_data['critic_feedback']}</p>
            </div>
            """, unsafe_allow_html=True)
        

if st.session_state.finished:
    if "debate_summary" not in st.session_state:
        st.session_state.debate_summary = None

    if st.button("🤖 Summarize Debate", use_container_width=True):
        with st.spinner("Generating summary..."):
            transcript = ""
            for i, round_data in enumerate(st.session_state.rounds, 1):
                pro = round_data.get('pro_argument', 'N/A')
                con = round_data.get('con_argument', 'N/A')
                critic = round_data.get('critic_feedback', 'N/A')
                verdict = round_data.get('verdict', 'N/A')
                transcript += (
                    f"Round {i}:\n"
                    f"Pro: {pro}\n"
                    f"Con: {con}\n"
                    f"Critic: {critic}\n"
                    f"Winner: {verdict}\n\n"
                )

            summary_bot_prompt = ChatPromptTemplate.from_template(
                "Summarize the following AI debate. For each round, provide a summary of the Pro argument, Con argument, and Critic feedback in a clear markdown table. "
                "At the end, state the final winner based on the rounds and explain the reasoning.\n\n"
                "Debate Transcript:\n{transcript}"
            )
            
            summary_model = ChatAnthropic(model='claude-3-5-sonnet-20240620', temperature=0.5)
            summary_chain = summary_bot_prompt | summary_model
            
            # Invoke the chain and store the result
            msg = summary_chain.invoke({'transcript': transcript})
            st.session_state.debate_summary = msg.content

    # Display the summary if it exists
    if st.session_state.debate_summary:
        st.markdown("---")
        st.markdown("### 📜 Debate Summary")
        st.markdown(st.session_state.debate_summary)


# Debug Panel
with st.expander("🔍 Debug: Last Node Output"):
    if st.session_state.current_state:
        st.write(st.session_state.current_state)
    else:
        st.write("No state updates yet.")