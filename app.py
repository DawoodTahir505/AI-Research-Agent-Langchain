import os
import time
import streamlit as st
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

# ------------------------------------------------------------
# 1. PAGE SETTINGS (must be the first Streamlit command)
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="centered",
)

# ------------------------------------------------------------
# 2. API KEY
# ------------------------------------------------------------
load_dotenv()

try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass


# ------------------------------------------------------------
# 3. AGENT (built once, then reused)
# ------------------------------------------------------------
@st.cache_resource
def get_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    @tool
    def web_search(query: str) -> str:
        """Search the web. Returns titles, snippets and links."""
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "SEARCH_ERROR: no results found"
            return "\n\n".join(
                f"{r['title']}\n{r['body']}\nURL: {r['href']}" for r in results
            )
        except Exception as e:
            return f"SEARCH_ERROR: {e}"

    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a research assistant. Always call web_search before answering. "
            "Answer concisely. End with a 'Sources' list of markdown links. "
            "Use only URLs returned by web_search. Never invent URLs. "
            "If web_search returns SEARCH_ERROR, report the error "
            "and do not answer from memory."
        ),
    )


def ask_agent(messages, retries=3):
    """Call the agent with exponential backoff."""
    for attempt in range(retries):
        try:
            return get_agent().invoke({"messages": messages})
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s


# ------------------------------------------------------------
# 4. SESSION STATE
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer), newest first

if "question" not in st.session_state:
    st.session_state.question = ""


def set_question(text):
    st.session_state.question = text


def clear_history():
    st.session_state.history = []


# ------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app searches the web and answers your question using AI.")

    st.subheader("How to use")
    st.markdown(
        """
1. Type a question or pick an example  
2. Click **Search**  
3. Read the answer below
"""
    )

    st.button("🗑️ Clear history", on_click=clear_history, use_container_width=True)
    st.caption("Built with Streamlit, LangChain and Groq")

# ------------------------------------------------------------
# 6. MAIN PAGE
# ------------------------------------------------------------
st.title("🔎 AI Research Agent")
st.caption("Ask any question and get a researched answer.")

st.write("**Try an example:**")
examples = [
    "What is data science?",
    "Latest AI news today",
    "How does solar energy work?",
]
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    col.button(
        example,
        on_click=set_question,
        args=(example,),
        use_container_width=True,
    )

question = st.text_input(
    "Your question",
    key="question",
    placeholder="Type your question here...",
)

search_clicked = st.button("🔍 Search", type="primary")

# ------------------------------------------------------------
# 7. RUN THE AGENT
# ------------------------------------------------------------
if search_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Researching, please wait..."):
            try:
                # last 5 turns, oldest first
                messages = []
                for q, a in reversed(st.session_state.history[:5]):
                    messages += [("user", q), ("assistant", a)]
                messages.append(("user", question))

                result = ask_agent(messages)
                answer = result["messages"][-1].content
                st.session_state.history.insert(0, (question, answer))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ------------------------------------------------------------
# 8. SHOW ANSWERS (newest first)
# ------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("Answers")
    for i, (q, a) in enumerate(st.session_state.history):
        with st.expander(f"Q: {q}", expanded=(i == 0)):
            st.write(a)
else:
    st.info("No questions yet. Ask something to get started.")
