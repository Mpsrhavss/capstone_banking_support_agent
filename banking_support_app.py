"""
Banking Customer Support AI Agent using Multi-Agent Architecture
================================================================
Applied Generative AI Specialisation — Capstone Project

A multi-agent GenAI system for banking customer support built with LangGraph.

Agents:
    1. Classifier Agent        — categorises a message as Positive Feedback,
                                 Negative Feedback, or Query, and routes it.
    2. Feedback Handler Agent  — thanks customers for positive feedback;
                                 opens a support ticket for negative feedback.
    3. Query Handler Agent     — extracts a ticket number and reports its status.

LLM   : Groq (configurable API key — sidebar or GROQ_API_KEY env variable).
        A deterministic rule-based fallback keeps every workflow functional
        when no API key is configured.
DB    : SQLite (support_tickets + interaction_logs tables, auto-created).
UI    : Streamlit — run with:  streamlit run banking_support_app.py
"""

import json
import os
import random
import re
import sqlite3
import time
from datetime import datetime
from typing import List, Optional, TypedDict

import pandas as pd
import streamlit as st
from langgraph.graph import END, StateGraph

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    GROQ_SDK_AVAILABLE = True
except ImportError:  # keeps the app usable even without the Groq SDK
    GROQ_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_tickets.db")

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

# Runtime configuration shared with the LangGraph nodes for the current run.
RUNTIME = {"api_key": "", "model": AVAILABLE_MODELS[0]}

CATEGORIES = ["Positive Feedback", "Negative Feedback", "Query"]

TICKET_STATUSES = ["Unresolved", "In Progress", "Resolved"]

SEED_TICKETS = [
    (650932, "Priya Sharma", "Debit card replacement not received", "Resolved"),
    (784521, "Rahul Verma", "Net banking login failure", "In Progress"),
    (213377, "Anita Desai", "Unexpected charge on savings account", "Unresolved"),
    (445210, "David Mathew", "Credit card statement discrepancy", "Resolved"),
    (908764, "Sneha Iyer", "UPI transaction stuck in pending state", "In Progress"),
    (562148, "Arjun Nair", "Cheque book request not processed", "Unresolved"),
]


# ---------------------------------------------------------------------------
# Database layer  (support_tickets + interaction_logs)
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    """Create tables and seed sample tickets on first run."""
    with get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS support_tickets (
                   ticket_id         INTEGER PRIMARY KEY,
                   customer_name     TEXT,
                   issue_description TEXT,
                   status            TEXT DEFAULT 'Unresolved',
                   created_at        TEXT,
                   updated_at        TEXT
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS interaction_logs (
                   log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp      TEXT,
                   user_input     TEXT,
                   classification TEXT,
                   agent_path     TEXT,
                   response       TEXT,
                   ticket_action  TEXT,
                   latency_ms     REAL,
                   llm_used       INTEGER,
                   success        INTEGER,
                   prompt_trace   TEXT
               )"""
        )
        if conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0] == 0:
            now = datetime.now().isoformat(timespec="seconds")
            conn.executemany(
                "INSERT INTO support_tickets VALUES (?,?,?,?,?,?)",
                [(tid, name, issue, status, now, now)
                 for tid, name, issue, status in SEED_TICKETS],
            )


def generate_ticket_number(conn: sqlite3.Connection) -> int:
    """Generate a unique 6-digit ticket number."""
    while True:
        candidate = random.randint(100000, 999999)
        row = conn.execute(
            "SELECT 1 FROM support_tickets WHERE ticket_id = ?", (candidate,)
        ).fetchone()
        if row is None:
            return candidate


def create_ticket(customer_name: str, issue_description: str) -> int:
    """Insert a new unresolved ticket and return its 6-digit number."""
    with get_conn() as conn:
        ticket_id = generate_ticket_number(conn)
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO support_tickets VALUES (?,?,?,?,?,?)",
            (ticket_id, customer_name, issue_description, "Unresolved", now, now),
        )
    return ticket_id


def get_ticket_status(ticket_id: int) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    return row[0] if row else None


def all_tickets_df() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT ticket_id AS 'Ticket #', customer_name AS 'Customer', "
            "issue_description AS 'Issue', status AS 'Status', "
            "created_at AS 'Created', updated_at AS 'Updated' "
            "FROM support_tickets ORDER BY created_at DESC",
            conn,
        )


def log_interaction(state: dict, latency_ms: float, success: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO interaction_logs (timestamp, user_input, classification, "
            "agent_path, response, ticket_action, latency_ms, llm_used, success, "
            "prompt_trace) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                state.get("message", ""),
                state.get("category", ""),
                " → ".join(state.get("agent_path", [])),
                state.get("response", ""),
                state.get("ticket_action", "None"),
                round(latency_ms, 1),
                int(state.get("llm_used", False)),
                int(success),
                json.dumps(state.get("prompt_trace", [])),
            ),
        )


def logs_df() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT timestamp AS 'Time', user_input AS 'User Input', "
            "classification AS 'Classification', agent_path AS 'Agent Path', "
            "response AS 'Response', ticket_action AS 'Ticket Action', "
            "latency_ms AS 'Latency (ms)', llm_used AS 'LLM Used', "
            "success AS 'Success', prompt_trace "
            "FROM interaction_logs ORDER BY log_id DESC",
            conn,
        )


# ---------------------------------------------------------------------------
# LLM helper  (Groq — configurable, with graceful fallback)
# ---------------------------------------------------------------------------

def get_llm():
    """Return a ChatGroq instance when a key is configured, else None."""
    api_key = RUNTIME.get("api_key") or os.environ.get("GROQ_API_KEY", "")
    if not (GROQ_SDK_AVAILABLE and api_key):
        return None
    return ChatGroq(
        groq_api_key=api_key,
        model_name=RUNTIME.get("model", AVAILABLE_MODELS[0]),
        temperature=0.3,
        max_tokens=200,
    )


def llm_call(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Single LLM call; returns None on any failure so callers can fall back."""
    llm = get_llm()
    if llm is None:
        return None
    try:
        reply = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return reply.content.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    message: str            # raw user message
    customer_name: str      # provided or extracted customer name
    category: str           # classifier output
    response: str           # final response to the user
    ticket_id: Optional[int]
    ticket_action: str      # e.g. "Created ticket #123456" / "Looked up #650932"
    agent_path: List[str]   # ordered list of agents that handled the message
    prompt_trace: List[dict]
    llm_used: bool


def extract_customer_name(message: str, provided: str = "") -> str:
    """Use the provided name, else try to extract one from the message."""
    if provided and provided.strip():
        return provided.strip()
    match = re.search(
        r"(?:my name is|this is|i am|i'm)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)",
        message, flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().title()
    return "Valued Customer"


def extract_ticket_number(message: str) -> Optional[int]:
    """Extract a 6-digit ticket number from free text."""
    match = re.search(r"#?\b(\d{6})\b", message)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Agent 1 — Classifier
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a message classifier for a bank's customer support system.\n"
    "Classify the user's message into exactly one category:\n"
    "  POSITIVE_FEEDBACK — the customer is thanking us or praising the service.\n"
    "  NEGATIVE_FEEDBACK — the customer is complaining or reporting a problem "
    "and is NOT asking about an existing ticket.\n"
    "  QUERY — the customer asks about the status of an existing support "
    "ticket, or asks a service question (usually mentions a ticket number).\n"
    "Respond with ONLY the category label, nothing else."
)

POSITIVE_WORDS = [
    "thank", "thanks", "great", "excellent", "awesome", "wonderful", "amazing",
    "appreciate", "happy", "delighted", "resolved my", "sorted", "helpful",
    "quick", "love", "good service", "well done", "kudos", "fantastic", "best",
]
NEGATIVE_WORDS = [
    "not ", "n't", "no ", "issue", "problem", "complaint", "poor", "bad",
    "delay", "delayed", "failed", "failure", "error", "wrong", "unhappy",
    "frustrated", "disappointed", "waiting", "stuck", "pending", "unable",
    "charged", "deducted", "blocked", "worst", "terrible", "never", "missing",
]
QUERY_WORDS = [
    "status", "update on", "check", "track", "progress", "any news",
    "what happened to", "where is", "when will", "follow up", "followup",
]


def fallback_classify(message: str) -> str:
    """Deterministic keyword classifier used when the LLM is unavailable."""
    text = message.lower()
    has_ticket_ref = extract_ticket_number(message) is not None or "ticket" in text
    query_score = sum(1 for w in QUERY_WORDS if w in text)
    if has_ticket_ref and (query_score > 0 or "?" in text):
        return "Query"
    positive_score = sum(1 for w in POSITIVE_WORDS if w in text)
    negative_score = sum(1 for w in NEGATIVE_WORDS if w in text)
    if positive_score > negative_score:
        return "Positive Feedback"
    if negative_score > 0:
        return "Negative Feedback"
    if query_score > 0 or "?" in text:
        return "Query"
    return "Negative Feedback"  # safest default: a human follows up on a ticket


def classifier_agent(state: AgentState) -> AgentState:
    """Agent 1 — classify the message and record the routing decision."""
    message = state["message"]
    label_map = {
        "POSITIVE_FEEDBACK": "Positive Feedback",
        "NEGATIVE_FEEDBACK": "Negative Feedback",
        "QUERY": "Query",
    }
    raw = llm_call(CLASSIFIER_SYSTEM_PROMPT, message)
    category = None
    if raw is not None:
        cleaned = raw.upper().replace(" ", "_").strip()
        for key, value in label_map.items():
            if key in cleaned:
                category = value
                break
    llm_used = category is not None
    if category is None:
        category = fallback_classify(message)
    state["category"] = category
    state["llm_used"] = state.get("llm_used", False) or llm_used
    state.setdefault("agent_path", []).append("Classifier Agent")
    state.setdefault("prompt_trace", []).append(
        {
            "agent": "Classifier Agent",
            "engine": "LLM" if llm_used else "Rule-based fallback",
            "prompt": CLASSIFIER_SYSTEM_PROMPT + "\n\nUSER: " + message,
            "output": raw if raw is not None else category,
            "decision": category,
        }
    )
    return state


def route_by_category(state: AgentState) -> str:
    """Conditional edge — route to the downstream agent for the category."""
    return {
        "Positive Feedback": "positive_feedback_handler",
        "Negative Feedback": "negative_feedback_handler",
        "Query": "query_handler",
    }[state["category"]]


# ---------------------------------------------------------------------------
# Agent 2 — Feedback Handler (positive + negative branches)
# ---------------------------------------------------------------------------

def positive_feedback_agent(state: AgentState) -> AgentState:
    """Generate a warm, personalised thank-you message."""
    name = state["customer_name"]
    system = (
        "You are a warm, professional bank customer-support agent. The customer "
        "has left positive feedback. Write a SHORT (1-2 sentence) personalised "
        f"thank-you addressed to {name}. Mention what they thanked us for if "
        "clear from their message. No placeholders, no sign-off."
    )
    reply = llm_call(system, state["message"])
    llm_used = reply is not None
    if reply is None:
        reply = (
            f"Thank you for your kind words, {name}! "
            "We're delighted to assist you."
        )
    state["response"] = reply
    state["ticket_action"] = "None"
    state["llm_used"] = state.get("llm_used", False) or llm_used
    state.setdefault("agent_path", []).append("Positive Feedback Handler")
    state.setdefault("prompt_trace", []).append(
        {
            "agent": "Positive Feedback Handler",
            "engine": "LLM" if llm_used else "Template fallback",
            "prompt": system + "\n\nUSER: " + state["message"],
            "output": reply,
            "decision": "Thank-you message generated",
        }
    )
    return state


def negative_feedback_agent(state: AgentState) -> AgentState:
    """Open a support ticket and reply with an empathetic message."""
    name = state["customer_name"]
    ticket_id = create_ticket(name, state["message"])
    state["ticket_id"] = ticket_id
    state["ticket_action"] = f"Created ticket #{ticket_id} (Unresolved)"
    system = (
        "You are an empathetic bank customer-support agent. The customer has a "
        "complaint. Apologise sincerely in 1-2 sentences and tell them that "
        f"ticket #{ticket_id} has been created and our team will follow up "
        f"shortly. You MUST include the exact ticket number #{ticket_id}."
    )
    reply = llm_call(system, state["message"])
    llm_used = reply is not None
    if reply is None or str(ticket_id) not in reply:
        reply = (
            f"We apologize for the inconvenience, {name}. A new ticket "
            f"#{ticket_id} has been generated, and our team will follow up shortly."
        )
    state["response"] = reply
    state["llm_used"] = state.get("llm_used", False) or llm_used
    state.setdefault("agent_path", []).append("Negative Feedback Handler")
    state.setdefault("prompt_trace", []).append(
        {
            "agent": "Negative Feedback Handler",
            "engine": "LLM" if llm_used else "Template fallback",
            "prompt": system + "\n\nUSER: " + state["message"],
            "output": reply,
            "decision": state["ticket_action"],
        }
    )
    return state


# ---------------------------------------------------------------------------
# Agent 3 — Query Handler
# ---------------------------------------------------------------------------

def query_handler_agent(state: AgentState) -> AgentState:
    """Extract the ticket number, look it up, and report its status."""
    ticket_id = extract_ticket_number(state["message"])
    if ticket_id is None:
        state["response"] = (
            "I'd be happy to check that for you. Could you please share your "
            "6-digit ticket number?"
        )
        state["ticket_action"] = "No ticket number found in message"
    else:
        status = get_ticket_status(ticket_id)
        state["ticket_id"] = ticket_id
        if status is None:
            state["response"] = (
                f"I'm sorry, ticket #{ticket_id} was not found in our system. "
                "Please verify the number and try again."
            )
            state["ticket_action"] = f"Lookup failed — ticket #{ticket_id} not found"
        else:
            state["response"] = (
                f"Your ticket #{ticket_id} is currently marked as: {status}."
            )
            state["ticket_action"] = f"Looked up ticket #{ticket_id} → {status}"
    state.setdefault("agent_path", []).append("Query Handler")
    state.setdefault("prompt_trace", []).append(
        {
            "agent": "Query Handler",
            "engine": "Deterministic (regex + SQL)",
            "prompt": f"Extract ticket number from: {state['message']}",
            "output": state["response"],
            "decision": state["ticket_action"],
        }
    )
    return state


# ---------------------------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------------------------

def build_graph():
    """Wire the agents into a LangGraph StateGraph and compile it."""
    graph = StateGraph(AgentState)
    graph.add_node("classifier", classifier_agent)
    graph.add_node("positive_feedback_handler", positive_feedback_agent)
    graph.add_node("negative_feedback_handler", negative_feedback_agent)
    graph.add_node("query_handler", query_handler_agent)
    graph.set_entry_point("classifier")
    graph.add_conditional_edges(
        "classifier",
        route_by_category,
        {
            "positive_feedback_handler": "positive_feedback_handler",
            "negative_feedback_handler": "negative_feedback_handler",
            "query_handler": "query_handler",
        },
    )
    graph.add_edge("positive_feedback_handler", END)
    graph.add_edge("negative_feedback_handler", END)
    graph.add_edge("query_handler", END)
    return graph.compile()


WORKFLOW = build_graph()


def run_pipeline(message: str, customer_name: str = "",
                 api_key: str = "", model: str = "") -> dict:
    """Run one message through the multi-agent workflow and log the result."""
    RUNTIME["api_key"] = api_key
    if model:
        RUNTIME["model"] = model
    initial: AgentState = {
        "message": message,
        "customer_name": extract_customer_name(message, customer_name),
        "agent_path": [],
        "prompt_trace": [],
        "llm_used": False,
    }
    start = time.time()
    success = True
    try:
        final = WORKFLOW.invoke(initial)
    except Exception as exc:  # never crash the UI — log and report the failure
        final = dict(initial)
        final["category"] = final.get("category", "Error")
        final["response"] = f"Sorry, something went wrong while processing: {exc}"
        final["ticket_action"] = "Pipeline error"
        success = False
    latency_ms = (time.time() - start) * 1000
    final["latency_ms"] = latency_ms
    final["success"] = success
    log_interaction(final, latency_ms, success)
    return final


# ---------------------------------------------------------------------------
# Evaluation module  (test coverage, routing success, QA scoring)
# ---------------------------------------------------------------------------

EXPECTED_AGENT = {
    "Positive Feedback": "Positive Feedback Handler",
    "Negative Feedback": "Negative Feedback Handler",
    "Query": "Query Handler",
}

TEST_CASES = [
    # --- Positive feedback (10) ---
    ("Thanks for sorting out my net banking login issue.", "Positive Feedback"),
    ("Thank you for resolving my credit card issue so quickly!", "Positive Feedback"),
    ("Great service at the branch today, really appreciate it.", "Positive Feedback"),
    ("My name is Ramesh and I want to say the staff was excellent.", "Positive Feedback"),
    ("Awesome! My loan got approved in just two days. Thanks a lot.", "Positive Feedback"),
    ("I appreciate the quick help with my UPI problem yesterday.", "Positive Feedback"),
    ("The mobile app update is wonderful, well done team.", "Positive Feedback"),
    ("Kudos to your support team for the fantastic assistance.", "Positive Feedback"),
    ("I'm delighted with how fast my chequebook arrived. Thank you!", "Positive Feedback"),
    ("Best banking experience I've had, keep up the good service.", "Positive Feedback"),
    # --- Negative feedback (10) ---
    ("My debit card replacement still hasn't arrived.", "Negative Feedback"),
    ("I was charged twice for the same transaction, please fix this.", "Negative Feedback"),
    ("Net banking has been down for three days, this is frustrating.", "Negative Feedback"),
    ("This is Anjali. My cheque book request was never processed.", "Negative Feedback"),
    ("Very poor service — my complaint about the ATM was ignored.", "Negative Feedback"),
    ("My UPI payment failed but the money was deducted from my account.", "Negative Feedback"),
    ("I am unhappy with the delay in my loan disbursement.", "Negative Feedback"),
    ("The branch staff was rude and my issue is still unresolved.", "Negative Feedback"),
    ("My account got blocked without any notice, this is terrible.", "Negative Feedback"),
    ("I have been waiting two weeks for my fixed deposit receipt.", "Negative Feedback"),
    # --- Queries (10) ---
    ("Could you check the status of ticket 650932?", "Query"),
    ("What is the update on my ticket #784521?", "Query"),
    ("Please tell me the progress of ticket number 213377.", "Query"),
    ("Any news on ticket 445210?", "Query"),
    ("I want to track my complaint, ticket #908764.", "Query"),
    ("When will ticket 562148 be resolved? Please check the status.", "Query"),
    ("Can you give me a status update on ticket #650932?", "Query"),
    ("What happened to my ticket 784521? Please check.", "Query"),
    ("Status of my support ticket 213377 please.", "Query"),
    ("Follow up on ticket #445210 — what is the current status?", "Query"),
]


def qa_check_response(category: str, response: str, state: dict) -> bool:
    """Rule-based QA: does the response satisfy the required format/content?"""
    text = response.lower()
    if category == "Positive Feedback":
        return any(w in text for w in ["thank", "delighted", "appreciate", "glad", "happy"])
    if category == "Negative Feedback":
        has_ticket = re.search(r"#\d{6}", response) is not None
        has_empathy = any(w in text for w in ["apolog", "sorry", "regret", "inconvenience"])
        return has_ticket and has_empathy
    if category == "Query":
        return ("marked as" in text and re.search(r"#\d{6}", response) is not None) \
            or "not found" in text or "ticket number" in text
    return False


def run_evaluation(api_key: str = "", model: str = "") -> tuple:
    """Run every test case through the pipeline; return (DataFrame, metrics)."""
    rows = []
    for message, expected in TEST_CASES:
        result = run_pipeline(message, api_key=api_key, model=model)
        predicted = result.get("category", "Error")
        routed_to = result["agent_path"][-1] if result.get("agent_path") else "None"
        routing_ok = routed_to == EXPECTED_AGENT.get(predicted, "")
        qa_ok = qa_check_response(predicted, result.get("response", ""), result)
        rows.append(
            {
                "Test Message": message,
                "Expected": expected,
                "Predicted": predicted,
                "Correct": predicted == expected,
                "Routed To": routed_to,
                "Routing OK": routing_ok,
                "QA Pass": qa_ok,
                "Engine": "LLM" if result.get("llm_used") else "Fallback",
                "Latency (ms)": round(result.get("latency_ms", 0), 1),
                "Response": result.get("response", ""),
            }
        )
    df = pd.DataFrame(rows)
    metrics = {
        "Classification Accuracy": df["Correct"].mean(),
        "Routing Success Rate": df["Routing OK"].mean(),
        "QA Pass Rate": df["QA Pass"].mean(),
        "Avg Latency (ms)": df["Latency (ms)"].mean(),
    }
    for cat in CATEGORIES:
        subset = df[df["Expected"] == cat]
        if len(subset):
            metrics[f"Accuracy — {cat}"] = subset["Correct"].mean()
    return df, metrics


JUDGE_PROMPT = (
    "You are a strict QA reviewer for a bank's support chatbot. Score the "
    "agent response on three dimensions from 1 (poor) to 5 (excellent):\n"
    "empathy, clarity, accuracy.\n"
    'Reply ONLY with JSON like {"empathy": 4, "clarity": 5, "accuracy": 5}.'
)


def llm_judge_scores(sample_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """LLM-as-judge QA scoring on evaluated responses (needs an API key)."""
    if get_llm() is None:
        return None
    scored = []
    for _, row in sample_df.iterrows():
        raw = llm_call(
            JUDGE_PROMPT,
            f"Customer message: {row['Test Message']}\n"
            f"Agent response: {row['Response']}",
        )
        if raw is None:
            continue
        try:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            scored.append(
                {
                    "Test Message": row["Test Message"],
                    "Empathy": data.get("empathy"),
                    "Clarity": data.get("clarity"),
                    "Accuracy": data.get("accuracy"),
                }
            )
        except (json.JSONDecodeError, AttributeError):
            continue
    return pd.DataFrame(scored) if scored else None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

SAMPLE_SCENARIOS = {
    "Positive Feedback → Thank-you": "Thanks for sorting out my net banking login issue.",
    "Negative Feedback → New ticket": "My debit card replacement still hasn't arrived.",
    "Query → Ticket status": "Could you check the status of ticket 650932?",
}


def render_agent_path(path: List[str]) -> str:
    return "  ➜  ".join(f"**{p}**" for p in path) if path else "—"


def main() -> None:
    st.set_page_config(
        page_title="Banking Support AI Agent",
        page_icon="🏦",
        layout="wide",
    )
    init_db()

    # ---------------- Sidebar : configuration ----------------
    with st.sidebar:
        st.title("🏦 Banking Support AI")
        st.caption("Multi-Agent Architecture · LangGraph + Groq")
        st.divider()
        st.subheader("⚙️ LLM Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Paste your key from console.groq.com. "
                 "Leave blank to use the rule-based fallback engine.",
        )
        model = st.selectbox("Groq Model", AVAILABLE_MODELS, index=0)
        RUNTIME["api_key"] = api_key
        RUNTIME["model"] = model
        if api_key:
            st.success("LLM mode: **Groq** ✅")
        else:
            st.warning("Fallback mode: **rule-based** (no API key)")
        st.divider()
        customer_name = st.text_input(
            "Customer Name (optional)",
            help="Used to personalise responses; auto-extracted from the "
                 "message when possible.",
        )
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
        st.divider()
        st.caption(
            "Agents: Classifier → Feedback Handler / Query Handler\n\n"
            "DB: SQLite · support_tickets"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tab_chat, tab_db, tab_eval, tab_logs, tab_scenarios = st.tabs(
        ["💬 Assistant", "🎫 Ticket Database", "📊 Evaluation",
         "🧾 Logs & Debug", "🧪 Test Scenarios"]
    )

    # ---------------- Tab 1 : Assistant ----------------
    with tab_chat:
        st.subheader("Customer Support Assistant")
        st.caption(
            "Type a message — the Classifier Agent routes it to the right "
            "handler and the full agent path is shown with every response."
        )
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["user"])
            with st.chat_message("assistant"):
                st.write(entry["response"])
                meta = (
                    f"`{entry['category']}` · {entry['path']} · "
                    f"{entry['latency']:.0f} ms · {entry['engine']}"
                )
                st.caption(meta)
                if entry.get("ticket_action") not in (None, "", "None"):
                    st.info(f"🗄️ Database interaction: {entry['ticket_action']}")

        user_message = st.chat_input("Type your message to the bank…")
        if user_message:
            with st.spinner("Agents at work…"):
                result = run_pipeline(
                    user_message, customer_name=customer_name,
                    api_key=api_key, model=model,
                )
            st.session_state.chat_history.append(
                {
                    "user": user_message,
                    "response": result.get("response", ""),
                    "category": result.get("category", "?"),
                    "path": " ➜ ".join(result.get("agent_path", [])),
                    "latency": result.get("latency_ms", 0),
                    "engine": "LLM" if result.get("llm_used") else "Rule-based",
                    "ticket_action": result.get("ticket_action", "None"),
                }
            )
            st.rerun()

    # ---------------- Tab 2 : Ticket Database ----------------
    with tab_db:
        st.subheader("Support Tickets (SQLite · table: support_tickets)")
        tickets = all_tickets_df()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", len(tickets))
        col2.metric("Unresolved", int((tickets["Status"] == "Unresolved").sum()))
        col3.metric("In Progress", int((tickets["Status"] == "In Progress").sum()))
        col4.metric("Resolved", int((tickets["Status"] == "Resolved").sum()))
        status_filter = st.multiselect(
            "Filter by status", TICKET_STATUSES, default=TICKET_STATUSES
        )
        st.dataframe(
            tickets[tickets["Status"].isin(status_filter)],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Negative feedback automatically inserts a new Unresolved ticket "
            "with a unique 6-digit number."
        )

    # ---------------- Tab 3 : Evaluation ----------------
    with tab_eval:
        st.subheader("Model Evaluation (LLMOps)")
        st.caption(
            f"{len(TEST_CASES)} labelled test cases — classification accuracy, "
            "agent routing success rate, and QA response checks."
        )
        if st.button("▶️ Run evaluation suite", type="primary"):
            with st.spinner("Running all test cases through the workflow…"):
                df, metrics = run_evaluation(api_key=api_key, model=model)
            st.session_state.eval_df = df
            st.session_state.eval_metrics = metrics
        if "eval_df" in st.session_state:
            metrics = st.session_state.eval_metrics
            df = st.session_state.eval_df
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Classification Accuracy",
                      f"{metrics['Classification Accuracy']:.0%}")
            m2.metric("Routing Success", f"{metrics['Routing Success Rate']:.0%}")
            m3.metric("QA Pass Rate", f"{metrics['QA Pass Rate']:.0%}")
            m4.metric("Avg Latency", f"{metrics['Avg Latency (ms)']:.0f} ms")
            per_class = {
                cat: metrics.get(f"Accuracy — {cat}", 0) for cat in CATEGORIES
            }
            st.bar_chart(pd.DataFrame(
                {"Accuracy": per_class.values()}, index=per_class.keys()
            ))
            st.dataframe(
                df.drop(columns=["Response"]),
                use_container_width=True, hide_index=True,
            )
            with st.expander("Show generated responses"):
                for _, row in df.iterrows():
                    st.markdown(f"**{row['Test Message']}**")
                    st.write(row["Response"])
                    st.divider()
            if RUNTIME.get("api_key"):
                if st.button("🧑‍⚖️ LLM-as-judge QA scoring (sample)"):
                    with st.spinner("Judging response quality…"):
                        judged = llm_judge_scores(df.head(6))
                    if judged is not None and len(judged):
                        st.dataframe(judged, use_container_width=True,
                                     hide_index=True)
                        st.metric(
                            "Mean judge score (1-5)",
                            f"{judged[['Empathy', 'Clarity', 'Accuracy']].mean().mean():.2f}",
                        )
                    else:
                        st.warning("Judge scoring failed — check the API key.")
            else:
                st.info(
                    "Add a Groq API key in the sidebar to enable "
                    "LLM-as-judge empathy/clarity/accuracy scoring."
                )

    # ---------------- Tab 4 : Logs & Debug ----------------
    with tab_logs:
        st.subheader("Interaction Logs & Debugging")
        logs = logs_df()
        if len(logs) == 0:
            st.info("No interactions logged yet — try the Assistant tab.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Interactions", len(logs))
            c2.metric("Success Rate", f"{logs['Success'].mean():.0%}")
            c3.metric("LLM-powered", f"{logs['LLM Used'].mean():.0%}")
            c4.metric("Avg Latency", f"{logs['Latency (ms)'].mean():.0f} ms")
            st.dataframe(
                logs.drop(columns=["prompt_trace"]),
                use_container_width=True, hide_index=True,
            )
            st.markdown("#### 🔍 Prompt trace inspector")
            options = [
                f"#{i} — {row['User Input'][:60]}"
                for i, (_, row) in enumerate(logs.iterrows())
            ]
            selected = st.selectbox("Inspect interaction", options)
            if selected:
                idx = int(selected.split(" — ")[0][1:])
                trace = json.loads(logs.iloc[idx]["prompt_trace"] or "[]")
                for step in trace:
                    with st.expander(
                        f"{step.get('agent')} · {step.get('engine')}"
                    ):
                        st.markdown("**Prompt**")
                        st.code(step.get("prompt", ""), language="text")
                        st.markdown("**Output**")
                        st.write(step.get("output", ""))
                        st.markdown(f"**Decision:** {step.get('decision', '')}")

    # ---------------- Tab 5 : Test Scenarios ----------------
    with tab_scenarios:
        st.subheader("Test Scenarios — one per agent role")
        st.caption(
            "The three canonical flows from the problem statement. Click to "
            "run a scenario end-to-end through the multi-agent workflow."
        )
        for label, message in SAMPLE_SCENARIOS.items():
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.code(message, language="text")
                if st.button(f"Run scenario", key=f"scenario_{label}"):
                    with st.spinner("Running…"):
                        result = run_pipeline(
                            message, customer_name=customer_name,
                            api_key=api_key, model=model,
                        )
                    st.markdown(
                        "**Agent path:** "
                        + render_agent_path(result.get("agent_path", []))
                    )
                    st.success(result.get("response", ""))
                    if result.get("ticket_action") not in (None, "", "None"):
                        st.info(
                            f"🗄️ Database interaction: {result['ticket_action']}"
                        )


if __name__ == "__main__":
    main()
