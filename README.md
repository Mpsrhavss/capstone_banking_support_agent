# 🏦 Banking Customer Support AI Agent — Multi-Agent Architecture

**Applied Generative AI Specialisation — Capstone Project**

A multi-agent GenAI system for banking customer support built with
**LangGraph**, **Groq**, **SQLite**, and **Streamlit**.

| Agent | Responsibility |
|---|---|
| **Classifier Agent** | Categorises each message as *Positive Feedback*, *Negative Feedback*, or *Query* and routes it |
| **Feedback Handler** | Positive → personalised thank-you · Negative → unique 6-digit ticket inserted into `support_tickets` + empathetic reply |
| **Query Handler** | Extracts the ticket number, queries SQLite, and reports the status |

## Files

| File | Purpose |
|---|---|
| `banking_support_app.py` | Complete Streamlit application (agents + DB + evaluation + UI) |
| `Banking_Support_Agent.ipynb` | Development notebook with step-by-step explanations and executed outputs |
| `requirements.txt` | Python dependencies |
| `Capstone_Report.docx` | Detailed project report |
| `architecture_diagram.png` | System architecture diagram |
| `screenshots/` | UI screenshots of every dashboard tab |

`support_tickets.db` (SQLite) is **created and seeded automatically** on first
run — no external dataset is required.

## How to run

```bash
pip install -r requirements.txt
streamlit run banking_support_app.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Groq API key (configurable)

The LLM is optional but recommended:

1. Create a free key at https://console.groq.com
2. Paste it into the **sidebar → Groq API Key** field, *or* set it before
   launching:

```bash
export GROQ_API_KEY="gsk_..."
```

Without a key the app automatically switches to a deterministic **rule-based
fallback engine**, so every workflow (classification, ticketing, status
queries, evaluation) still works end-to-end.

## Evaluation (LLMOps)

The **📊 Evaluation** tab runs a labelled 30-case test suite and reports
classification accuracy, agent routing success rate, QA pass rate, and
latency — plus optional LLM-as-judge scoring (empathy / clarity / accuracy)
when an API key is configured. The **🧾 Logs & Debug** tab shows interaction
logs and per-message prompt traces.
