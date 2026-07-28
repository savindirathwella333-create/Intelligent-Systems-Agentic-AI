# SME Business-Ops Copilot

An agentic AI assistant for a small Sri Lankan manufacturing business, answering questions about **inventory & production** and **HR & compliance (EPF/ETF)** by retrieving grounded answers from the company's own policy documents.

**Live demo:** 
**GitHub repo:** https://github.com/savindirathwella333-create/Intelligent-Systems-Agentic-AI

---

## Project Description

Small and medium manufacturing businesses in Sri Lanka rely on scattered internal documents — SOPs, HR policies, statutory compliance rules — that staff have to search manually to answer routine questions. This project addresses that with a Retrieval-Augmented Generation (RAG) system fronted by multiple cooperating agents: a router that classifies each incoming question, domain-specialist agents (inventory/production and HR/compliance) that retrieve and answer from the relevant document set, and a reflection agent that checks each answer against its source material before it's shown to the user.

<!-- TODO (you write this): 1-2 more sentences on WHY you picked this specific problem —
your own connection to it (e.g. prior work building an ERP system for a similar business)
makes this more defensible than a generic justification. -->

---

## Architecture

<!-- TODO: Add a diagram here. Two good options:
1. Generate one directly from your actual code (most defensible in a viva, since it's
   generated from what you built, not hand-drawn):

     print(app.get_graph().draw_mermaid())

   Paste the output between triple-backtick ```mermaid fences below — GitHub renders
   Mermaid diagrams natively in README files.

2. Or take a screenshot of the architecture diagram from our conversation and add it as
   an image: ![architecture](docs/architecture.png)
-->

```mermaid
%% Paste your app.get_graph().draw_mermaid() output here
```

---

## Agentic Design Patterns Used

This system implements three agentic design patterns:

1. **Router pattern** — `router_node` in `graph.py` classifies each query as `inventory` or `compliance` using a fast model before any retrieval happens, via `classify()`.
2. **Tool-use / RAG-augmented reasoning** — `inventory_node` and `compliance_node` call `retrieve()` (a Chroma vector search tool) before generating an answer with `synthesize()`, so answers are grounded in retrieved documents rather than the model's own memory.
3. **Reflection / self-critique** — `reflect_node` calls `reflect()`, which checks the draft answer against the retrieved chunks and flags it as unverified if it isn't fully supported.

<!-- TODO (you write this): for each pattern, add the file + line number, since the
brief specifically asks "where in the code it lives." -->

---

## Agent-to-Agent Communication

The system uses two cooperating agent roles — a router agent and domain-specialist agents (inventory, compliance) — that exchange a structured state object (`AgentState`, a `TypedDict` in `graph.py`) rather than plain text. Each node reads specific fields from the shared state and writes new ones, e.g. the router writes `route`, the domain agent reads `route` and writes `draft_answer` and `retrieved_chunks`, and the reflection agent reads those and writes `needs_review` and `final_answer`.

<!-- TODO (you write this, 3-4 sentences): describe this in your own words — what does
each field in AgentState represent, and why pass structured state through LangGraph's
graph rather than, say, separate API calls? What would happen if a node didn't return
one of the expected fields? -->

---

## Model Selection Strategy

| Sub-task | Model (provider) | Why chosen |
|---|---|---|
| Intent routing / classification | Free-tier models via OpenRouter (Gemma, GPT-OSS, fallback chain) | <!-- TODO: your reasoning — think about latency/cost, and what you observed today --> |
| Deep reasoning / answer synthesis | Larger free-tier model (Nemotron-3-Super-120B) via OpenRouter, fallback chain | <!-- TODO: why does synthesis need a different model than routing? --> |
| Reflection / fact-checking | Free-tier models via OpenRouter, wider fallback chain | <!-- TODO: why a wider fallback list here specifically? --> |

<!-- TODO (you write this, a short paragraph): You have real, first-hand material for
this section that most students won't — you watched free-tier models get deprecated
mid-session, hit daily rate limits, and return reasoning-model output that got truncated
before reaching an answer. Describe the fallback-chain design as a direct response to
those observed failures. This is genuinely strong, honest material for a viva question
like "why does your code retry across multiple models instead of just calling one?" —
don't lose it by leaving this section generic. -->

---

## RAG Pipeline

- **Corpus:** Domain-specific documents split into `documents/inventory/` (stock, reorder, and production procedures) and `documents/compliance/` (HR and EPF/ETF procedures, including excerpts from Sri Lanka's EPF Act).
- **Chunking:** Paragraph-based chunking with a ~500 character target size (`chunk_text()` in `graph.py`), keeping related sentences together rather than splitting mid-thought.
- **Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers) — a free, local embedding model requiring no API key.
- **Vector store:** Chroma, with two separate collections (`inventory`, `compliance`) matching the router's classification, so retrieval only ever searches the relevant domain.

### Retrieval Evaluation

<!-- TODO: fill this table with 5 real queries you ran, and your honest assessment of
whether the retrieved chunks were actually relevant. You already have this material —
we ran these earlier in the conversation. -->

| Query | Domain | Retrieved relevant chunk? | Notes |
|---|---|---|---|
| "When should we reorder raw materials?" | Inventory | | |
| "Who is responsible for placing purchase orders?" | Inventory | | |
| "What happens if stock drops below 5 days?" | Inventory | | |
| "What does HR need to collect when a new employee joins?" | Compliance | | |
| "Who is responsible for reconciling EPF discrepancies?" | Compliance | | |
| "What's our current headcount and payroll cost?" (out-of-scope test) | — | | Should correctly decline rather than hallucinate |

---

## Setup Instructions

### Run locally

```bash
git clone https://github.com/savindirathwella333-create/Intelligent-Systems-Agentic-AI.git
cd Intelligent-Systems-Agentic-AI
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
source venv/bin/activate          # Mac/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your-key-here
```

Run the app:
```bash
streamlit run app.py
```

### Secrets management

API keys are loaded from a local `.env` file (via `python-dotenv`) or, in the deployed version, from Streamlit Community Cloud's built-in Secrets manager. The `.env` file is excluded from version control via `.gitignore` and is never present in the committed source code or commit history.

---

## Known Limitations

- **Free-tier model reliability:** the system depends on OpenRouter's free-tier models, which are subject to daily rate limits (50 requests/day per account), per-model deprecation without notice, and shared congestion during peak usage. The fallback-chain design mitigates but does not eliminate this — under heavy congestion, all fallback models can fail simultaneously.
- **Small corpus:** the current document set is a representative sample rather than a full production corpus; retrieval quality would need re-evaluation at a larger scale.
- **No conversation memory:** each query is handled independently; the system does not currently support multi-turn follow-up questions.
- **English only:** the router and synthesis prompts assume English-language queries.

<!-- TODO: add any other limitations you're aware of from your own testing. -->

---

## Technologies Used

- **Orchestration:** LangGraph
- **LLM access:** OpenRouter (OpenAI-compatible API)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** Chroma
- **UI:** Streamlit
- **Deployment:** Streamlit Community Cloud
