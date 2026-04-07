# M.A.R.A. - Multi-Agent Research Analyst

A self-correcting, multi-agent research system built with **LangGraph** that autonomously plans, researches, critiques, and writes comprehensive reports on any topic.

M.A.R.A. uses five specialized agents orchestrated through a stateful graph with conditional routing. The system searches the web, synthesizes findings, identifies knowledge gaps, and iteratively refines its research before producing a polished markdown report.

---

## Architecture

```
                          +----------+
                          | Planner  |
                          +----+-----+
                               |
                        breaks topic into
                        3-5 sub-questions
                               |
                       +-------v--------+
                       |   Researcher   |
                       +-------+--------+
                               |
                      searches web per question
                      and synthesizes findings
                               |
                        +------v------+
                   +--->|   Critic    |
                   |    +------+------+
                   |           |
                   |     gaps found?
                   |      /        \
                   |    yes         no
                   |    /             \
            +------v-------+    +-----v-----+
            |Gap Researcher|    |   Writer   |
            +--------------+    +-----------+
```

The **Critic -> Gap Researcher** loop is the key differentiator. Instead of producing a report from a single pass of research, M.A.R.A. evaluates its own findings, identifies what is missing, and loops back to fill those gaps. This continues until the Critic is satisfied or the max iteration limit is reached.

Each agent is a LangGraph node with typed state passing via `ResearchState`. Conditional edges control the flow between critique and gap-filling, making the graph self-correcting by design.

---

## Demo

The Streamlit dashboard shows each agent's activity in real time as M.A.R.A. works through a research query.

| Agent | Role |
|-------|------|
| **Planner** | Decomposes the topic into 3-5 targeted sub-questions |
| **Researcher** | Searches the web (DuckDuckGo) and synthesizes findings per question |
| **Critic** | Reviews research quality and identifies knowledge gaps |
| **Gap Researcher** | Fills gaps found by the Critic, then loops back for re-evaluation |
| **Writer** | Produces a polished markdown report with executive summary and sections |

---

## Quick Start

```bash
# Clone
git clone https://github.com/MuditNautiyal-21/MARA.git
cd MARA

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your Google API key to .env (free at aistudio.google.com/apikey)

# Run
streamlit run app.py
```

The app will open at `http://localhost:8501`. Enter a research topic, set the refinement loop count, and hit Run Research.

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| **LangGraph** | Multi-agent orchestration with conditional edges and stateful graph |
| **LangChain** | LLM abstraction and DuckDuckGo search integration |
| **Google Gemma 4 31B** | Free LLM via Google AI Studio (Apache 2.0, 256K context) |
| **Streamlit** | Real-time agent activity dashboard |
| **DuckDuckGo Search** | Free web search with no API key required |

---

## Project Structure

```
MARA/
├── agents/
│   ├── __init__.py   # Package marker
│   ├── state.py      # ResearchState TypedDict (shared graph state)
│   ├── nodes.py      # Agent node functions, LLM calls, and routing logic
│   └── graph.py      # LangGraph compilation with conditional edges
├── app.py            # Streamlit UI with real-time agent activity panel
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Configuration

M.A.R.A. uses **Gemma 4 31B** through Google AI Studio by default. You can swap models by changing the `model` parameter in `agents/nodes.py`:

```python
# Other free options via Google AI Studio
model="gemini-2.0-flash"
model="gemma-4-26b-a4b-it"    # MoE variant, faster
```

---

## How It Works (Under the Hood)

1. **Planner** receives the topic and prompts the LLM to decompose it into 3-5 focused sub-questions, returned as a JSON array.

2. **Researcher** takes each sub-question, runs a DuckDuckGo search, and passes the raw results to the LLM for a concise 2-3 paragraph synthesis.

3. **Critic** reviews all findings together, assesses quality, and returns a JSON object with a critique paragraph and an array of gap questions. If gaps exist and the iteration limit has not been reached, the graph routes to the Gap Researcher.

4. **Gap Researcher** researches the missing questions using the same search-and-synthesize pipeline, then routes back to the Critic for re-evaluation.

5. **Writer** receives all findings (original + gap research) plus the critique, and produces a structured markdown report with title, executive summary, themed sections, and conclusion.

---

## License

MIT