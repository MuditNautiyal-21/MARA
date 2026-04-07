"""Node functions for the multi-agent research graph."""

from __future__ import annotations
import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from agents.state import ResearchState

llm = ChatOpenAI(
    model="gemma-4-31b-it",
    temperature=0.3,
    openai_api_key=os.getenv("GOOGLE_API_KEY"),
    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
)
search_tool = DuckDuckGoSearchResults(max_results=4)


def _clean(text: str) -> str:
    """Strip Gemma thinking tags from output."""
    return re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL).strip()


# ### Planner ######

def planner(state: ResearchState) -> dict:
    """Break the topic into 3-5 focused sub-questions."""
    resp = llm.invoke(
        f"You are a research planner. Given the topic below, produce 3-5 specific "
        f"sub-questions that, when answered together, would form a comprehensive "
        f"research report. Return ONLY a JSON array of strings. No thinking tags.\n\n"
        f"Topic: {state['topic']}"
    )
    content = _clean(resp.content)
    try:
        questions = json.loads(content)
    except json.JSONDecodeError:
        questions = [q.strip("- ").strip() for q in content.splitlines() if q.strip()]
    return {"sub_questions": questions, "status": "Planning complete"}


# ### Researcher ###

def _research_question(question: str) -> dict:
    """Search the web and summarize findings for a single question."""
    raw = search_tool.invoke(question)
    summary = llm.invoke(
        f"You are a research analyst. Based on the search results below, write a "
        f"concise 2-3 paragraph summary answering the question. No thinking tags.\n\n"
        f"Question: {question}\n\nSearch results:\n{raw}"
    )
    return {"question": question, "findings": _clean(summary.content)}


def researcher(state: ResearchState) -> dict:
    """Research all sub-questions."""
    results = [_research_question(q) for q in state["sub_questions"]]
    return {"research_results": results, "status": "Research complete"}


# ### Gap Researcher (runs only on iteration > 0) ##############################

def gap_researcher(state: ResearchState) -> dict:
    """Research gaps identified by the critic."""
    results = [_research_question(g) for g in state.get("gaps", [])]
    return {"gap_research": results, "status": "Gap research complete"}


# ### Critic ######

def critic(state: ResearchState) -> dict:
    """Review research for gaps and quality."""
    all_findings = "\n\n".join(
        f"Q: {r['question']}\n{r['findings']}"
        for r in state["research_results"] + state.get("gap_research", [])
    )
    resp = llm.invoke(
        f"You are a research critic. Review the findings below for the topic "
        f'"{state["topic"]}".\n\n'
        f"Findings:\n{all_findings}\n\n"
        f"Respond in JSON with two keys:\n"
        f'  "critique": a short paragraph assessing quality,\n'
        f'  "gaps": a JSON array of missing sub-questions (empty array if none).\n'
        f"No thinking tags. Return only valid JSON."
    )
    content = _clean(resp.content)
    try:
        parsed = json.loads(content)
        critique = parsed.get("critique", content)
        gaps = parsed.get("gaps", [])
    except json.JSONDecodeError:
        critique = content
        gaps = []
    return {
        "critique": critique,
        "gaps": gaps,
        "iteration": state.get("iteration", 0) + 1,
        "status": "Critique complete",
    }


# ### Writer ######

def writer(state: ResearchState) -> dict:
    """Produce the final markdown report."""
    all_findings = "\n\n".join(
        f"Q: {r['question']}\n{r['findings']}"
        for r in state["research_results"] + state.get("gap_research", [])
    )
    resp = llm.invoke(
        f"You are a senior research writer. Using the findings and critique below, "
        f"write a polished, well-structured markdown report on the topic "
        f'"{state["topic"]}".\n\n'
        f"Include: title, executive summary, sections for each key area, and a "
        f"conclusion. Use headers, bullet points where helpful, and keep it "
        f"professional. Do not include any thinking tags.\n\n"
        f"Findings:\n{all_findings}\n\nCritique:\n{state['critique']}"
    )
    return {"final_report": _clean(resp.content), "status": "Report complete"}


# ### Routing ######

def should_continue(state: ResearchState) -> str:
    """Decide whether to loop back for gap research or proceed to writing."""
    if state.get("gaps") and state.get("iteration", 0) < state.get("max_iterations", 2):
        return "gap_researcher"
    return "writer"
