---
title: Job Application Writer
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
python_version: 3.12.8
---

# Job Application Writer Agent

An AI-powered agent workflow that generates tailored job application materials — cover letters, bullet-point summaries, and LinkedIn connection notes — using LangGraph, LangChain, and DSPy. The system performs automated company research, drafts content with self-consistency voting, applies AI critique, and supports human-in-the-loop feedback before finalizing output.

**Live Demo:** [EasyApply](https://rishabh2095-easyapply.hf.space/) | [Hugging Face Space](https://huggingface.co/spaces/Rishabh2095/AgentWorkflowJobApplications)

## Features

- **Multi-format input** — Accepts resumes in PDF, TXT, MD, and JSON; job descriptions via URL or Google Docs link
- **Automated company research** — Uses Tavily search with LLM-based relevance filtering to gather context about the target company
- **Multiple output types** — Cover letters, bullet-point highlights, and LinkedIn connection messages
- **Quality control pipeline** — Generates multiple draft variations, selects the best via self-consistency voting, and applies AI critique
- **Human-in-the-loop** — LangGraph interrupt-based approval step so you can provide feedback before finalization
- **Multi-provider LLM support** — Factory pattern supporting OpenRouter, Cerebras, and Ollama (both LangChain and DSPy)
- **Observability** — Full LangSmith tracing with metadata and tags
- **Parallel processing** — Resume and job description parsing run concurrently

## Architecture

The workflow is a LangGraph state machine composed of subgraphs:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Main Workflow Graph                          │
│                                                                     │
│  ┌──────────┐    ┌─────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   LOAD   │───▶│RESEARCH │───▶│ CREATE_DRAFT│───▶│  CRITIQUE │  │
│  │(subgraph)│    │(subgraph│    │             │    │           │  │
│  └──────────┘    └─────────┘    └─────────────┘    └─────┬─────┘  │
│         ▲                                                │        │
│         └── retry on                           ┌─────────▼──────┐ │
│             validation                         │HUMAN_APPROVAL  │ │
│             failure                            │  (interrupt)   │ │
│                                                └─────────┬──────┘ │
│                                                          │        │
│                                                ┌─────────▼──────┐ │
│                                                │    FINALIZE    │ │
│                                                └────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for your chosen LLM provider (OpenRouter, Cerebras, or Ollama) and Tavily

### Installation

```bash
# Clone the repository
git clone https://github.com/rishabh1024/job_writer.git
cd job_writer

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Usage

#### CLI

```bash
# With uv
uv run -m job_writing_agent.workflow \
  --resume path/to/resume.pdf \
  --job https://example.com/job-posting \
  --type cover_letter

# With python directly
python -m job_writing_agent.workflow \
  --resume path/to/resume.pdf \
  --job https://example.com/job-posting \
  --type cover_letter
```

Content types: `cover_letter`, `bullet_points`, `linkedin_note`

#### LangGraph API

When deployed, the workflow is exposed as a LangGraph API with three graphs:

| Graph | Endpoint | Description |
|---|---|---|
| `job_app_graph` | Full pipeline | End-to-end: load → research → draft → critique → approve → finalize |
| `research_workflow` | Research only | Company research with Tavily search and relevance filtering |
| `data_loading_workflow` | Data loading only | Resume and job description parsing |

## Tech Stack

| Category | Tools |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Framework | [LangChain](https://github.com/langchain-ai/langchain), [DSPy](https://github.com/stanfordnlp/dspy) |
| Search | [Tavily](https://tavily.com/) |
| Evaluation | [OpenEvals](https://github.com/langchain-ai/openevals) (LLM-as-judge) |
| Observability | [LangSmith](https://smith.langchain.com/) |
| Web Scraping | [Playwright](https://playwright.dev/) |
| Deployment | Docker, Hugging Face Spaces |
| Package Management | [uv](https://docs.astral.sh/uv/) |

## License

This project is provided as-is for educational and personal use.
