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

# Job Writer Module

A modular, well-structured package for creating tailored job applications using LangChain and LangGraph with LangSmith observability.

## Features

- Creates personalized job application materials based on resumes and job descriptions
- Supports multiple application types: cover letters, bullet points, and LinkedIn messages
- Uses RAG for personalization and web search for company research
- Provides human-in-the-loop feedback integration
- Implements self-consistency voting for quality control

## Installation

```bash
# Install the package and its dependencies
pip install -e .

# Install development dependencies (including linting tools)
pip install -r requirements-dev.txt
```

## Code Standards and Linting

This project uses several tools to ensure code quality:

1. **Black** - Code formatter that enforces consistent style
2. **isort** - Sorts imports according to best practices
3. **Flake8** - Style guide enforcement
4. **mypy** - Static type checking

### Running the Linters

```bash
# Format code with Black
black job_writer/

# Sort imports
isort job_writer/

# Check style with Flake8
flake8 job_writer/

# Type checking with mypy
mypy job_writer/
```

### Pre-commit Hooks

We use pre-commit hooks to automatically run linters before each commit:

```bash
# Install the pre-commit hooks
pip install pre-commit
pre-commit install

# You can also run the hooks manually
pre-commit run --all-files
```

## Usage Example

```python
import asyncio
from job_writer.workflow import run_job_application_writer

# Run the job application writer
result = asyncio.run(run_job_application_writer(
    resume_path="path/to/resume.pdf",
    job_desc_path="https://example.com/job-posting",
    content="cover_letter"
))

print(result["final"])
```

Alternatively, you can use the command-line interface:

```bash
python -m job_writer.workflow --resume path/to/resume.pdf --job https://example.com/job-posting --type cover_letter
```

Run with uv

```bash
uv run --active -m job_writing_agent.workflow --resume resumefilepath --job jobposturl
```