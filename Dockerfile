# syntax=docker/dockerfile:1.4
FROM langchain/langgraph-api:3.12

# HuggingFace Spaces requires port 7860
ENV PORT=7860
ENV LANGGRAPH_PORT=7860

ENV LANGSERVE_GRAPHS='{"job_app_graph": "/deps/job_writer/src/job_writing_agent/workflow.py:job_app_graph", "research_workflow": "/deps/job_writer/src/job_writing_agent/nodes/research_workflow.py:research_workflow", "data_loading_workflow": "/deps/job_writer/src/job_writing_agent/nodes/data_loading_workflow.py:data_loading_workflow"}'

COPY pyproject.toml langgraph.json /deps/job_writer/
COPY src/ /deps/job_writer/src/

RUN for dep in /deps/*; do \
  if [ -d "$dep" ]; then \
  echo "Installing $dep"; \
  (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .); \
  fi; \
  done

# Use cache mount for Playwright - browsers persist between builds!
RUN --mount=type=cache,target=/root/.cache/ms-playwright \
  playwright install chromium && \
  playwright install-deps

RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && \
  touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py && \
  PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

WORKDIR /deps/job_writer

# Expose port for HuggingFace Spaces
EXPOSE 7860