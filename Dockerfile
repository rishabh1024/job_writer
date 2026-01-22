# syntax=docker/dockerfile:1.4
FROM langchain/langgraph-api:3.12

# Set Python environment variables (best practice)
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PORT=7860 \
  LANGGRAPH_PORT=7860

# Create user with UID 1000 for HuggingFace Spaces compatibility
RUN useradd -m -u 1000 hf_user

ENV LANGSERVE_GRAPHS='{"job_app_graph": "/deps/job_writer/src/job_writing_agent/graph/agent_workflow_graph.py:job_app_graph", "research_workflow": "/deps/job_writer/src/job_writing_agent/nodes/research_workflow.py:research_workflow", "data_loading_workflow": "/deps/job_writer/src/job_writing_agent/nodes/data_loading_workflow.py:data_loading_workflow"}'

# Copy package metadata and structure files (needed for editable install)
COPY --chown=hf_user:hf_user pyproject.toml langgraph.json README.md /deps/job_writer/

# Create src directory structure (needed for setuptools to find packages)
RUN mkdir -p /deps/job_writer/src

# Copy source code (required for editable install)
COPY --chown=hf_user:hf_user src/ /deps/job_writer/src/

# Install Python dependencies as ROOT using --system flag
# Using cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
  for dep in /deps/*; do \
  if [ -d "$dep" ]; then \
  echo "Installing $dep"; \
  (cd "$dep" && uv pip install --system --no-cache-dir -c /api/constraints.txt -e .); \
  fi; \
  done

# Install Playwright system dependencies (after playwright package is installed)
RUN playwright install-deps chromium

# Create user's cache directory for Playwright browsers (BEFORE installing browsers)
# This ensures browsers are installed to the correct location that persists in the image
RUN mkdir -p /home/hf_user/.cache/ms-playwright && \
  chown -R hf_user:hf_user /home/hf_user/.cache

# Install Playwright browser binaries to user's home directory
# Set PLAYWRIGHT_BROWSERS_PATH to ensure browsers are installed to the right location
# Use cache mount ONLY for the download cache, but install to persistent location
RUN --mount=type=cache,target=/root/.cache/ms-playwright \
  PLAYWRIGHT_BROWSERS_PATH=/home/hf_user/.cache/ms-playwright \
  playwright install chromium && \
  # Fix ownership after installation (browsers are installed as root)
  chown -R hf_user:hf_user /home/hf_user/.cache/ms-playwright

# Create API directories and install langgraph-api as ROOT
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && \
  touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py && \
  uv pip install --system --no-cache-dir --no-deps -e /api

# Fix permissions for packages that write to their own directories
# Make ONLY the specific directories writable (not entire site-packages)
RUN mkdir -p /usr/local/lib/python3.12/site-packages/litellm/litellm_core_utils/tokenizers && \
  chown -R hf_user:hf_user /usr/local/lib/python3.12/site-packages/litellm/litellm_core_utils/tokenizers && \
  chmod -R u+w /usr/local/lib/python3.12/site-packages/litellm/litellm_core_utils/tokenizers

# Create user cache directories with proper permissions (BEFORE switching user)
# Following XDG Base Directory Specification: https://specifications.freedesktop.org/basedir-spec/
RUN mkdir -p /home/hf_user/.cache/tiktoken \
  /home/hf_user/.cache/litellm \
  /home/hf_user/.cache/huggingface \
  /home/hf_user/.cache/torch \
  /home/hf_user/.local/share && \
  chown -R hf_user:hf_user /home/hf_user/.cache /home/hf_user/.local

# Switch to hf_user for runtime (after all root operations)
USER hf_user

# Set environment variables following XDG Base Directory Specification
# This ensures all packages respect standard cache locations
ENV HOME=/home/hf_user \
  PATH="/home/hf_user/.local/bin:$PATH" \
  XDG_CACHE_HOME=/home/hf_user/.cache \
  XDG_DATA_HOME=/home/hf_user/.local/share \
  XDG_CONFIG_HOME=/home/hf_user/.config \
  # Package-specific cache directories (for packages that don't fully respect XDG)
  TIKTOKEN_CACHE_DIR=/home/hf_user/.cache/tiktoken \
  HF_HOME=/home/hf_user/.cache/huggingface \
  TORCH_HOME=/home/hf_user/.cache/torch \
  # Playwright browsers path (so it knows where to find browsers at runtime)
  PLAYWRIGHT_BROWSERS_PATH=/home/hf_user/.cache/ms-playwright


WORKDIR /deps/job_writer

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Healthcheck (LangGraph API typically has /ok endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/ok')" || exit 1