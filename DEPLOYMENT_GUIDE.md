# Deployment Guide for Job Application Agent

## Option 1: LangGraph Cloud (Easiest & Recommended)

### Prerequisites
- LangGraph CLI installed (`langgraph-cli` in requirements.txt)
- `langgraph.json` already configured ✅

### Steps

1. **Install LangGraph CLI** (if not already):
```powershell
pip install langgraph-cli
```

2. **Login to LangGraph Cloud**:
```powershell
langgraph login
```

3. **Deploy your agent**:
```powershell
langgraph deploy
```

4. **Get your API endpoint** - LangGraph Cloud provides a REST API automatically

### Cost
- **Free tier**: Limited requests/month
- **Paid**: Pay-per-use pricing

### Pros
- ✅ Zero infrastructure management
- ✅ Built-in state persistence
- ✅ Automatic API generation
- ✅ LangSmith integration
- ✅ Perfect for LangGraph apps

### Cons
- ⚠️ Vendor lock-in
- ⚠️ Limited customization

---

## Option 2: Railway.app (Simple & Cheap)

### Steps

1. **Create a FastAPI wrapper** (create `api.py`):
```python
from fastapi import FastAPI, File, UploadFile
from job_writing_agent.workflow import JobWorkflow
import tempfile
import os

app = FastAPI()

@app.post("/generate")
async def generate_application(
    resume: UploadFile = File(...),
    job_description: str,
    content_type: str = "cover_letter"
):
    # Save resume temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        resume_path = tmp.name
    
    try:
        workflow = JobWorkflow(
            resume=resume_path,
            job_description_source=job_description,
            content=content_type
        )
        result = await workflow.run()
        return {"result": result}
    finally:
        os.unlink(resume_path)
```

2. **Create `Procfile`**:
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

3. **Deploy to Railway**:
   - Sign up at [railway.app](https://railway.app)
   - Connect GitHub repo
   - Railway auto-detects Python and runs `Procfile`

### Cost
- **Free tier**: $5 credit/month
- **Hobby**: $5/month for 512MB RAM
- **Pro**: $20/month for 2GB RAM

### Pros
- ✅ Very simple deployment
- ✅ Auto-scaling
- ✅ Free tier available
- ✅ Automatic HTTPS

### Cons
- ⚠️ Need to add FastAPI wrapper
- ⚠️ State management needs Redis/Postgres

---

## Option 3: Render.com (Similar to Railway)

### Steps

1. **Create `render.yaml`**:
```yaml
services:
  - type: web
    name: job-writer-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENROUTER_API_KEY
        sync: false
      - key: TAVILY_API_KEY
        sync: false
```

2. **Deploy**:
   - Connect GitHub repo to Render
   - Render auto-detects `render.yaml`

### Cost
- **Free tier**: 750 hours/month (sleeps after 15min inactivity)
- **Starter**: $7/month (always on)

### Pros
- ✅ Free tier for testing
- ✅ Simple YAML config
- ✅ Auto-deploy from Git

### Cons
- ⚠️ Free tier sleeps (cold starts)
- ⚠️ Need FastAPI wrapper

---

## Option 4: Fly.io (Good Free Tier)

### Steps

1. **Install Fly CLI**:
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

2. **Create `Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
```

3. **Deploy**:
```powershell
fly launch
fly deploy
```

### Cost
- **Free tier**: 3 shared-cpu VMs, 3GB storage
- **Paid**: $1.94/month per VM

### Pros
- ✅ Generous free tier
- ✅ Global edge deployment
- ✅ Docker-based (flexible)

### Cons
- ⚠️ Need Docker knowledge
- ⚠️ Need FastAPI wrapper

---

## Option 5: AWS Lambda (Serverless - Pay Per Use)

### Steps

1. **Create Lambda handler** (`lambda_handler.py`):
```python
import json
from job_writing_agent.workflow import JobWorkflow

def lambda_handler(event, context):
    # Parse event
    body = json.loads(event['body'])
    
    workflow = JobWorkflow(
        resume=body['resume_path'],
        job_description_source=body['job_description'],
        content=body.get('content_type', 'cover_letter')
    )
    
    result = workflow.run()
    
    return {
        'statusCode': 200,
        'body': json.dumps({'result': result})
    }
```

2. **Package and deploy** using AWS SAM or Serverless Framework

### Cost
- **Free tier**: 1M requests/month
- **Paid**: $0.20 per 1M requests + compute time

### Pros
- ✅ Pay only for usage
- ✅ Auto-scaling
- ✅ Very cheap for low traffic

### Cons
- ⚠️ 15min timeout limit
- ⚠️ Cold starts
- ⚠️ Complex setup
- ⚠️ Need to handle state externally

---

## Recommendation

**For your use case, I recommend:**

1. **Start with LangGraph Cloud** - Easiest, built for your stack
2. **If you need more control → Railway** - Simple, good free tier
3. **If you need serverless → AWS Lambda** - Cheapest for low traffic

---

## Quick Start: FastAPI Wrapper (for Railway/Render/Fly.io)

Create `api.py` in your project root:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from job_writing_agent.workflow import JobWorkflow
import tempfile
import os
import asyncio

app = FastAPI(title="Job Application Writer API")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate_application(
    resume: UploadFile = File(...),
    job_description: str,
    content_type: str = "cover_letter"
):
    """Generate job application material."""
    # Save resume temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await resume.read()
        tmp.write(content)
        resume_path = tmp.name
    
    try:
        workflow = JobWorkflow(
            resume=resume_path,
            job_description_source=job_description,
            content=content_type
        )
        
        # Run workflow (assuming it's async or can be wrapped)
        result = await asyncio.to_thread(workflow.run)
        
        return JSONResponse({
            "status": "success",
            "result": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(resume_path):
            os.unlink(resume_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then update `requirements.txt` to ensure FastAPI and uvicorn are included (they already are ✅).

