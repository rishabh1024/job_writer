# Resume Storage Options for HF Spaces Deployment

This guide explains different ways to store and access your resume file for the deployed LangGraph application on HuggingFace Spaces.

## Problem

HuggingFace Spaces doesn't allow binary files (PDFs) in git repositories. We removed `resume.pdf` from git, but the workflow needs access to it.

## Solution Options

### ✅ Option 1: URL Support (Easiest - Already Implemented!)

**Status:** ✅ **Code updated - now supports URLs!**

You can now provide a resume URL instead of a file path. The code will automatically download it.

**Supported URL formats:**
- `https://example.com/resume.pdf` - Direct HTTP/HTTPS links
- `https://github.com/username/repo/raw/main/resume.pdf` - GitHub raw files
- `https://drive.google.com/uc?export=download&id=FILE_ID` - Google Drive (public)
- Any publicly accessible URL

**How to use:**

1. **Upload resume to a public location:**
   - GitHub: Upload to a repo and use the "raw" file URL
   - Google Drive: Make file public, get shareable link
   - Dropbox: Get public link
   - Any web server or CDN

2. **Use the URL in your API call:**
   ```json
   {
     "assistant_id": "job_app_graph",
     "input": {
       "resume_path": "https://github.com/username/repo/raw/main/resume.pdf",
       "job_description_source": "https://example.com/job",
       "content_category": "cover_letter"
     }
   }
   ```

**Pros:**
- ✅ No code changes needed (already implemented)
- ✅ Works with any public URL
- ✅ No additional services required
- ✅ Easy to update (just replace the file at the URL)

**Cons:**
- ⚠️ File must be publicly accessible
- ⚠️ Requires internet connection to download

---

### Option 2: HuggingFace Hub Dataset (Recommended for Production)

Store your resume in HF Hub as a dataset - native integration with HF Spaces.

**Steps:**

1. **Install HF Hub CLI:**
   ```bash
   pip install huggingface_hub
   ```

2. **Login to HF:**
   ```bash
   huggingface-cli login
   ```

3. **Create a dataset and upload resume:**
   ```bash
   # Create dataset (one-time)
   huggingface-cli repo create resume-dataset --type dataset
   
   # Upload resume
   huggingface-cli upload Rishabh2095/resume-dataset resume.pdf resume.pdf
   ```

4. **Access in code (add to workflow):**
   ```python
   from huggingface_hub import hf_hub_download
   import tempfile
   
   # Download resume from HF Hub
   resume_path = hf_hub_download(
       repo_id="Rishabh2095/resume-dataset",
       filename="resume.pdf",
       cache_dir="/tmp"
   )
   ```

5. **Use in API call:**
   ```json
   {
     "assistant_id": "job_app_graph",
     "input": {
       "resume_path": "/tmp/resume.pdf",  # After downloading from HF Hub
       "job_description_source": "https://example.com/job",
       "content_category": "cover_letter"
     }
   }
   ```

**Pros:**
- ✅ Native HF integration
- ✅ Private datasets supported
- ✅ Version control for resume
- ✅ No external dependencies

**Cons:**
- ⚠️ Requires code modification to download from HF Hub
- ⚠️ Slight overhead for downloading

---

### Option 3: Object Storage (S3, GCS, Azure Blob)

Use cloud object storage for production scalability.

**Example: AWS S3**

1. **Upload to S3:**
   ```bash
   aws s3 cp resume.pdf s3://your-bucket/resume.pdf --acl public-read
   ```

2. **Use public URL:**
   ```json
   {
     "resume_path": "https://your-bucket.s3.amazonaws.com/resume.pdf"
   }
   ```

**For private S3 (requires credentials):**
- Add AWS credentials as HF Space secrets
- Use `boto3` to download in code

**Pros:**
- ✅ Scalable and reliable
- ✅ Supports private files with auth
- ✅ Industry standard

**Cons:**
- ⚠️ Requires cloud account setup
- ⚠️ May incur costs
- ⚠️ More complex setup

---

### Option 4: HF Spaces Persistent Storage

HF Spaces provides `/tmp` directory that persists across restarts.

**Steps:**

1. **Upload file via API or during build:**
   - Add file to Docker image (but this increases image size)
   - Or download during container startup

2. **Use in code:**
   ```python
   # In your workflow initialization
   DEFAULT_RESUME_PATH = "/tmp/resume.pdf"
   ```

**Pros:**
- ✅ No external dependencies
- ✅ Fast access (local file)

**Cons:**
- ⚠️ File must be in Docker image (increases size)
- ⚠️ Not easily updatable without rebuild

---

### Option 5: Environment Variable with URL

Store resume URL as an HF Space secret.

**Steps:**

1. **Add to HF Space Secrets:**
   - Go to Space Settings → Variables and secrets
   - Add: `RESUME_URL=https://example.com/resume.pdf`

2. **Use in code:**
   ```python
   import os
   resume_path = os.getenv("RESUME_URL", "default_path_or_url")
   ```

**Pros:**
- ✅ Easy to update (change secret, no code deploy)
- ✅ Can point to any URL
- ✅ Works with Option 1 (URL support)

**Cons:**
- ⚠️ Requires code modification to read env var

---

## Recommended Approach

**For Quick Start:** Use **Option 1 (URL Support)** - just upload your resume to GitHub, Google Drive, or any public URL and use that URL in your API calls.

**For Production:** Use **Option 2 (HF Hub Dataset)** - native integration, private support, version control.

## Implementation Status

- ✅ **URL Support:** Implemented in `parse_resume()` function
- ⏳ **HF Hub Integration:** Can be added if needed
- ⏳ **Environment Variable:** Can be added if needed

## Testing

Test with a public resume URL:

```powershell
# Test with GitHub raw file URL
$body = @{
    assistant_id = "job_app_graph"
    input = @{
        resume_path = "https://github.com/username/repo/raw/main/resume.pdf"
        job_description_source = "https://example.com/job"
        content_category = "cover_letter"
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://rishabh2095-agentworkflowjobapplications.hf.space/runs/wait" `
    -Method POST -Body $body -ContentType "application/json"
```

## Next Steps

1. Upload your resume to a public location (GitHub, Google Drive, etc.)
2. Get the public URL
3. Use that URL in your API calls as `resume_path`
4. The code will automatically download and process it!
