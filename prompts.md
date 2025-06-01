## 1. Title / One-Line Summary
> *E.g.* “embed_query returns empty vector with OllamaEmbeddings”

---

## 2. Goal / Expected Behavior
- What you’re trying to achieve  
  *E.g.* “Index documents with OllamaEmbeddings and query via Pinecone, then feed them into Llama3.2 for answer generation.”

---

## 3. Environment
- **Python**:  
- **langchain**:  
- **Ollama CLI / Daemon**:  
- **OS** (and version):  
- **Other dependencies**:

---

## 4. Minimal Reproducible Code
```python
# Paste just enough code to reproduce the issue:
from langchain.embeddings import OllamaEmbeddings
emb = OllamaEmbeddings(model="ollama/llama3.2-embed")
vec = emb.embed_query("hello")
print(len(vec))  # unexpected result
