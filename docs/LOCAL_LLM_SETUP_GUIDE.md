# Local LLM Setup & Optimization Guide — Project-RAG_LLM

This guide explains how to configure, run, and optimize local Large Language Models (LLMs) with **Project-RAG_LLM** using either **Ollama** or **LM Studio**, and how our architectural safeguards prevent weak local models from hallucinating or slowing down.

---

## 🛡️ 1. How the Architecture Protects Weak Local Models

Local models (especially 3B–8B quantized models on consumer hardware) typically have lower instruction-following ability and smaller effective context windows than trillion-parameter cloud APIs. Project-RAG_LLM is engineered specifically to protect against these weaknesses:

| Common Local LLM Weakness | How Project-RAG_LLM Mitigates It |
|---|---|
| **Hallucinations & Rambling** | Strict system prompt isolation pinned at **`temperature = 0.3`**. If context is missing, the model is strictly constrained to abstain rather than guess. |
| **Context Window Choking** | **Cross-Encoder Reranking (`bge-reranker-v2-m3`)** filters out noisy chunks with `min_rerank_score = 0.08`. The local model is sent only 2–4 verified parent chunks (~1,200 chars each), never massive PDF dumps. |
| **Broken Function/Tool Calling** | Pure-Python **Deterministic FSM State Machine**. Tool dispatching (`document_search`, `document_summary`, safe AST math, comparisons) is handled by Python heuristics, so the local model is never burdened with fragile JSON schema generation. |
| **Infinite Reformulation Loops** | **Corrective RAG (CRAG)** enforces a strict **`loop_count <= 1`** boundary. If query reformulation fails on the second try, it exits cleanly with a transparent abstention notice. |
| **Crash / Offline Stalls** | **`<250ms` connection timeout**. If Ollama or LM Studio crashes or runs out of VRAM, the pipeline detects it in milliseconds and fails over automatically. |
| **3-Tier Failover Resilience** | **Local LLM** $\to$ **Auto-Gemini Cloud Failover** (if API key present) $\to$ **Grounded Extractive Synthesis** (direct citation extracts if completely offline). |

---

## 💻 2. Recommended Local Models by Hardware Tier

### Tier 1: Consumer GPU (6GB – 8GB VRAM) — e.g., RTX 3060, RTX 4060, Laptop GPUs
* **Primary Recommendation:** `qwen2.5:7b-instruct` (or `qwen2.5:7b-instruct-q4_k_m`)
  * *Why:* Best-in-class multi-lingual reasoning, excellent structured context adherence, and low VRAM footprint (~5.2GB).
* **Alternative:** `llama3.1:8b-instruct-q4_k_m`
  * *Why:* Solid conversational cadence, widely supported across all runtimes (~5.5GB VRAM).
* **Ultra-Lightweight (4GB VRAM):** `phi-3.5-mini:3.8b-instruct-q4_k_m`

### Tier 2: Mid-Range GPU (12GB – 16GB VRAM) — e.g., RTX 3080, RTX 4070, RTX 4080
* **Primary Recommendation:** `qwen2.5:14b-instruct-q4_k_m`
  * *Why:* Exceptional synthesis quality, near GPT-4-mini level reasoning on dense technical documents.
* **Alternative:** `mistral-nemo:12b-instruct-2407-q4_k_m`

---

## 🚀 3. Quickstart Option A: Ollama (Recommended for Command-Line Simplicity)

### Step 1: Install Ollama
Download and install Ollama from [https://ollama.com](https://ollama.com).

### Step 2: Pull the Recommended Model
Open your terminal (PowerShell or Bash) and run:
```powershell
# Pull the recommended 8B model
ollama pull llama3.1:8b

# Or pull Qwen 2.5 7B
ollama pull qwen2.5:7b
```

### Step 3: Configure Project-RAG_LLM
In your `.env` file (or pass as environment variables):
```env
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```
Ollama runs in the background on `http://localhost:11434`. Start the RAG server, and it will immediately connect!

---

## 🎨 4. Quickstart Option B: LM Studio (Recommended for Visual GUI & Tuning)

### Step 1: Install LM Studio
Download LM Studio from [https://lmstudio.ai](https://lmstudio.ai).

### Step 2: Download a Model
1. Open LM Studio and search for `qwen2.5-7b-instruct` or `llama-3.1-8b-instruct`.
2. Download a `Q4_K_M` or `Q5_K_M` quantized version.

### Step 3: Start the Local Inference Server
1. Click the **Local Server (Double-Arrow `<->`)** icon in the left navigation sidebar.
2. Select your downloaded model from the top dropdown.
3. In the right panel, configure settings:
   * **GPU Offload:** Set to **Max** (slider all the way to the right).
   * **Context Length:** Set to **`4096`** or **`8192`**.
   * **Port:** Ensure it is set to **`1234`** (Default: `http://localhost:1234/v1`).
4. Click **Start Server**.

### Step 4: Configure Project-RAG_LLM
In your `.env` file:
```env
DEFAULT_LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=qwen2.5-7b-instruct
```

---

## ☁️ 5. Setting Up Zero-Config Cloud Failover (Gemini)

If your local model is offline or your PC is running heavy tasks, Project-RAG_LLM includes seamless cloud failover:

1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com).
2. Enter it into the Web Studio UI:
   * Open the Web Studio at `http://127.0.0.1:8000`.
   * Open the **Model Engine** panel in the left sidebar.
   * Paste your key into the **Cloud Gemini API Key** field and click **Save**.
   * The key is stored locally in your browser's `localStorage` and sent securely via the `X-Gemini-API-Key` request header.
3. Alternatively, save it in your project `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

---

## 🔍 6. How to Verify Local LLM Connectivity

You can check whether your local LLM is detected and running with one API call:

```powershell
# Check connectivity status across all 3 providers
curl -s http://127.0.0.1:8000/api/config/llm-status
```
Example response:
```json
{
  "active_provider": "lmstudio",
  "providers": {
    "lmstudio": {"configured": true, "connected": true, "model": "qwen2.5-7b-instruct"},
    "ollama": {"configured": true, "connected": false, "model": "llama3.1:8b"},
    "gemini": {"configured": true, "connected": true, "model": "gemini-2.5-flash"}
  }
}
```
If your local provider shows `"connected": false`, verify that Ollama or LM Studio is open and that the port is not blocked by a firewall.
