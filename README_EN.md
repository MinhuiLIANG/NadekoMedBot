[中文](README.md) | **English**

---

## Nadeko Sengoku's Ramblings (Project Notes)

<p align="center">
  <img src="FrontEnd/images/CircleC.jpg" alt="Nadeko Sengoku · CircleC" width="260">
</p>

Since you've scrolled all the way here, I might as well explain how this little project works. So you don't poke around the codebase and end up with a headache.

You can think of it as a **minimal** Agent app that covers **fine-tuning** a local LLM, **workflow**, **frontend**, and **server deployment**. If you find this helpful, a star would be nice. Thanks.

---

## Project Structure

- **`goofyAgent.py`**  
  The brain of the whole service.  
  - Runs a FastAPI HTTP server exposing the `/generate` endpoint.  
  - Sends user input to DeepSeek's API to decide if it's "medical-related."  
  - If yes, forwards to the local medical model (`medicalAPI`) for advice;  
  - If no, uses Nadeko's usual chat style for small talk.  
  - Keeps a bit of conversation history in `memory.json` for context (nothing scary, just a simple notebook).  

- **`medicalAPI.py`**  
  Where the local medical model lives.  
  - Uses `Qwen/Qwen3-8B` from Hugging Face, plus the LoRA weights you trained in `exp1`.  
  - Loads the model in 4bit quantization to save VRAM.  
  - Exposes `evaluate(input_text)` for `goofyAgent` to get medical suggestions.  
  - No HTTP server here—pure inference.  

- **`FrontEnd/goofyDoc.html`**  
  The chat interface.  
  - Single-page HTML with the chat UI.  
  - Calls the backend `/generate` endpoint.  
  - Open in a browser and you're good to go.  

- **`Finetune/`**  
  For those who want to mess with fine-tuning. My creator Hamsuke referenced [`this repo`](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN), which is really helpful.  
  - `Finetune/medicalLLM.py` — fine-tuning / data prep script.  
  - `Finetune/tmp_medical.json` — a small slice of Chinese medical dialogue data from [`Chinese-medical-dialogue-data`](https://github.com/Toyhom/Chinese-medical-dialogue-data).  
  - Used for the LoRA fine-tuning pipeline; you can ignore it if you only want to chat.  

- **`FrontEnd/images/`**  
  Background and decoration images for the UI.  

- **`exp1/`**  
  Fine-tuning output directory.  
  - `checkpoint-*/` holds trained LoRA weights and tokenizer files.  
  - `medicalAPI.py` auto-picks the latest checkpoint to load.  

- **`environment.yml`**  
  Conda environment file.  
  - Required if you want to set things up on a new machine.  

---

## Environment Setup

If you're starting on a clean machine:

1. **Create Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate xxx (env name)
   ```

2. **Prepare model files**

   - Make sure `exp1/` contains at least one trained `checkpoint-*` subdirectory.  
   - On first run, `medicalAPI.py` will download the base model from Hugging Face to `cache/`.  

3. **Set DeepSeek API Key**

   Either:

   - Environment variable:

     ```bash
     set DEEPSEEK_API_KEY=your_key
     ```

   - Or as a CLI argument:

     ```bash
     python goofyAgent.py --apikey your_key
     ```

   Without a key, the medical-question classifier will fail.

---

## Start the Backend

Run `goofyAgent.py`:

- **Option 1: Direct run**

  ```bash
  python goofyAgent.py --apikey your_key
  ```

  Uses uvicorn to serve at:

  - `http://0.0.0.0:8000`
  - Main endpoint: `POST /generate`

- **Option 2: Manual uvicorn**

  ```bash
  uvicorn goofyAgent:app --host 0.0.0.0 --port 8000
  ```

  Make sure `DEEPSEEK_API_KEY` is set in the environment.

---

## Frontend

1. Open `FrontEnd/goofyDoc.html` in a browser.  
2. Ensure the backend URL in the page points to your FastAPI server (e.g. `http://127.0.0.1:8000/generate`).  
3. Start chatting.

For fine-tuning or swapping datasets, check the scripts and data in `Finetune/`.

---

## API

- **`POST /generate`**  
  - Request (JSON): `user_input`, `user_id` (optional, default `"anonymous"`)  
  - Response (JSON): `reply`, `route` (`"medical"` | `"chat"`), `judge_raw`  

---

## Notes

- Model loading and inference are VRAM-heavy; weak machines may be slow.  
- `memory.json` is plain conversation history; remove or change it if you prefer.  
- After fine-tuning, a new `exp1/` appears in `Finetune/`. To use the new weights, replace the root `exp1/` or copy the checkpoints into it and ensure the latest one is used (see `medicalAPI.py` lines 27–35).  
- Hamsuke's deployment: backend + `exp1/` on a GPU server with nginx + HTTPS, frontend on a cheap server or GitHub Pages / Cloudflare.

That's about it. Treat me well.
