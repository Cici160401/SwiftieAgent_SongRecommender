<h1 align="center">🎧 Swiftie Mood Agent</h1>
<p align="center"><em>Recommends the Taylor Swift song that best matches your current mood.</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python">
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi">
  <img src="https://img.shields.io/badge/SBERT-all--mpnet--base--v2-ff69b4">
  <img src="https://img.shields.io/badge/Deploy-Render-46e3b7?logo=render">
</p>

<!-- Replace with your own GIF / screenshot -->
<p align="center">
  <img src="docs/demo.gif" width="700" alt="Demo GIF">
</p>

---

## ✨ Why?

* **There’s always** a Taylor song for every feeling. I just love her songs, and wanted to make something fun!
* Practice an **end-to-end GenAI pipeline**—embeddings, vector search, API.
* Minimal footprint, free-tier friendly, deployable in minutes.

---

## ⚙️ Tech Stack

| Component             | Purpose                               | Notes |
|-----------------------|---------------------------------------|-------|
| Taylor-Swift API      | Lyrics & metadata                     | <https://github.com/sarbor/taylor_swift_api> |
| **SentenceTransformers** | Text embeddings (768 dims)           | `all-mpnet-base-v2` |
| **Annoy / NumPy**     | Cosine search (pre-built index)       | Generated via `build_index.py` |
| **FastAPI**           | REST backend                          | End-point `/recommend` (Top-k) |
| **Render**            | Free hosting                          | Buildpacks, no Docker required |

---

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/<your-user>/swiftie-mood-agent.git
cd swiftie-mood-agent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1️⃣ Build embeddings once (~6 min, 208 songs)
python build_index.py

# 2️⃣ Run the API
uvicorn app.main:app --reload

# Docs → http://127.0.0.1:8000/docs

---

## 🚀 Architecture


[ user text ]              (write how you feel)
      |
      v
[ SBERT encoder ] -------->   vector u (1×768)
      |
      | dot-product (cosine)
      v
[ song vectors X ]  (208×768 preload)
      |
      v
top-k similarities
      |
  +---+---+
  | score < 0.15 ? |
  +---+---+
      | yes                    | no
      |                        |
“No match 😕”        title + lyrics + score



## 🌱 Mini Roadmap

 -React front-end (cards + audio preview).

 -Zero-shot emotion classifier to pre-filter lyrics.

 -Local lyrics cache (no external API calls in prod).

 -Docker + GitHub Actions CI/CD pipeline.


## Contributing
Pull-requests are welcome! 



