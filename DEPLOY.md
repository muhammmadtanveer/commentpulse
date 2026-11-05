# CommentPulse — See it, Test it, Deploy it, Change it

## 1. See it RIGHT NOW (no install, 10 seconds)
Open `demo-report.html` in any browser — it's the finished demo report
(148 sample comments analysed). Static file, works offline.
The live dashboard needs the server running (options 2 or 3).

## 2. Test it on your own laptop (5 minutes, step by step)
You need Python 3.10+ installed. Run commands ONE at a time, in the
`commentpulse` folder — not in your home folder.

**Mac shortcut:** in the commentpulse folder just run `bash run-mac.sh`
— it does all steps below for you.

**Mac (manual — use `python3`, plain `python` does not exist on Mac):**
```bash
cd ~/Downloads/commentpulse
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --port 8100
```
If `python3 --version` also says "command not found", install Python
from https://www.python.org/downloads/ first, then repeat.

**Windows (manual):**
```bash
cd %USERPROFILE%\Downloads\commentpulse
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8100
```

Open **http://localhost:8100** — dashboard loads in Demo Mode.
Try these:
- http://localhost:8100/report/demo — printable report
- http://localhost:8100/api/demo/whatsapp — WhatsApp-ready summary
- Upload your own comments: dashboard CSV import, columns `text,timestamp,platform,author`

Stop the server: `Ctrl + C`.

## 3. Deploy it LIVE — DONE 2026-11-05
**Live: https://commentpulse-86rh.onrender.com** (Render free tier, Docker,
Ohio region; sleeps when idle — first load after sleep can take ~50s).
Repo: https://github.com/muhammmadtanveer/commentpulse — every push to
`main` auto-redeploys. Original deploy notes below.

## 3b. How the deploy was set up — recommended: Render
Why Render, honestly:
- **Vercel is a bad fit here** — CommentPulse uses scikit-learn (heavy Python);
  Vercel's serverless size limits will likely break it. Vercel was fine for
  Freelance Dost (light), not for this.
- Render / Railway / Hugging Face Spaces run it as a real Docker server.
  This repo now ships a `Dockerfile` + `render.yaml` so Render needs no config.

Steps (about 10 minutes, free tier):
1. Push this folder to a GitHub repo named `commentpulse` (see section 4).
2. Go to render.com → New → Web Service → connect that repo.
3. Render reads `render.yaml` automatically → Deploy.
4. You get a public link like `https://commentpulse.onrender.com`
   Free tier sleeps after ~15 min idle; first open after sleep takes ~30–50s.
5. (Later, for real FB/IG data) add `META_ACCESS_TOKEN` in Render →
   Environment — never put the token in code or GitHub.

Alternatives: Railway (same Dockerfile, small monthly credit) or
Hugging Face Spaces (free, Docker SDK, good for portfolio demos).

## 4. Make DIRECT changes yourself
The flow that gives you full control:
1. Code lives on **GitHub** (`github.com/muhammmadtanveer/commentpulse` — to be created).
2. Edit anywhere:
   - Quick edit: GitHub website → open file → pencil icon → Commit.
   - Proper edit: clone the repo in VS Code, change, test with option 2, push.
3. Render auto-redeploys on every push — your live link updates in ~2–3 min.
No other step. Change → push → live.

What lives where:
- `app/analyze.py` — the brain (topics, overlap, sentiment, gist, predictions)
- `app/dashboard.html` — what the dashboard looks like
- `app/main.py` — routes / URLs
- `app/meta_api.py` — Facebook/Instagram fetching
- `data/sample_comments.csv` — demo dataset (swap this to change the demo)

## Note on Muse's copy
Muse's working copy is at `~/workspace/commentpulse/` in Muse's workspace —
you can ask Muse to change anything and Muse can also run/deploy it for you,
but for YOUR direct changes, GitHub (section 4) is your copy.
