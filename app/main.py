"""
CommentPulse — FastAPI app.

Demo Mode: on load, the dashboard analyses the bundled sample dataset
(data/sample_comments.csv — a fictional pet grooming business).
Live Meta fetching is guarded behind META_ACCESS_TOKEN (see app/meta_api.py).
"""
import csv
import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from app.analyze import analyze
from app.meta_api import (
    NotConfigured,
    fetch_facebook_post_comments,
    fetch_instagram_media_comments,
    is_configured,
)
from app import auth as auth_module
from app.urlfetch import UrlError, fetch_by_url

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_comments.csv"
DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard.html"

app = FastAPI(title="CommentPulse", version="0.1.0")

auth_module.register(app)

_demo_cache = {}


def load_sample_comments():
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "text": row.get("text", ""),
                "timestamp": row.get("timestamp") or None,
                "platform": row.get("platform") or None,
                "author": row.get("author") or None,
            }
            for row in reader
            if (row.get("text") or "").strip()
        ]


def demo_analysis():
    if "result" not in _demo_cache:
        _demo_cache["result"] = analyze(load_sample_comments())
    return _demo_cache["result"]


class TextImport(BaseModel):
    comments: list


class MetaFetchRequest(BaseModel):
    platform: str  # "facebook" or "instagram"
    object_id: str  # post id (facebook) or media id (instagram)
    limit: int = 100
    access_token: str | None = None  # user's own token; used once, never stored


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@app.get("/api/demo")
def api_demo():
    return demo_analysis()


MAX_COMMENTS = 5000


@app.post("/api/import/text")
def import_text(payload: TextImport):
    comments = []
    for item in payload.comments:
        if isinstance(item, str):
            comments.append({"text": item})
        elif isinstance(item, dict):
            comments.append({
                "text": item.get("text", ""),
                "timestamp": item.get("timestamp"),
                "platform": item.get("platform"),
                "author": item.get("author"),
            })
    comments = [c for c in comments if str(c.get("text", "")).strip()]
    if not comments:
        return JSONResponse(status_code=400, content={
            "error": "No comments found. Paste at least one comment (one per line) and try again."})
    return analyze(comments[:MAX_COMMENTS])


@app.get("/grabber-extension.zip")
def grabber_extension():
    """Download the CommentPulse Grabber Chrome extension (zipped on the fly
    from app/extension/)."""
    import base64
    import zipfile
    ext_dir = BASE_DIR / "app" / "extension"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ext_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name == "icon.png.b64":
                # Icon ships base64-encoded in the repo (push tooling is
                # text-only); decode it into a real PNG inside the zip.
                z.writestr("commentpulse-grabber/icon.png",
                           base64.b64decode(f.read_text(encoding="utf-8")))
            else:
                z.write(f, arcname=f"commentpulse-grabber/{f.name}")
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=commentpulse-grabber-extension.zip"})


@app.get("/api/sample-csv")
def sample_csv():
    return FileResponse(SAMPLE_CSV, media_type="text/csv",
                        filename="commentpulse-sample.csv")


@app.post("/api/import/csv")
async def import_csv(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        return JSONResponse(status_code=400, content={
            "error": "File too large (max 5 MB). Split the export and try again."})
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return JSONResponse(status_code=400, content={
            "error": "That file has no header row. The first row must be: text,timestamp,platform,author"})
    headers = {h.strip().lower(): h for h in reader.fieldnames if h}
    if "text" not in headers:
        return JSONResponse(status_code=400, content={
            "error": "Your CSV needs a 'text' column with the comment text. "
                     "Expected columns: text,timestamp,platform,author — download the sample CSV to see the format."})
    text_col = headers["text"]
    ts_col = headers.get("timestamp")
    pf_col = headers.get("platform")
    au_col = headers.get("author")
    comments = [
        {
            "text": row.get(text_col, ""),
            "timestamp": (row.get(ts_col) or None) if ts_col else None,
            "platform": (row.get(pf_col) or None) if pf_col else None,
            "author": (row.get(au_col) or None) if au_col else None,
        }
        for row in reader
        if (row.get(text_col) or "").strip()
    ]
    if not comments:
        return JSONResponse(status_code=400, content={
            "error": "Found the 'text' column but no comments in it. Check the file has comment rows and try again."})
    return analyze(comments[:MAX_COMMENTS])


@app.post("/api/meta/fetch")
def meta_fetch(req: MetaFetchRequest):
    import httpx as _httpx
    try:
        if req.platform.lower() == "facebook":
            comments = fetch_facebook_post_comments(req.object_id, req.limit, token=req.access_token)
        elif req.platform.lower() == "instagram":
            comments = fetch_instagram_media_comments(req.object_id, req.limit, token=req.access_token)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "platform must be 'facebook' or 'instagram'"},
            )
    except NotConfigured as exc:
        return JSONResponse(status_code=501, content={"error": str(exc)})
    except _httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return JSONResponse(status_code=502, content={
            "error": "Meta rejected the request" + (f": {detail}" if detail else
                     f" (HTTP {exc.response.status_code}). Check the token and the post/media ID.")})
    except Exception as exc:
        return JSONResponse(status_code=502, content={
            "error": f"Could not fetch from Meta ({exc.__class__.__name__}). "
                     "Check the token and the post/media ID, then try again."})
    if not comments:
        return JSONResponse(status_code=404, content={
            "error": "Meta returned 0 comments for that ID. Check the ID is a post/media you own "
                     "and that the token has permission to read its comments."})
    return {"fetched": len(comments), "analysis": analyze(comments)}


@app.get("/api/meta/status")
def meta_status():
    return {"configured": is_configured()}


class UrlAnalyzeRequest(BaseModel):
    url: str


@app.post("/api/url/analyze")
def url_analyze(req: UrlAnalyzeRequest, request: Request):
    try:
        platform, object_id, comments = fetch_by_url(req.url)
    except UrlError as exc:
        return JSONResponse(status_code=exc.status, content={"error": exc.message})
    result = analyze(comments[:MAX_COMMENTS])
    label = f"{platform.title()} {object_id} ({len(comments)} comments)"
    user = auth_module.current_user(request)
    if user:
        from app import db as _db
        _db.save_analysis(user["id"], label, result)
    return {"fetched": len(comments), "platform": platform, "label": label, "analysis": result}


@app.get("/api/demo/whatsapp", response_class=PlainTextResponse)
def demo_whatsapp():
    a = demo_analysis()
    lines = ["*CommentPulse — comment summary*", ""]
    lines.append(a["gist"]["headline"])
    lines.append("")
    lines.append("Top topics:")
    for t in a["topics"][:3]:
        lines.append(f"- {t['label']}: {t['count']} comments ({t['share_pct']}%)")
    if a["overlap"]:
        o = a["overlap"][0]
        lines.append("")
        lines.append(f"Most repeated question ({o['count']}x, {o['theme']}):")
        lines.append(f'"{o["representative"]}"')
    preds = a["predictions"]
    rising = preds.get("rising_topics") or []
    if rising:
        lines.append("")
        lines.append("Rising (ESTIMATE): " + ", ".join(
            f"{r['topic']} (+{r['change_points']} pts)" for r in rising[:3]))
    act = a.get("actions") or {}
    leads = act.get("leads") or []
    if leads:
        lines.append("")
        lines.append(f"Leads in comments: {len(leads)} people showing buying intent.")
    objs = act.get("objections") or []
    if objs:
        o0 = objs[0]
        lines.append(f"Top objection: {o0['theme']} ({o0['count']} comments).")
    drafts = act.get("reply_drafts") or []
    if drafts:
        d0 = drafts[0]
        lines.append("")
        lines.append(f"Reply draft for the most repeated theme ({d0['theme']}, {d0['times_asked']}x):")
        lines.append(d0["draft"])
    return "\n".join(lines)


_LEGAL_STYLE = ("body{font-family:Georgia,serif;color:#18181b;max-width:760px;"
                "margin:32px auto;padding:0 20px;line-height:1.6}"
                "h1{font-size:25px}h2{font-size:17px;margin-top:22px}"
                "p,li{font-size:14.5px}")


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>CommentPulse — Privacy Policy</title><style>{_LEGAL_STYLE}</style></head><body>
<h1>CommentPulse — Privacy Policy</h1>
<p>Last updated: 6 November 2026. CommentPulse is operated by UpSkillz Solutions
(contact: digitalmarketingexpert1016@gmail.com).</p>
<h2>What we collect</h2>
<ul>
<li><strong>Account basics</strong> — when you log in with Facebook or Google we receive your
name, email address and profile picture from that provider.</li>
<li><strong>Social tokens</strong> — if you connect Facebook, your access token is stored
on our server so the app can list your Pages and fetch comments you ask it to analyse.
Tokens are never shown in the browser, never sold, and never shared.</li>
<li><strong>Comments you analyse</strong> — comments fetched from your own Pages/posts, or
that you paste, upload or link, are processed to produce your analysis. Analyses you run
while logged in are saved to your account history.</li>
</ul>
<h2>What we do not do</h2>
<ul>
<li>We do not sell personal data or use it for advertising.</li>
<li>We do not fetch data from accounts you have not connected or content you have not provided.</li>
<li>We do not post, message or change anything on your social accounts — access is read-only.</li>
</ul>
<h2>Your control</h2>
<p>You can log out at any time, disconnect Facebook in your Facebook settings
(Settings → Business integrations / Apps), and request full deletion of your account
data by emailing digitalmarketingexpert1016@gmail.com.</p>
<h2>Third parties</h2>
<p>Login and data fetching use Meta (Facebook/Instagram), Google/YouTube and Reddit
official APIs under their own policies. Hosting is provided by Render.</p>
</body></html>"""


@app.get("/terms", response_class=HTMLResponse)
def terms():
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>CommentPulse — Terms of Use</title><style>{_LEGAL_STYLE}</style></head><body>
<h1>CommentPulse — Terms of Use</h1>
<p>Last updated: 6 November 2026.</p>
<p>CommentPulse analyses comments from accounts you own or content you provide,
and produces topic, repetition, sentiment and trend reports. Predictions shown in
reports are estimates computed from the analysed data, not guarantees.</p>
<p>You are responsible for having the right to analyse any content you connect,
paste, upload or link. The service is provided "as is" without warranties; the
operator (UpSkillz Solutions) is not liable for decisions made from reports.
Contact: digitalmarketingexpert1016@gmail.com.</p>
</body></html>"""


@app.get("/report/demo", response_class=HTMLResponse)
def report_demo():
    a = demo_analysis()
    return render_report(a, "Paws & Shine Grooming — sample data (Demo Mode)")


def _esc(s):
    return (
        str(s if s is not None else "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_report(a, title):
    m, s = a["meta"], a["sentiment"]
    topics_html = "".join(
        f"""<div class="topic">
          <div class="row"><strong>{_esc(t['label'])}</strong>
          <span>{t['count']} comments · {t['share_pct']}%</span></div>
          <div class="bar"><div style="width:{t['share_pct']}%"></div></div>
          <div class="terms">Top terms: {_esc(', '.join(t['top_terms']))} ·
          {_esc(t['sentiment']['positive']['pct'])}% positive /
          {_esc(t['sentiment']['negative']['pct'])}% negative</div>
          {''.join(f'<p class="ex">&ldquo;{_esc(e)}&rdquo;</p>' for e in t['examples'])}
        </div>"""
        for t in a["topics"]
    )
    overlap_html = "".join(
        f"""<li><strong>{_esc(o['theme'])}</strong> — {o['count']} comments ({o['share_pct']}%)
        <br><span class="ex">&ldquo;{_esc(o['representative'])}&rdquo;</span></li>"""
        for o in a["overlap"]
    ) or "<li>No repeated themes found.</li>"
    gist_bullets = "".join(f"<li>{_esc(b)}</li>" for b in a["gist"]["bullets"])

    p = a["predictions"]
    pred_parts = []
    if p.get("rising_topics"):
        items = "".join(
            f"<li>{_esc(r['topic'])}: {r['earlier_share_pct']}% &rarr; {r['latest_week_share_pct']}% "
            f"of comments in the latest week (+{r['change_points']} pts)</li>"
            for r in p["rising_topics"]
        )
        pred_parts.append(f"<p><strong>Rising topics</strong> <span class='est'>ESTIMATE</span></p><ul>{items}</ul>")
    if p.get("sentiment_direction"):
        d = p["sentiment_direction"]
        pred_parts.append(
            f"<p><strong>Sentiment direction: {_esc(d['direction'])}</strong> <span class='est'>ESTIMATE</span>"
            f" — average sentiment {d['first_half_avg']} (first half) &rarr; {d['second_half_avg']} (second half).</p>"
        )
    if p.get("likely_questions"):
        items = "".join(
            f"<li>{_esc(q['question_theme'])} — asked {q['times_asked']} times, "
            f"e.g. &ldquo;{_esc(q['representative'])}&rdquo;</li>"
            for q in p["likely_questions"]
        )
        pred_parts.append(f"<p><strong>Likely to flood the next post</strong> <span class='est'>ESTIMATE</span></p><ul>{items}</ul>")
    if p.get("content_ideas"):
        items = "".join(f"<li>{_esc(c['idea'])}</li>" for c in p["content_ideas"])
        pred_parts.append(f"<p><strong>Content ideas</strong> <span class='est'>ESTIMATE</span></p><ul>{items}</ul>")
    predictions_html = "".join(pred_parts) or "<p>Not enough data for predictions.</p>"

    act = a.get("actions") or {}
    act_parts = []
    intents = act.get("intents") or {}
    if intents:
        pretty = {"buying_intent": "Buying intent", "question": "Questions",
                  "complaint": "Complaints", "praise": "Praise",
                  "spam": "Spam / self-promo", "other": "Other"}
        act_parts.append("<p>" + " · ".join(
            f"{pretty.get(k, k)}: {v['count']} ({v['pct']}%)" for k, v in intents.items()) + "</p>")
    leads = act.get("leads") or []
    if leads:
        items = "".join(
            f"<li>&ldquo;{_esc(l['text'])}&rdquo; — {_esc(l['author'])}</li>" for l in leads[:10])
        act_parts.append(f"<p><strong>Leads in comments ({len(leads)})</strong> — buying intent detected:</p><ul>{items}</ul>")
    objs = act.get("objections") or []
    if objs:
        items = "".join(
            f"<li><strong>{_esc(o['theme'])}</strong> — {o['count']} comments "
            f"({o['share_of_negative_pct']}% of negative)"
            + "".join(f"<br><span class='ex'>&ldquo;{_esc(e)}&rdquo;</span>" for e in o["examples"])
            + "</li>"
            for o in objs)
        act_parts.append(f"<p><strong>Objections</strong></p><ul>{items}</ul>")
    prio = act.get("priority") or []
    if prio:
        items = "".join(
            f"<li>&ldquo;{_esc(x['text'])}&rdquo; — {_esc(', '.join(x['reasons']))}</li>"
            for x in prio[:5])
        act_parts.append(f"<p><strong>Reply first</strong> (ranked by intent signals)</p><ul>{items}</ul>")
    drafts = act.get("reply_drafts") or []
    if drafts:
        items = "".join(
            f"<li><strong>{_esc(d['theme'])}</strong> ({d['times_asked']}x): {_esc(d['draft'])}</li>"
            for d in drafts)
        act_parts.append(f"<p><strong>Suggested reply drafts</strong> — fill the [brackets] with your real details before posting</p><ul>{items}</ul>")
    actions_html = "".join(act_parts) or "<p>No action items detected.</p>"

    date_line = ""
    if m.get("date_range"):
        date_line = f"Comments from {_esc(m['date_range']['from'])} to {_esc(m['date_range']['to'])}. "
    note_line = _esc(m.get("sample_note") or "")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CommentPulse Report — {_esc(title)}</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; color:#1d2733; max-width:820px; margin:32px auto; padding:0 20px; line-height:1.5; }}
  h1 {{ font-size:26px; margin-bottom:2px; }} h2 {{ font-size:19px; margin-top:28px; border-bottom:2px solid #e4572e; padding-bottom:4px; }}
  .sub {{ color:#5b6b7c; margin-top:0; }}
  .row {{ display:flex; justify-content:space-between; font-size:14px; }}
  .bar {{ height:12px; background:#e4eaf0; border-radius:6px; overflow:hidden; margin:4px 0; }}
  .bar > div {{ height:100%; background:#e4572e; }}
  .terms, .ex {{ color:#44576c; font-size:13px; }} .ex {{ font-style:italic; margin:2px 0 2px 12px; }}
  .topic {{ margin-bottom:14px; }} li {{ margin-bottom:8px; }}
  .est {{ background:#fff3cd; color:#856404; border:1px solid #ffe08a; border-radius:4px; font-size:11px; font-weight:700; padding:1px 6px; }}
  .stats {{ display:flex; gap:24px; flex-wrap:wrap; }} .stats div b {{ font-size:22px; display:block; }}
  @media print {{ body {{ margin:10px auto; }} }}
</style></head><body>
<h1>CommentPulse Report</h1>
<p class="sub">{_esc(title)}</p>
<div class="stats">
  <div><b>{m['total_comments']}</b>comments analysed</div>
  <div><b>{len(a['topics'])}</b>topics found</div>
  <div><b>{m['avg_sentiment']}</b>avg sentiment (−1 to +1)</div>
  <div><b>{s['positive']['pct']}% / {s['neutral']['pct']}% / {s['negative']['pct']}%</b>positive / neutral / negative</div>
</div>
<p class="sub">{date_line}{note_line}</p>

<h2>Lub-e-lubab — the gist</h2>
<p><strong>{_esc(a['gist']['headline'])}</strong></p>
<ul>{gist_bullets}</ul>

<h2>Topics discussed</h2>
{topics_html}

<h2>Overlap — what keeps repeating</h2>
<ul>{overlap_html}</ul>

<h2>Predictions</h2>
{predictions_html}

<h2>Action centre</h2>
{actions_html}

<p class="sub">Method: {_esc(m['method'])}. All figures are computed from the comments analysed; predictions are estimates based on the patterns stated.</p>
</body></html>"""
