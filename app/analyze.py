"""
CommentPulse — analysis engine.

Every number produced here is computed from the input comments. Nothing is
invented: the gist is templated from computed clusters/sentiment, and every
prediction is labelled ESTIMATE.
"""
import re
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")

# Keyword sets used to map a cluster / overlap group to a human label.
# The raw top TF-IDF terms are always returned alongside, so the mapping is
# never hidden.
LABEL_KEYWORDS = {
    "Price questions": [
        "price", "prices", "cost", "costs", "charge", "charges", "charged",
        "much", "quote", "rate", "rates", "fee", "fees", "expensive",
        "discount", "steep", "penny",
    ],
    "Booking & appointments": [
        "book", "booking", "booked", "appointment", "appointments",
        "schedule", "slot", "slots", "reserve", "reservation", "walk",
        "walk-in", "walk-ins", "availability", "available",
    ],
    "Complaints": [
        "reply", "replies", "replied", "waiting", "waited", "wait", "slow",
        "disappointed", "terrible", "frustrating", "ignored", "worst",
        "unhappy", "cancel", "cancelled", "answer", "answered", "response",
        "nicked", "uneven", "shorter",
    ],
    "Praise & love": [
        "love", "loved", "amazing", "wonderful", "best", "great", "thank",
        "thanks", "happy", "perfect", "lovely", "brilliant", "fantastic",
        "excellent", "beautiful", "adorable", "grateful", "obsessed",
        "incredible", "talented", "gentle", "patient",
    ],
    "Hours & location": [
        "hours", "open", "opens", "close", "closes", "closed", "sunday",
        "sundays", "saturday", "monday", "weekdays", "located", "location",
        "parking", "holidays", "time",
    ],
    "Delivery & pickup": [
        "delivery", "deliver", "pickup", "drop", "dropoff", "shipping",
        "collect", "collection",
    ],
    "Services & grooming details": [
        "groom", "grooming", "bath", "nail", "nails", "haircut", "cut",
        "deshedding", "teeth", "shampoo", "trim", "flea", "ears", "wash",
        "blow", "dry", "package", "treatment",
    ],
    "Spam / self-promo": [
        "followers", "follow", "crypto", "iphone", "win", "click", "bio",
        "earn", "money", "guaranteed", "likes", "deals",
    ],
}

QUESTION_STARTERS = (
    "how", "what", "can", "do", "does", "is", "are", "when", "where",
    "price", "prices", "want", "need", "looking", "any",
)

# Intent labels checked in priority order for overlap groups: subject words
# ("groom", "bath") appear in almost every comment for a grooming business,
# so intent is judged by how many of the group's comments carry the label's
# keywords — not by raw word counts.
INTENT_PRIORITY = [
    "Spam / self-promo", "Price questions", "Booking & appointments",
    "Complaints", "Praise & love", "Hours & location", "Delivery & pickup",
    "Services & grooming details",
]


def _intent_label(member_texts, fallback_terms):
    """Label an overlap group by keyword coverage across its comments."""
    if not member_texts:
        return f"Topic: {', '.join(fallback_terms[:3])}"
    coverage = {}
    for label in INTENT_PRIORITY:
        keywords = LABEL_KEYWORDS[label]
        hits = sum(
            1 for t in member_texts
            if any(re.search(r"\b" + re.escape(kw) + r"\b", t.lower()) for kw in keywords)
        )
        coverage[label] = hits / len(member_texts)
    for label in INTENT_PRIORITY:
        if coverage[label] >= 0.34:
            return label
    best = max(coverage, key=coverage.get)
    if coverage[best] >= 0.25:
        return best
    return f"Topic: {', '.join(fallback_terms[:3])}"


def clean_text(text: str) -> str:
    """Strip URLs and mentions for analysis; originals are kept for citation."""
    t = URL_RE.sub(" ", text or "")
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()


def _label_for(terms, sample_texts):
    """Score keyword sets against a group's terms + sampled text."""
    hay_terms = set(terms)
    hay_text = " ".join(sample_texts).lower()
    best_label, best_score = None, 0
    for label, keywords in LABEL_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in hay_terms:
                score += 2
            if re.search(r"\b" + re.escape(kw) + r"\b", hay_text):
                score += 1
        if score > best_score:
            best_label, best_score = label, score
    if best_label is None or best_score == 0:
        top = ", ".join(terms[:3]) if terms else "general"
        return f"Topic: {top}"
    return best_label


def _sentiment_bucket(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _pct(count, total):
    return round(100.0 * count / total, 1) if total else 0.0


def _split_sentiment(buckets, total):
    return {
        name: {"count": buckets.get(name, 0), "pct": _pct(buckets.get(name, 0), total)}
        for name in ("positive", "neutral", "negative")
    }


def analyze(comments):
    """
    comments: list of dicts with keys text, timestamp (optional),
    platform (optional), author (optional).
    Returns a JSON-serialisable analysis dict.
    """
    raw = [c for c in comments if c and str(c.get("text", "")).strip()]
    total = len(raw)
    if total == 0:
        return {
            "meta": {"total_comments": 0, "platforms": {}, "date_range": None,
                     "avg_sentiment": 0.0, "sample_note": "No comments supplied."},
            "sentiment": _split_sentiment(Counter(), 0),
            "topics": [], "overlap": [],
            "gist": {"headline": "No comments to analyse.", "bullets": [], "text": "No comments to analyse."},
            "predictions": {"rising_topics": [], "likely_questions": [],
                            "sentiment_direction": None, "content_ideas": []},
        }

    originals = [str(c["text"]).strip() for c in raw]
    cleaned = [clean_text(t) for t in originals]
    platforms = Counter(str(c.get("platform") or "unknown").lower() for c in raw)
    timestamps = [_parse_ts(c.get("timestamp")) for c in raw]

    # ---- Sentiment (VADER) ----
    analyzer = SentimentIntensityAnalyzer()
    compounds = [analyzer.polarity_scores(t)["compound"] for t in cleaned]
    buckets = Counter(_sentiment_bucket(s) for s in compounds)
    avg_sentiment = round(float(np.mean(compounds)), 3) if compounds else 0.0

    # ---- Topic clustering (TF-IDF + KMeans) ----
    topics = []
    cluster_of = [0] * total
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([t.lower() for t in cleaned])
        if matrix.shape[1] == 0:
            raise ValueError("empty vocabulary")
    except ValueError:
        matrix = None

    if matrix is None:
        topics = [{
            "id": 0, "label": "General comments", "top_terms": [],
            "count": total, "share_pct": 100.0,
            "sentiment": _split_sentiment(buckets, total),
            "examples": originals[:2],
        }]
    else:
        k = int(round(total / 15))
        k = max(1, min(10, k)) if total < 3 else max(3, min(10, k))
        k = min(k, total)
        if k == 1:
            labels = np.zeros(total, dtype=int)
            centers_terms = [[]]
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(matrix)
            feature_names = vectorizer.get_feature_names_out()
            centers_terms = []
            for center in km.cluster_centers_:
                top_idx = center.argsort()[::-1][:6]
                centers_terms.append([feature_names[i] for i in top_idx if center[i] > 0])
        cluster_of = [int(x) for x in labels]

        for cid in range(k):
            members = [i for i in range(total) if cluster_of[i] == cid]
            if not members:
                continue
            terms = centers_terms[cid] if cid < len(centers_terms) else []
            unigrams = [t for t in terms if " " not in t]
            sample_texts = [cleaned[i] for i in members[:8]]
            label = _label_for(unigrams, sample_texts)
            cbuckets = Counter(_sentiment_bucket(compounds[i]) for i in members)
            # Examples: most positive then most negative member, else first two.
            by_score = sorted(members, key=lambda i: compounds[i], reverse=True)
            examples = []
            for i in by_score[:1] + by_score[-1:]:
                if originals[i] not in examples:
                    examples.append(originals[i])
            topics.append({
                "id": cid,
                "label": label,
                "top_terms": terms[:6],
                "count": len(members),
                "share_pct": _pct(len(members), total),
                "sentiment": _split_sentiment(cbuckets, len(members)),
                "examples": examples[:2],
            })
        topics.sort(key=lambda t: t["count"], reverse=True)

    # De-duplicate labels: if two clusters map to the same human label, suffix.
    seen_labels = Counter()
    for t in topics:
        seen_labels[t["label"]] += 1
        if seen_labels[t["label"]] > 1:
            t["label"] = f'{t["label"]} ({seen_labels[t["label"]]})'

    # ---- Overlap: near-duplicate / repeated-intent groups ----
    # Dedicated unigram TF-IDF space, then seed-and-prune grouping on cosine
    # similarity: grow a group from its densest comment, then drop drifters
    # whose mean similarity to the rest of the group is too low.
    overlap = []
    if total >= 3:
        try:
            ovec = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
            omatrix = ovec.fit_transform([t.lower() for t in cleaned])
            if omatrix.shape[1] == 0:
                raise ValueError("empty vocabulary")
        except ValueError:
            omatrix = None
        if omatrix is not None:
            osim = cosine_similarity(omatrix)
            np.fill_diagonal(osim, 0.0)
            threshold, prune_floor = 0.35, 0.18
            remaining = set(range(total))
            raw_groups = []
            while remaining:
                rem = sorted(remaining)
                seed = max(rem, key=lambda i: sum(
                    1 for j in rem if j != i and osim[i, j] >= threshold))
                grp = [j for j in rem if j == seed or osim[seed, j] >= threshold]
                if len(grp) < 3:
                    break
                changed = True
                while changed and len(grp) >= 3:
                    changed = False
                    for j in list(grp):
                        if j == seed:
                            continue
                        mean_sim = float(np.mean([osim[j, k] for k in grp if k != j]))
                        if mean_sim < prune_floor:
                            grp.remove(j)
                            changed = True
                if len(grp) >= 3:
                    raw_groups.append(grp)
                    remaining -= set(grp)
                else:
                    remaining.discard(seed)
                if len(raw_groups) >= 8:
                    break
            for grp in raw_groups:
                rep = max(grp, key=lambda i: float(np.mean(
                    [osim[i, j] for j in grp if j != i])))
                group_terms = Counter()
                for i in grp:
                    for tok in set(re.findall(r"[a-z']+", cleaned[i].lower())):
                        group_terms[tok] += 1
                top_terms = [t for t, _ in group_terms.most_common(6)]
                theme = _intent_label([cleaned[i] for i in grp], top_terms)
                examples = [originals[rep]] + [originals[i] for i in grp if i != rep][:1]
                overlap.append({
                    "theme": theme,
                    "representative": originals[rep],
                    "count": len(grp),
                    "share_pct": _pct(len(grp), total),
                    "top_terms": top_terms,
                    "examples": examples,
                })
            overlap.sort(key=lambda o: o["count"], reverse=True)
            seen_themes = Counter()
            for o in overlap:
                seen_themes[o["theme"]] += 1
                if seen_themes[o["theme"]] > 1:
                    o["theme"] = f'{o["theme"]} ({seen_themes[o["theme"]]})'

    # ---- Gist (lub-e-lubab): templated strictly from computed data ----
    gist = _build_gist(total, buckets, avg_sentiment, topics, overlap,
                       [originals[i] for i in range(total)], cluster_of, compounds)

    # ---- Predictions ----
    predictions = _build_predictions(topics, overlap, timestamps, cluster_of,
                                     compounds, originals)

    valid_ts = [t for t in timestamps if t is not None]
    date_range = None
    if valid_ts:
        date_range = {"from": min(valid_ts).date().isoformat(),
                      "to": max(valid_ts).date().isoformat()}

    sample_note = None
    if total < 20:
        sample_note = ("Small sample: fewer than 20 comments — clusters are thin, "
                       "treat results as indicative only.")

    return {
        "meta": {
            "total_comments": total,
            "platforms": dict(platforms),
            "date_range": date_range,
            "avg_sentiment": avg_sentiment,
            "sample_note": sample_note,
            "method": "TF-IDF + KMeans clustering, TF-IDF cosine overlap, VADER sentiment",
        },
        "sentiment": _split_sentiment(buckets, total),
        "topics": topics,
        "overlap": overlap,
        "gist": gist,
        "predictions": predictions,
    }


def _build_gist(total, buckets, avg_sentiment, topics, overlap, originals,
                cluster_of, compounds):
    pos = buckets.get("positive", 0)
    neg = buckets.get("negative", 0)
    if pos >= neg * 2 and pos > 0:
        tone = "mostly positive"
    elif neg > pos:
        tone = "mostly negative"
    else:
        tone = "mixed"
    headline = (
        f"Across {total} comments the conversation is {tone}: "
        f"{_pct(pos, total)}% positive, {_pct(buckets.get('neutral', 0), total)}% neutral, "
        f"{_pct(neg, total)}% negative (average sentiment score {avg_sentiment})."
    )

    bullets = []
    if topics:
        top = topics[0]
        b = (f"Biggest topic: {top['label']} — {top['count']} comments "
             f"({top['share_pct']}% of all comments).")
        if top["examples"]:
            b += f' Example: "{top["examples"][0]}"'
        bullets.append(b)

        praise = next((t for t in topics if t["label"].startswith("Praise")), None)
        if praise is None:
            praise = max(topics, key=lambda t: t["sentiment"]["positive"]["pct"])
        if praise["sentiment"]["positive"]["count"] > 0:
            b = (f"What people loved: {praise['label']} — "
                 f"{praise['sentiment']['positive']['count']} of its {praise['count']} comments are positive.")
            if praise["examples"]:
                b += f' Example: "{praise["examples"][0]}"'
            bullets.append(b)

        complaint = next((t for t in topics if t["label"].startswith("Complaints")), None)
        if complaint is None:
            complaint = max(topics, key=lambda t: t["sentiment"]["negative"]["pct"])
        if complaint["sentiment"]["negative"]["count"] > 0:
            b = (f"Main complaint theme: {complaint['label']} — "
                 f"{complaint['sentiment']['negative']['count']} negative comments in this theme.")
            ex = complaint["examples"]
            if ex:
                b += f' Example: "{ex[-1]}"'
            bullets.append(b)

    if overlap:
        top_o = overlap[0]
        b = (f"Biggest repeated demand: {top_o['count']} comments ask versions of the same thing "
             f"({top_o['theme']}). Representative comment: \"{top_o['representative']}\"")
        bullets.append(b)

    text = headline + " " + " ".join(bullets)
    return {"headline": headline, "bullets": bullets, "text": text}


def _build_predictions(topics, overlap, timestamps, cluster_of, compounds, originals):
    preds = {"rising_topics": [], "likely_questions": [],
             "sentiment_direction": None, "content_ideas": []}

    # --- Week-over-week topic share change (needs timestamps) ---
    dated = [(timestamps[i], cluster_of[i], i) for i in range(len(timestamps))
             if timestamps[i] is not None]
    if len(dated) >= 10:
        weeks = sorted({(t.isocalendar()[0], t.isocalendar()[1]) for t, _, _ in dated})
        if len(weeks) >= 2:
            latest_week = weeks[-1]
            latest_idx = [i for t, c, i in dated if (t.isocalendar()[0], t.isocalendar()[1]) == latest_week]
            earlier_idx = [i for t, c, i in dated if (t.isocalendar()[0], t.isocalendar()[1]) != latest_week]
            cluster_topic = {}
            # cluster id -> topic (topic ids came from cluster ids)
            topic_by_id = {t["id"]: t for t in topics}
            for t, c, i in dated:
                cluster_topic[c] = topic_by_id.get(c)
            for cid, topic in cluster_topic.items():
                if topic is None:
                    continue
                latest_share = _pct(sum(1 for i in latest_idx if cluster_of[i] == cid), len(latest_idx)) if latest_idx else 0.0
                earlier_share = _pct(sum(1 for i in earlier_idx if cluster_of[i] == cid), len(earlier_idx)) if earlier_idx else 0.0
                delta = round(latest_share - earlier_share, 1)
                if delta >= 3.0 and sum(1 for i in latest_idx if cluster_of[i] == cid) >= 3:
                    preds["rising_topics"].append({
                        "topic": topic["label"],
                        "earlier_share_pct": earlier_share,
                        "latest_week_share_pct": latest_share,
                        "change_points": delta,
                        "label": "ESTIMATE",
                        "basis": "Week-over-week share of comments in this topic.",
                    })
            preds["rising_topics"].sort(key=lambda r: -r["change_points"])

        # --- Sentiment direction: first half vs second half of the timeline ---
        ordered = sorted(dated, key=lambda x: x[0])
        mid = len(ordered) // 2
        if mid >= 5:
            first_avg = float(np.mean([compounds[i] for _, _, i in ordered[:mid]]))
            second_avg = float(np.mean([compounds[i] for _, _, i in ordered[mid:]]))
            diff = round(second_avg - first_avg, 3)
            direction = "rising" if diff > 0.05 else "falling" if diff < -0.05 else "steady"
            preds["sentiment_direction"] = {
                "direction": direction,
                "first_half_avg": round(first_avg, 3),
                "second_half_avg": round(second_avg, 3),
                "change": diff,
                "label": "ESTIMATE",
                "basis": "Average VADER sentiment, first half vs second half of the comment timeline.",
            }

    # --- Likely next-post questions: top repeated question-form groups ---
    for group in overlap:
        rep = group["representative"].strip()
        is_question = rep.endswith("?") or rep.lower().startswith(QUESTION_STARTERS)
        if is_question:
            preds["likely_questions"].append({
                "question_theme": group["theme"],
                "representative": rep,
                "times_asked": group["count"],
                "label": "ESTIMATE",
                "basis": "Most repeated unanswered-style questions in the current comments.",
            })
        if len(preds["likely_questions"]) >= 3:
            break

    # --- Content ideas: drawn directly from top overlap + top topic ---
    if overlap:
        g = overlap[0]
        preds["content_ideas"].append({
            "idea": (f"A post that answers the repeated question behind \"{g['representative']}\" "
                     f"would address a theme raised {g['count']} times."),
            "label": "ESTIMATE",
            "basis": "Derived from the #1 overlap group.",
        })
    if topics:
        t = topics[0]
        preds["content_ideas"].append({
            "idea": (f"More content on {t['label'].lower()} — it is the largest topic at "
                     f"{t['count']} comments ({t['share_pct']}%)."),
            "label": "ESTIMATE",
            "basis": "Derived from the largest topic cluster.",
        })

    return preds
