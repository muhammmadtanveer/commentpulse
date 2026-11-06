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

# --- Competitive-upgrade lexicons (all rules are transparent keyword sets) ---

# Roman Urdu / Urdu sentiment assist: VADER is English-only, so common
# Roman-Urdu praise/complaint phrases adjust the compound score by a bounded
# amount. Every hit is visible in the words themselves; nothing is inferred.
URDU_POSITIVE = [
    "bohat acha", "bohat achha", "boht acha", "bohut acha", "acha hai",
    "bohat acha hai", "zabardast", "behtareen", "behtreen", "mashallah",
    "masha allah", "khubsurat", "shukriya", "bohat shukriya", "pasand aya",
    "pasand aaya", "bohat pyara", "kamaal", "kamal", "aala", "alaa",
    "khush hua", "dil khush", "maza aya", "maza aaya",
]
URDU_NEGATIVE = [
    "bohat bura", "boht bura", "bohut bura", "bura hai", "bakwas",
    "bekar", "ghatiya", "fazool", "faltu", "kharab", "dhoka", "dhokha",
    "loot liya", "loot raha", "naraz", "shikayat", "ganda", "bura laga",
    "pasand nahi", "acha nahi", "achha nahi", "mehnga bohat", "bohat mehnga",
]
URDU_SINGLE_POS = {"acha", "achha", "zabardast", "behtareen", "shukriya", "khubsurat"}
URDU_SINGLE_NEG = {"bakwas", "bekar", "ghatiya", "fazool", "kharab", "dhoka"}

# Buying-intent: commercial keywords + a question form or a strong verb.
BUY_KEYWORDS = [
    "price", "prices", "cost", "much", "rate", "rates", "charge", "charges",
    "order", "buy", "book", "booking", "appointment", "available",
    "delivery", "dm", "inbox", "contact", "whatsapp", "discount", "offer",
    "quote", "charges", "fees",
]
BUY_STRONG = [
    "want to", "i want", "need this", "i need", "order karna", "book karna",
    "interested", "book me", "sign me up", "how to order", "how can i order",
    "lena hai", "chahiye", "buy karna",
]
QUESTION_HINTS = [
    "anyone know", "does anyone", "can anyone", "is there", "do you",
    "kya", "kia", "kb", "kab", "kahan", "kaise", "kitne", "kitna",
]

# Objection sub-themes, checked in order against complaint/negative comments.
OBJECTION_THEMES = [
    ("Price & value", ["expensive", "overpriced", "too much", "steep", "costly",
                        "pricey", "rip off", "ripoff", "loot", "mehnga"]),
    ("Slow response / waiting", ["no reply", "not replying", "no response",
                                  "ignored", "waiting", "waited", "slow reply",
                                  "reply nahi", "jawab nahi", "late reply"]),
    ("Service quality", ["rude", "unprofessional", "bad service", "poor service",
                          "worst service", "pathetic", "bakwas service",
                          "staff behaviour", "staff behavior"]),
    ("Work quality", ["uneven", "nicked", "ruined", "damaged", "bad job",
                       "poor quality", "kharab kar", "ghatiya kaam"]),
    ("Trust concerns", ["scam", "fraud", "fake", "cheat", "dhoka", "dhokha",
                         "fraud hai", "scam hai"]),
]

# Reply-draft templates per repeated-theme label. Bracketed parts are
# placeholders the owner fills with their REAL details — drafts never invent
# prices, hours or policies.
REPLY_DRAFT_TEMPLATES = [
    ("Price questions",
     "Hi! Thanks for asking about prices. [Add your price or starting range here] "
     "— send us a DM and we'll confirm the exact quote for you."),
    ("Booking & appointments",
     "Hi! We'd love to book you in. [Add your booking link or next available days here] "
     "— or DM us and we'll reserve your slot."),
    ("Hours & location",
     "Hi! Our hours are [add your opening hours here] and we're at "
     "[add your location here]. See you soon!"),
    ("Delivery & pickup",
     "Hi! [Add your delivery areas and charges here]. DM us your location "
     "and we'll confirm right away."),
    ("Complaints",
     "Hi, we're really sorry about this. [Add one line on how you fix it / your "
     "policy here] — please DM us so we can make it right."),
]
REPLY_DRAFT_GENERIC = (
    "Great question! [Add your answer here] — and feel free to DM us if you "
    "need anything else."
)
REPLY_DRAFT_NOTE = ("Draft only — fill the [brackets] with your real details "
                    "before posting.")

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
            "comments": [],
            "sentiment": _split_sentiment(Counter(), 0),
            "topics": [], "overlap": [],
            "gist": {"headline": "No comments to analyse.", "bullets": [], "text": "No comments to analyse."},
            "predictions": {"rising_topics": [], "likely_questions": [],
                            "sentiment_direction": None, "content_ideas": []},
            "actions": {"intents": {}, "leads": [], "objections": [],
                        "priority": [], "reply_drafts": []},
        }

    originals = [str(c["text"]).strip() for c in raw]
    cleaned = [clean_text(t) for t in originals]
    platforms = Counter(str(c.get("platform") or "unknown").lower() for c in raw)
    timestamps = [_parse_ts(c.get("timestamp")) for c in raw]

    # ---- Sentiment (VADER + Roman-Urdu lexicon assist) ----
    analyzer = SentimentIntensityAnalyzer()
    compounds = [
        _urdu_assist(t, analyzer.polarity_scores(t)["compound"]) for t in cleaned
    ]
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
    predictions["based_on"] = {
        "comments": total,
        "dated_comments": sum(1 for t in timestamps if t is not None),
        "note": ("ESTIMATE — computed only from the comments in this analysis; "
                 "larger samples give steadier estimates."),
    }

    # ---- Intents + Action Centre (leads, objections, priority, drafts) ----
    intents = [
        _classify_intent(cleaned[i], originals[i], compounds[i]) for i in range(total)
    ]
    actions = _build_actions(intents, originals, raw, timestamps, compounds, overlap)

    valid_ts = [t for t in timestamps if t is not None]
    date_range = None
    if valid_ts:
        date_range = {"from": min(valid_ts).date().isoformat(),
                      "to": max(valid_ts).date().isoformat()}

    sample_note = None
    if total < 20:
        sample_note = ("Small sample: fewer than 20 comments — clusters are thin, "
                       "treat results as indicative only.")

    # ---- Per-comment rows (for the dashboard table / volume chart) ----
    topic_label_by_id = {t["id"]: t["label"] for t in topics}
    comments_out = []
    for i in range(total):
        ts = timestamps[i]
        comments_out.append({
            "id": i + 1,
            "text": originals[i],
            "author": raw[i].get("author") or "unknown",
            "platform": str(raw[i].get("platform") or "unknown").lower(),
            "timestamp": ts.isoformat() if ts is not None else raw[i].get("timestamp"),
            "sentiment": _sentiment_bucket(compounds[i]),
            "sentiment_score": round(float(compounds[i]), 3),
            "topic": topic_label_by_id.get(cluster_of[i], "General comments"),
            "intent": intents[i],
        })

    return {
        "meta": {
            "total_comments": total,
            "platforms": dict(platforms),
            "date_range": date_range,
            "avg_sentiment": avg_sentiment,
            "sample_note": sample_note,
            "method": ("TF-IDF + KMeans clustering, TF-IDF cosine overlap, VADER "
                       "sentiment with Roman-Urdu lexicon assist, rule-based intent"),
        },
        "sentiment": _split_sentiment(buckets, total),
        "topics": topics,
        "overlap": overlap,
        "gist": gist,
        "predictions": predictions,
        "actions": actions,
        "comments": comments_out,
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


def _urdu_assist(text: str, compound: float) -> float:
    """Bounded Roman-Urdu/Urdu lexicon adjustment to a VADER compound score.

    Phrase hits move the score by 0.3 each (bounded to ±0.6 total), single
    strong words by 0.2. Result is clamped to [-1, 1]. English-only comments
    are untouched unless they contain one of the listed words.
    """
    low = " " + (text or "").lower() + " "
    delta = 0.0
    for phrase in URDU_POSITIVE:
        if phrase in low:
            delta += 0.3
    for phrase in URDU_NEGATIVE:
        if phrase in low:
            delta -= 0.3
    tokens = set(re.findall(r"[a-z']+", low))
    for w in URDU_SINGLE_POS & tokens:
        delta += 0.2
    for w in URDU_SINGLE_NEG & tokens:
        delta -= 0.2
    delta = max(-0.6, min(0.6, delta))
    return max(-1.0, min(1.0, compound + delta))


def _is_question(original: str, low: str) -> bool:
    if "?" in original:
        return True
    stripped = low.strip()
    if any(stripped.startswith(s + " ") or stripped == s for s in QUESTION_STARTERS):
        return True
    return any(h in low for h in QUESTION_HINTS)


def _has_any(low: str, phrases) -> bool:
    return any(p in low for p in phrases)


def _has_word(low: str, words) -> bool:
    return any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words)


def _classify_intent(cleaned: str, original: str, compound: float) -> str:
    """One intent per comment, checked in a fixed priority order."""
    low = cleaned.lower()
    bucket = _sentiment_bucket(compound)
    if _has_word(low, LABEL_KEYWORDS["Spam / self-promo"]):
        return "spam"
    question = _is_question(original, low)
    if _has_word(low, BUY_KEYWORDS) and (question or _has_any(low, BUY_STRONG)):
        return "buying_intent"
    if _has_any(low, BUY_STRONG):
        return "buying_intent"
    complaint_hit = _has_word(low, LABEL_KEYWORDS["Complaints"])
    if bucket == "negative" and (complaint_hit or compound <= -0.45):
        return "complaint"
    if question:
        return "question"
    if bucket == "positive" and (_has_word(low, LABEL_KEYWORDS["Praise & love"])
                                  or compound >= 0.6):
        return "praise"
    return "other"


def _build_actions(intents, originals, raw, timestamps, compounds, overlap):
    total = len(originals)
    counts = Counter(intents)
    intent_summary = {
        name: {"count": counts.get(name, 0), "pct": _pct(counts.get(name, 0), total)}
        for name in ("buying_intent", "question", "complaint", "praise", "spam", "other")
        if counts.get(name, 0) > 0
    }

    def row(i):
        ts = timestamps[i]
        return {
            "id": i + 1,
            "author": raw[i].get("author") or "unknown",
            "platform": str(raw[i].get("platform") or "unknown").lower(),
            "text": originals[i],
            "timestamp": ts.isoformat() if ts is not None else raw[i].get("timestamp"),
        }

    # Leads: buying-intent comments, newest first when dates exist.
    lead_idx = [i for i in range(total) if intents[i] == "buying_intent"]
    lead_idx.sort(key=lambda i: (timestamps[i] is not None, timestamps[i]),
                  reverse=True)
    leads = [row(i) for i in lead_idx[:20]]

    # Objections: complaint-intent + remaining negative comments grouped by
    # sub-theme keywords; examples are real comments from the group.
    neg_idx = [i for i in range(total)
               if intents[i] == "complaint" or _sentiment_bucket(compounds[i]) == "negative"]
    neg_total = max(1, len(neg_idx))
    groups = {}
    for i in neg_idx:
        low = clean_text(originals[i]).lower()
        theme = "General complaints"
        for name, kws in OBJECTION_THEMES:
            if _has_any(low, kws):
                theme = name
                break
        groups.setdefault(theme, []).append(i)
    objections = []
    for theme, members in groups.items():
        members_sorted = sorted(members, key=lambda i: compounds[i])
        objections.append({
            "theme": theme,
            "count": len(members),
            "share_of_negative_pct": _pct(len(members), neg_total),
            "examples": [originals[i] for i in members_sorted[:2]],
        })
    objections.sort(key=lambda o: -o["count"])
    objections = objections[:6]

    # Priority reply queue: rule-scored; reasons shown so the ranking is
    # auditable. This ranks by intent signals, NOT by whether the business
    # already replied (reply status is not in the comment data).
    dated_sorted = sorted([t for t in timestamps if t is not None])
    recent_cutoff = dated_sorted[int(len(dated_sorted) * 0.75)] if len(dated_sorted) >= 8 else None
    scored = []
    for i in range(total):
        if intents[i] == "spam":
            continue
        score, reasons = 0, []
        if intents[i] == "buying_intent":
            score += 3
            reasons.append("Buying intent — possible customer")
        if intents[i] == "complaint":
            score += 3
            reasons.append("Complaint — needs a response")
        if intents[i] == "question" or (intents[i] == "buying_intent" and "?" in originals[i]):
            if intents[i] != "buying_intent":
                score += 2
                reasons.append("Question")
        if _sentiment_bucket(compounds[i]) == "negative" and intents[i] != "complaint":
            score += 1
            reasons.append("Negative tone")
        if recent_cutoff is not None and timestamps[i] is not None and timestamps[i] >= recent_cutoff:
            score += 1
            reasons.append("Recent")
        if score >= 3 and reasons:
            r = row(i)
            r["score"] = score
            r["reasons"] = reasons
            r["sentiment"] = _sentiment_bucket(compounds[i])
            scored.append(r)
    def _epoch(r):
        ts = _parse_ts(r.get("timestamp"))
        return ts.timestamp() if ts is not None else 0.0
    scored.sort(key=lambda r: (-r["score"], -_epoch(r)))
    priority = scored[:10]

    # Suggested reply drafts for the most-repeated question themes.
    drafts = []
    for group in overlap:
        theme = group["theme"]
        rep = group["representative"].strip()
        is_q = rep.endswith("?") or rep.lower().startswith(QUESTION_STARTERS)
        template = None
        for prefix, tpl in REPLY_DRAFT_TEMPLATES:
            if theme.startswith(prefix):
                template = tpl
                break
        if template is None and is_q:
            template = REPLY_DRAFT_GENERIC
        if template is None:
            continue
        drafts.append({
            "theme": theme,
            "times_asked": group["count"],
            "representative": rep,
            "draft": template,
            "note": REPLY_DRAFT_NOTE,
        })
        if len(drafts) >= 4:
            break

    return {
        "intents": intent_summary,
        "leads": leads,
        "objections": objections,
        "priority": priority,
        "reply_drafts": drafts,
        "priority_basis": ("Ranked by intent signals in the comment text "
                           "(buying intent, complaint, question, negative tone, "
                           "recency). Reply status is not tracked."),
    }
