from __future__ import annotations
from typing import List, Tuple, Dict
import re

# --- Optional fast fallback (kept for safety) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Canonical skills & aliases (regex, case-insensitive) ----------
SKILL_PATTERNS = {
    # Backend / data
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "c++": r"\bc\+\+\b",
    "sql": r"\bsql\b",
    "postgresql": r"\b(postgres|postgresql)\b",
    "mysql": r"\bmy\s*sql|mysql\b",
    "mongodb": r"\bmongo(db)?\b",

    # Web / frontend
    "javascript": r"\b(java\s*script|javascript|js)\b",
    "typescript": r"\b(type\s*script|typescript|ts)\b",
    "react": r"\breact(\.js)?\b",
    "node": r"\b(node(\.js)?|express)\b",
    "jquery": r"\bjquery\b",
    "sass/scss": r"\b(sass|scss)\b",
    "backbone.js": r"\bbackbone(\.js)?\b",

    # APIs / infra
    "rest": r"\brest(ful)?\b",
    "grpc": r"\bgrpc\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bk8s|kubernetes\b",
    "terraform": r"\bterraform\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "gcp": r"\bgoogle cloud|gcp\b",

    # CI/CD & testing
    "ci/cd": r"\b(ci\s*/?\s*cd|cicd|continuous\s+integration|continuous\s+delivery|continuous\s+deployment)\b",
    "github actions": r"\bgithub\s+actions\b",
    "jenkins": r"\bjenkins\b",
    "testing": r"\btesting|unit\s*tests?\b",
    "pytest": r"\bpytest\b",
    "unit testing": r"\bunit\s*test(s|ing)\b",
    "tdd": r"\btdd\b|\btest[-\s]?driven\b",

    # Domain specifics seen in your JD
    "salesforce": r"\bsalesforce(\.com)?\b",
    "saas": r"\bsaas\b|\bsoftware as a service\b",
    "accessibility/wcag": r"\b(wcag\s*2(\.0|\.\d+)?\+?|accessibility|a11y)\b",
    "security best practices": r"\bsecurity\b|\bowasp\b|\bsecure\s+coding\b",

    # ML/NLP (keep for other JDs)
    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "scikit-learn": r"\bscikit-?learn|sklearn\b",
    "pytorch": r"\bpytorch\b",
    "tensorflow": r"\btensorflow|tf\b",
    "nlp": r"\bnlp\b|\bnatural language processing\b",
    "rag": r"\brag\b|\bretrieval-?augmented\b",
    "transformers": r"\btransformers?\b",
    "faiss": r"\bfaiss\b|\bvector\s*db|\bvector\s*database\b",
    "mlflow": r"\bmlflow\b",
    "sagemaker": r"\bsagemaker\b",
    "jwt": r"\bjwt\b|\bjson web token(s)?\b",
    "oauth": r"\boauth(2)?|oidc\b",
    "microservices": r"\bmicro-?services?\b",
}
_CANON = list(SKILL_PATTERNS.keys())
_RX = {k: re.compile(v, re.IGNORECASE) for k, v in SKILL_PATTERNS.items()}

# ---------- Helpers ----------
def _norm(s: str) -> str:
    return (s or "").strip()

def _skills_in(text: str) -> List[str]:
    t = _norm(text)
    return [k for k, rx in _RX.items() if rx.search(t)]

def _split_sections(jd: str) -> Dict[str, str]:
    """
    Naive section splitter to weight 'Required', 'Preferred/Nice', and the rest.
    Works well enough for most JDs without dependencies.
    """
    jd = jd.replace("\r", "")
    sections = {
        "required": "",
        "preferred": "",
        "other": jd
    }
    # Try to cut out sections by headings
    patterns = [
        (re.compile(r"(required skills?|requirements?)\s*[:\n]", re.I), "required"),
        (re.compile(r"(preferred|nice to have|nice-to-have)\s*[:\n]", re.I), "preferred"),
    ]
    # Find spans
    spans = []
    for rx, name in patterns:
        m = rx.search(jd)
        if m:
            spans.append((m.start(), m.end(), name))
    spans.sort()
    if not spans:
        return sections

    # Build ranges
    pieces = []
    for i, (s, e, name) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(jd)
        pieces.append((name, jd[e:end]))
    # Assign
    req_text = "\n".join(p[1] for p in pieces if p[0] == "required")
    pref_text = "\n".join(p[1] for p in pieces if p[0] == "preferred")
    sections["required"] = req_text or ""
    sections["preferred"] = pref_text or ""
    # other = jd minus those pieces
    used = "".join(p[1] for p in pieces)
    sections["other"] = jd.replace(used, "")
    return sections

def _coverage_weighted(jd: str, resume: str) -> float:
    """
    Weighted coverage: skills found in 'Required' count more than 'Preferred'.
    Returns 0..1
    """
    sec = _split_sections(jd)
    weights = {"required": 2.0, "preferred": 1.2, "other": 1.0}
    found = 0.0
    total = 0.0
    resume_hits = set(_skills_in(resume))
    for name, text in sec.items():
        skills = set(_skills_in(text))
        w = weights[name]
        total += w * len(skills)
        found += w * len(skills & resume_hits)
    if total == 0:
        return 0.0
    return found / total

# ---------- Semantic embeddings (lazy) ----------
_MODEL = None  # lazy singletons

def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # import lazily
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def _semantic_sim(jd: str, resume: str) -> float:
    """
    Cosine similarity on MiniLM embeddings, mapped from [-1,1] to [0,1].
    """
    model = _get_model()
    from sentence_transformers import util  # within function to avoid global import when not used
    jd_emb = model.encode(jd, convert_to_tensor=True, normalize_embeddings=True)
    rs_emb = model.encode(resume, convert_to_tensor=True, normalize_embeddings=True)
    cos = float(util.cos_sim(jd_emb, rs_emb).item())
    return (cos + 1.0) / 2.0  # [-1..1] -> [0..1]

# ---------- Fast TF-IDF fallback (if embeddings not available) ----------
def _tfidf_sim(jd: str, resume: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
    X = vec.fit_transform([jd.lower(), resume.lower()])
    sim = float(cosine_similarity(X[0:1], X[1:2])[0][0])
    return max(0.0, min(1.0, sim))

# ---------- Public API ----------
def compute_match_score(jd: str, resume: str) -> float:
    """
    Hybrid score in 0..100:
      70% semantic (MiniLM) + 30% weighted skills coverage (Required>Preferred>Other).
    Falls back to TF-IDF if embeddings packages are not installed.
    """
    jd = _norm(jd)
    resume = _norm(resume)
    if not jd or not resume:
        return 0.0

    try:
        sem = _semantic_sim(jd, resume)     # 0..1
        cov = _coverage_weighted(jd, resume)  # 0..1
        final = 0.7 * sem + 0.3 * cov
    except Exception:
        # If sentence-transformers/torch isn’t available, use TF-IDF
        sim = _tfidf_sim(jd, resume)
        cov = _coverage_weighted(jd, resume)
        # stretch TF-IDF to feel reasonable
        lo, hi = 0.02, 0.35
        if sim <= lo:
            sim_s = 0.0
        elif sim >= hi:
            sim_s = 1.0
        else:
            sim_s = (sim - lo) / (hi - lo)
        final = 0.6 * sim_s + 0.4 * cov

    return round(final * 100.0, 2)

def extract_missing_skills(jd: str, resume: str, top_k: int = 8) -> List[str]:
    want = set(_skills_in(jd))
    have = set(_skills_in(resume))
    if not want:
        return []
    gaps = [s for s in _CANON if s in want and s not in have]
    return gaps[:top_k]

def suggest_bullets(jd: str, gaps: List[str]) -> List[str]:
    bullets: List[str] = []
    for g in gaps[:5]:
        bullets.append(f"Implemented {g} features/APIs and wrote unit tests to improve reliability and performance.")
    bullets.append("Optimized backend endpoints, reducing latency by 20%.")
    bullets.append("Set up CI on GitHub Actions and Dockerized services for reproducible deployments.")
    return bullets