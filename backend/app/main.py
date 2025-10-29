from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from .pdf_utils import extract_text_from_pdf_bytes
from .nlp import compute_match_score, extract_missing_skills, suggest_bullets

app = FastAPI(title="Resume ↔ JD Match Scorer", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for local dev / frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, Any]:
    return {"message": "Resume–JD Match API is running. See /docs for Swagger UI."}

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(jd_text: str = Form(...), resume: UploadFile = File(...)) -> Dict[str, Any]:
    # read file
    raw = await resume.read()
    # choose extractor based on extension; allow .txt as fast path
    if resume.filename and resume.filename.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf_bytes(raw)
    else:
        resume_text = raw.decode("utf-8", errors="ignore")

    # guard: if PDF had little/no extractable text, return a clear warning
    if len(resume_text.strip()) < 300:
        return {
            "score": 0.0,
            "missing_skills": [],
            "suggested_bullets": [],
            "warning": (
                "Resume PDF has little extractable text. "
                "Try exporting as a text-based PDF or upload a .txt/.docx version."
            ),
        }

    # compute results
    score = compute_match_score(jd_text, resume_text)
    gaps = extract_missing_skills(jd_text, resume_text, top_k=8)
    bullets = suggest_bullets(jd_text, gaps)

    return {"score": score, "missing_skills": gaps, "suggested_bullets": bullets}
