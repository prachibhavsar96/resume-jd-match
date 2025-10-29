# Resume ↔ JD Match Scorer

Upload a resume (PDF) + paste a job description. Get a **match score (0–100)**, **top missing skills**, and **suggested resume bullets**.

## Tech
- **Backend:** FastAPI, scikit-learn (TF‑IDF similarity), pdfminer.six
- **Frontend:** React (Vite)
- **DevOps:** Docker, GitHub Actions CI

## Quickstart (no Docker)
### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open docs at: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open: http://localhost:5173 (uses `VITE_API_URL` env, default `http://localhost:8000`).

## Quickstart (Docker)
```bash
docker compose up --build
```
- Frontend: http://localhost:5173
- Backend: http://localhost:8000/docs

## API
`POST /analyze` form-data fields:
- `jd_text`: string (job description)
- `resume`: file (PDF or .txt)

Returns JSON:
```json
{
  "score": 78.4,
  "missing_skills": ["kubernetes", "pytorch", "redis"],
  "suggested_bullets": ["...", "..."]
}
```

## Roadmap
- Embedding-based similarity (MiniLM)
- Better skill extraction (spaCy patterns, ONET/ESCO lists)
- PDF export of the analysis
- Auth + saved analyses
```
