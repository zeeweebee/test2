import os
import time
import psycopg2
import numpy as np
import threading
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import HTMLResponse
from typing import Optional, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.isotonic import IsotonicRegression

app = FastAPI(title="Epidermix Phase 0 Live API")

# Fallback DSN for local development, overwritten by cloud providers via env strings
DATABASE_URL = os.getenv("DATABASE_URL", "host=localhost dbname=epidermix_demo user=epidermix password=epidermix")

# Global state items populated on app startup
dense_model: Optional['DenseModel'] = None
calibration_model: Optional[IsotonicRegression] = None
all_document_ids: List[int] = []

# --- CORE LOGIC INHERITED FROM MODULE 4 ---
class DenseModel:
    def __init__(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(12, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        self.doc_vectors = self.svd.fit_transform(tfidf)

    def embed_query(self, query: str):
        return self.svd.transform(self.vectorizer.transform([query]))[0]

    def similarities(self, query: str):
        qv = self.embed_query(query).reshape(1, -1)
        return cosine_similarity(qv, self.doc_vectors)[0]

def fetch_corpus_internal():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, source_name, license, body FROM corpus ORDER BY id;")
    rows = cur.fetchall()
    conn.close()
    return rows

def bm25_scores_internal(query: str, doc_ids: List[int]) -> Dict[int, float]:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score
        FROM corpus WHERE id = ANY(%s)
    """, (query, doc_ids))
    scores = dict(cur.fetchall())
    conn.close()
    return {i: float(scores.get(i, 0.0)) for i in doc_ids}

def rrf_fuse(rank_lists: List[Dict[int, int]], k=60) -> Dict[int, float]:
    scores = {}
    for ranks in rank_lists:
        for doc_id, rank in ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores

def ranks_from_scores(score_dict: Dict[int, float]) -> Dict[int, int]:
    ordered = sorted(score_dict.items(), key=lambda x: -x[1])
    return {doc_id: i + 1 for i, (doc_id, _) in enumerate(ordered)}

@app.on_event("startup")
def startup_event():
    """Seeds tables if required, loads datasets, runs calibration routine."""
    global dense_model, calibration_model, all_document_ids
    
    # Simple table initialization verification pattern
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS corpus (
            id SERIAL PRIMARY KEY,
            source_name TEXT NOT NULL,
            license TEXT NOT NULL CHECK (license IN ('cc_by','cleared','restricted')),
            body TEXT NOT NULL,
            tsv TSVECTOR
        );
        CREATE INDEX IF NOT EXISTS corpus_tsv_idx ON corpus USING GIN(tsv);
        
        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            status TEXT NOT NULL,
            claimed_by TEXT,
            version INT NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS consent_events (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            event TEXT NOT NULL
        );
    """)
    conn.commit()
    
    # Check if empty - seed sample set if so
    cur.execute("SELECT COUNT(*) FROM corpus;")
    if cur.fetchone()[0] == 0:
        from seed_data import DOCS  # Separated out to keep app file sleek
        for source, lic, body in DOCS:
            cur.execute(
                "INSERT INTO corpus (source_name, license, body, tsv) VALUES (%s, %s, %s, to_tsvector('english', %s))",
                (source, lic, body, body)
            )
        conn.commit()
    
    # Hydrate models
    corpus = fetch_corpus_internal()
    all_document_ids = [row[0] for row in corpus]
    texts = [row[3] for row in corpus]
    dense_model = DenseModel(texts)
    
    # Fit Isotonic Regression curve using dev sets
    dev_pairs = [
        ("red itchy patches on both elbows for three weeks with scaling", 1, 1),
        ("red itchy patches on both elbows for three weeks with scaling", 3, 1),
        ("red itchy patches on both elbows for three weeks with scaling", 4, 1),
        ("red itchy patches on both elbows for three weeks with scaling", 9, 0),
        ("redness after using a new soap that comes and goes", 6, 1),
        ("redness after using a new soap that comes and goes", 7, 1),
        ("redness after using a new soap that comes and goes", 8, 0),
        ("sudden round patches of hair loss with joint pain", 10, 1),
        ("sudden round patches of hair loss with joint pain", 11, 1),
        ("sudden round patches of hair loss with joint pain", 12, 1)
    ]
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(all_document_ids)}
    xs, ys = [], []
    for query, d_id, label in dev_pairs:
        if d_id in id_to_idx:
            bm25 = bm25_scores_internal(query, all_document_ids)
            sims = dense_model.similarities(query)
            dense_scores = {uid: float(sims[id_to_idx[uid]]) for uid in all_document_ids}
            fused = rrf_fuse([ranks_from_scores(bm25), ranks_from_scores(dense_scores)])
            xs.append(fused.get(d_id, 0.0))
            ys.append(label)
            
    calibration_model = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calibration_model.fit(xs, ys)
    conn.close()

# --- HOMEPAGE ---
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Epidermix Phase 0 - Search</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            padding: 40px 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            font-size: 24px;
            margin-bottom: 8px;
            color: #333;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .search-box {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
        }
        input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        input:focus {
            outline: none;
            border-color: #0066cc;
            box-shadow: 0 0 0 3px rgba(0,102,204,0.1);
        }
        button {
            padding: 12px 24px;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }
        button:hover {
            background: #0052a3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .results {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .result-card {
            background: #f9f9f9;
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 12px;
        }
        .verdict {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 15px;
        }
        .verdict.grounded { color: #059669; }
        .verdict.moderate { color: #d97706; }
        .verdict.no-grounding { color: #dc2626; }
        .confidence {
            font-size: 13px;
            color: #666;
            margin-bottom: 12px;
        }
        .document {
            font-size: 13px;
            line-height: 1.5;
            color: #555;
        }
        .document-label {
            font-weight: 600;
            color: #333;
            margin-bottom: 4px;
        }
        .error {
            color: #dc2626;
            padding: 12px;
            background: #fee2e2;
            border-radius: 4px;
            margin-top: 12px;
        }
        .loading {
            color: #0066cc;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Epidermix Phase 0 Live API</h1>
        <p class="subtitle">Hybrid search with clinical grounding</p>
        
        <div class="search-box">
            <input 
                type="text" 
                id="queryInput" 
                placeholder="Enter a clinical query..."
                value="red itchy patches on both elbows for three weeks with scaling"
            >
            <button id="searchBtn" onclick="performSearch()">Search</button>
        </div>
        
        <div class="results" id="results" style="display: none;">
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        const searchBtn = document.getElementById('searchBtn');
        const queryInput = document.getElementById('queryInput');
        const resultsDiv = document.getElementById('results');
        const resultContent = document.getElementById('resultContent');

        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });

        async function performSearch() {
            const query = queryInput.value.trim();
            if (!query) return;

            searchBtn.disabled = true;
            resultContent.innerHTML = '<p class="loading">Searching...</p>';
            resultsDiv.style.display = 'block';

            try {
                const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
                const data = await response.json();

                if (!response.ok) {
                    resultContent.innerHTML = `<div class="error">${data.detail || 'Search failed'}</div>`;
                    return;
                }

                let verdictClass = 'no-grounding';
                if (data.verdict.includes('GROUNDED')) verdictClass = 'grounded';
                else if (data.verdict.includes('MODERATE')) verdictClass = 'moderate';

                resultContent.innerHTML = `
                    <div class="result-card">
                        <div class="verdict ${verdictClass}">${data.verdict}</div>
                        <div class="confidence">Calibrated Confidence: ${data.calibrated_confidence}</div>
                        ${data.matched_document ? `
                            <div class="document">
                                <div class="document-label">Matched Document (ID: ${data.matched_document.id})</div>
                                <strong>${data.matched_document.source}</strong> [${data.matched_document.license}]<br>
                                ${data.matched_document.preview}
                            </div>
                        ` : ''}
                        ${data.reason ? `<div style="margin-top: 12px; font-size: 13px; color: #666;">${data.reason}</div>` : ''}
                    </div>
                `;
            } catch (error) {
                resultContent.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                searchBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_CONTENT

# --- MODULE 4 API ENDPOINT ---
@app.get("/search")
def hybrid_search(q: str = Query(..., description="The query string to evaluate")):
    corpus_rows = fetch_corpus_internal()
    corpus_dict = {row[0]: row for row in corpus_rows}
    
    bm25 = bm25_scores_internal(q, all_document_ids)
    sims = dense_model.similarities(q)
    id_to_idx = {d_id: idx for idx, d_id in enumerate(all_document_ids)}
    dense_scores = {d_id: float(sims[id_to_idx[d_id]]) for d_id in all_document_ids}
    
    fused = rrf_fuse([ranks_from_scores(bm25), ranks_from_scores(dense_scores)])
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    
    # Dynamic licensing screening criteria
    gated = [(d_id, score) for d_id, score in ranked if corpus_dict[d_id][2] != "restricted"]
    if not gated:
        return {"verdict": "NO GROUNDING", "reason": "No cleared resources passed the dynamic licensing evaluation window.", "action": "escalate to human"}
    
    top_doc_id, top_score = gated[0]
    raw_bm25_top = bm25.get(top_doc_id, 0.0)
    raw_dense_top = dense_scores.get(top_doc_id, 0.0)
    
    # Grounding fallback configuration check
    if raw_bm25_top < 1e-6 and raw_dense_top < 0.05:
        return {
            "verdict": "NO GROUNDING",
            "reason": f"Signal baseline floor failed. Raw BM25 ({raw_bm25_top:.4f}) and Dense ({raw_dense_top:.4f}) metrics too distant.",
            "action": "escalate to human"
        }
        
    confidence = float(calibration_model.predict([top_score])[0])
    _, source, lic, body = corpus_dict[top_doc_id]
    
    if confidence >= 0.75:
        verdict = "GROUNDED -- ready to route"
    elif confidence >= 0.40:
        verdict = "MODERATE -- flag for clinician review"
    else:
        verdict = "NO GROUNDING -- escalate to human"
        
    return {
        "verdict": verdict,
        "calibrated_confidence": round(confidence, 2),
        "matched_document": {
            "id": top_doc_id,
            "source": source,
            "license": lic,
            "preview": body[:120] + "..."
        }
    }

# --- MODULE 3 API ENDPOINT ---
@app.post("/queue/claim")
def claim_job(worker_name: str = Body(..., embed=True), method: str = Body("skip_locked", embed=True)):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    if method == "naive":
        cur.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="No queued jobs available")
        job_id = row[0]
        time.sleep(0.02) # Simulates race condition exposure window
        cur.execute("UPDATE jobs SET status='claimed', claimed_by=%s WHERE id=%s", (worker_name, job_id))
        conn.commit()
    else:
        # Atomic SKIP LOCKED pattern execution
        cur.execute("""
            UPDATE jobs SET status='claimed', claimed_by=%s, version = version + 1
            WHERE id = (
                SELECT id FROM jobs WHERE status = 'queued' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
            ) RETURNING id, session_id
        """, (worker_name,))
        row = cur.fetchone()
        conn.commit()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="No thread-safe jobs free")
        job_id, session_id = row[0], row[1]
        
    conn.close()
    return {"status": "claimed", "job_id": job_id, "method_used": method}
