import os
import time
import psycopg2
import numpy as np
import threading
from fastapi import FastAPI, HTTPException, Query, Body
from typing import Optional, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.isotonic import IsotonicRegression
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Epidermix Phase 0 Live API")
app.mount("/", StaticFiles(directory="."), name="static")

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

# --- HEALTH CHECK ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Application is running"}

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
