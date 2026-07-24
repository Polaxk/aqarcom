from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Imports mapped to your exact project structure and function names
from src.ingestion import ingest_pdf
from src.extractor import ExtractedFeatures, extract_features
from src.rules_engine import evaluate_risk
from src.narration import generate_report

app = FastAPI(
    title="Aqarcom API",
    description="Evidence-based property-financing risk intelligence API for Jordan.",
    version="0.1.0",
)

# Enables CORS so your React/TypeScript frontend can call this backend during the hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/cases/scan")
async def scan_case(
    file: UploadFile = File(..., description="Purchase agreement / title extract / case PDF"),
    profile: str = Query("lender", enum=["lender", "legal"], description="Target audience view for report formatting"),
) -> dict:
    """Full pipeline execution: PDF Upload -> Text Ingestion -> AI Extraction -> Deterministic Rules Engine -> Final Report."""

    # 1. Validate input file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported.")

    # 2. Persist temporary file to disk (Docling requires a physical file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Step A: Ingest PDF text using Docling
        try:
            raw_text = ingest_pdf(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF Ingestion failed: {exc}") from exc

        # Step B: Extract structured Pydantic features via NVIDIA Gemma NIM
        try:
            features: ExtractedFeatures = extract_features(raw_text)
            feature_data = features.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Feature extraction failed: {exc}") from exc

        # Step C: Deterministic policy evaluation
        try:
            evaluation = evaluate_risk(feature_data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Rules engine evaluation failed: {exc}") from exc

        # Step D: Generate presentation-ready report
        try:
            final_report = generate_report(evaluation)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

        # Return structured JSON payload containing features, score, evidence, and report text
        return {
            "file_name": file.filename,
            "profile": profile,
            "extracted_features": feature_data,
            "evaluation": evaluation,
            "final_report": final_report,
        }

    finally:
        # Clean up temporary disk file
        if tmp_path.exists():
            os.remove(tmp_path)