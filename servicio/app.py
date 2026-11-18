"""Servicio web de recuperación de información sobre SciFact (TF-IDF y BM25)."""
import json
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from modelos import MODELOS, Buscador, ConsultaInvalida, fragmento
from preprocesamiento import cargar_stopwords

AQUI = Path(__file__).resolve().parent
CONFIG = json.loads((AQUI / "config.json").read_text(encoding="utf-8"))
BUSCADOR = Buscador.desde_jsonl(AQUI / "corpus.jsonl", cargar_stopwords(AQUI / "stopwords_en.txt"),
                                stem=CONFIG["stem"], k1=CONFIG["k1"], b=CONFIG["b"])

app = FastAPI(title="RI SciFact — TF-IDF y BM25", version="1.0.0",
              description="Búsqueda sobre el corpus SciFact (BeIR) con TF-IDF (coseno) y BM25.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


class Consulta(BaseModel):
    query: str = Field(..., max_length=5000, examples=["0-dimensional biomaterials show inductive properties"])
    model: Literal["bm25", "tfidf"] = Field("bm25", examples=["bm25"])
    top_k: int = Field(10, ge=1, le=1000, examples=[10])

    @field_validator("model", mode="before")
    @classmethod
    def _normalizar_modelo(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class Resultado(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class Respuesta(BaseModel):
    model: str
    results: List[Resultado]


@app.get("/")
def raiz():
    return {"servicio": "RI SciFact — TF-IDF y BM25", "corpus": "BeIR/scifact", "documentos": BUSCADOR.indice.n_docs,
            "endpoints": {"GET /health": "disponibilidad", "POST /search": "búsqueda", "GET /docs": "documentación interactiva"}}


@app.get("/health")
def health():
    return {"status": "ok", "documents": BUSCADOR.indice.n_docs, "models": list(MODELOS)}


@app.post("/search", response_model=Respuesta)
def buscar(consulta: Consulta):
    texto = consulta.query.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="La consulta está vacía.")
    try:
        pares = BUSCADOR.buscar(texto, consulta.model, consulta.top_k)
    except ConsultaInvalida as error:
        raise HTTPException(status_code=422, detail=str(error))
    resultados = []
    for rank, (doc_id, score) in enumerate(pares, start=1):
        titulo, cuerpo = BUSCADOR.documentos[doc_id]
        resultados.append(Resultado(rank=rank, document_id=doc_id, score=score, title=titulo, snippet=fragmento(cuerpo)))
    return Respuesta(model=consulta.model, results=resultados)
