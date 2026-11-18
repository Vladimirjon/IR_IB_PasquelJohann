# Examen ICCD753 — Recuperación de Información

**Estudiante:** Pasquel Johann · **Corpus asignado:** SciFact (BeIR)

Sistema de recuperación de información con **TF-IDF (similitud coseno)** y **BM25**, implementados a mano sobre un índice invertido, y publicado como servicio web.

## Servicio publicado (Vercel)

El servicio está desplegado en **Vercel** (plan Hobby, gratuito) y es público, sin login:

**https://ri-scifact-pasquel.vercel.app** · documentación interactiva: [`/docs`](https://ri-scifact-pasquel.vercel.app/docs)

| Endpoint | Descripción |
| --- | --- |
| `GET /health` | Disponibilidad |
| `POST /search` | Búsqueda. Cuerpo: `{"query": "...", "model": "bm25" \| "tfidf", "top_k": 10}` |

```bash
curl https://ri-scifact-pasquel.vercel.app/health

curl -X POST https://ri-scifact-pasquel.vercel.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "0-dimensional biomaterials show inductive properties", "model": "bm25", "top_k": 5}'
```

Respuesta: `{"model": "bm25", "results": [{"rank", "document_id", "score", "title", "snippet"}]}`, ordenada por score descendente y sin duplicados. `model` por defecto `bm25`; `top_k` de 1 a 1000 (por defecto 10). Errores `422` si la consulta está vacía o solo tiene stopwords, el modelo no es válido o `top_k` está fuera de rango. Si hay menos documentos con coincidencia que `top_k`, se completa con documentos de score 0.

> Tras un rato sin visitas, la primera petición puede tardar unos segundos más (arranque en frío). Vercel Hobby es de uso no comercial y sin garantía de disponibilidad.

## Qué revisar

1. **Probar el servicio** con Postman, `curl` o `wget` en la URL de arriba.
2. **Leer el cuaderno [`examen1bim.ipynb`](examen1bim.ipynb)**: ya está ejecutado y contiene, paso a paso, la carga del corpus, el preprocesamiento, el índice, los modelos, la evaluación en `test`, el análisis de resultados, el servicio y sus pruebas (locales y contra la URL pública).

## Resultados en `test` (300 consultas)

| Modelo | P@10 | R@10 | NDCG@10 |
| --- | --- | --- | --- |
| BM25 | 0.0910 | 0.8245 | **0.6816** |
| TF-IDF (coseno) | 0.0903 | 0.8183 | 0.6450 |

BM25 supera a TF-IDF en NDCG@10 (+0.0366; prueba de signos p = 0.0012); P@10 y R@10 no difieren de forma significativa. Preprocesamiento y parámetros (Porter, `k1 = 1.2`, `b = 0.75`) se eligieron solo con `train`. Detalle en `resultados/`.

## Estructura

```
examen1bim.ipynb   Cuaderno del examen (ejecutado)
servicio/          Servicio FastAPI desplegado en Vercel (código, corpus y configuración)
resultados/        Métricas de test (JSON y CSV por consulta)
```

## Reproducir

Requiere Python 3.12 (PowerShell, desde la raíz del repositorio):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 nltk==3.10.4 requests==2.34.2 huggingface_hub==2.4.0 fastapi==0.121.2 uvicorn==0.54.0 ipykernel==7.1.0 nbconvert==7.16.6
$env:RI_SKIP_DEPLOY = "1"   # omite las pruebas contra la URL pública
python -m jupyter nbconvert --to notebook --execute --inplace examen1bim.ipynb
```

La primera ejecución descarga SciFact y las stopwords de NLTK (necesita conexión). Para levantar solo el servicio en local: `cd servicio` y `python -m uvicorn app:app --port 8000`.
