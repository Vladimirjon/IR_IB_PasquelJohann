# Recuperación de información sobre SciFact

Servicio web (FastAPI) que indexa el corpus SciFact (BeIR) y ofrece dos rankings independientes:
TF-IDF con similitud coseno y BM25, ambos implementados a mano sobre un índice invertido.

* `GET /health` — comprobación de disponibilidad.
* `POST /search` — cuerpo `{"query": "texto", "model": "bm25" | "tfidf", "top_k": 10}`.
* `GET /docs` — documentación interactiva.

Examen de Recuperación de Información (ICCD753). El código y la evaluación están en el cuaderno del examen.
Se despliega en Vercel (carpeta raíz `servicio/`, framework FastAPI, Python 3.12).
