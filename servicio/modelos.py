"""TF-IDF (coseno) y BM25 sobre el índice invertido."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from indice import IndiceInvertido
from preprocesamiento import preprocesar

MODELOS = ("bm25", "tfidf")


class ConsultaInvalida(ValueError):
    """La consulta no contiene ningún término indexable."""


def fragmento(texto: str, n: int = 300) -> str:
    texto = " ".join((texto or "").split())
    if len(texto) <= n:
        return texto
    corte = texto[:n].rsplit(" ", 1)[0] or texto[:n]
    return corte.rstrip(" ,;:.") + "…"


class Buscador:
    def __init__(self, indice, stopwords, stem=True, k1=1.5, b=0.75, documentos=None):
        self.indice, self.stopwords, self.stem = indice, stopwords, stem
        self.documentos = documentos or {}            # id -> (título, texto)
        n, v = indice.n_docs, len(indice.terminos)
        self._termino_de_posting = np.repeat(np.arange(v), indice.df)
        # TF-IDF: peso (1+ln tf)·idf normalizado por la norma L2 del documento
        self.idf_tfidf = np.log(n / indice.df)
        w = (1.0 + np.log(indice.post_tf)) * self.idf_tfidf[self._termino_de_posting]
        norma = np.sqrt(np.bincount(indice.post_docs, weights=w ** 2, minlength=n))
        norma[norma == 0] = 1.0
        self._w_tfidf = w / norma[indice.post_docs]
        # BM25
        self.idf_bm25 = np.log(1.0 + (n - indice.df + 0.5) / (indice.df + 0.5))
        self.fijar_bm25(k1, b)
        # rango de cada documento por id (desempates deterministas)
        orden = np.argsort(np.array(indice.doc_ids, dtype=str), kind="stable")
        self._rango_id = np.empty(n, dtype=np.int64)
        self._rango_id[orden] = np.arange(n)

    def fijar_bm25(self, k1, b):
        ind = self.indice
        self.k1, self.b = k1, b
        dl = ind.doc_len[ind.post_docs]
        tf = ind.post_tf
        sat = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / ind.avgdl))
        self._w_bm25 = self.idf_bm25[self._termino_de_posting] * sat

    @classmethod
    def desde_documentos(cls, documentos, stopwords, stem=True, k1=1.5, b=0.75):
        """documentos: lista de dicts con _id, title y text (documento indexado = título + texto)."""
        ids = [str(d["_id"]) for d in documentos]
        tokens = [preprocesar(f"{d.get('title') or ''} {d.get('text') or ''}", stopwords, stem) for d in documentos]
        indice = IndiceInvertido.construir(ids, tokens, {"stem": stem})
        textos = {str(d["_id"]): (d.get("title") or "", d.get("text") or "") for d in documentos}
        return cls(indice, stopwords, stem, k1, b, textos)

    @classmethod
    def desde_jsonl(cls, ruta, stopwords, stem=True, k1=1.5, b=0.75):
        with open(ruta, encoding="utf-8") as f:
            docs = [json.loads(linea) for linea in f if linea.strip()]
        return cls.desde_documentos(docs, stopwords, stem, k1, b)

    def _puntuar(self, tokens, modelo):
        ind = self.indice
        cuenta = Counter(t for t in tokens if t in ind.vocab)
        acc = np.zeros(ind.n_docs)
        if modelo == "bm25":
            for t, c in cuenta.items():
                tid = ind.vocab[t]
                s, e = ind.offsets[tid], ind.offsets[tid + 1]
                acc[ind.post_docs[s:e]] += c * self._w_bm25[s:e]
        else:
            pesos = {ind.vocab[t]: (1.0 + math.log(c)) * self.idf_tfidf[ind.vocab[t]] for t, c in cuenta.items()}
            norma_q = math.sqrt(sum(p * p for p in pesos.values()))
            if norma_q > 0:
                for tid, wq in pesos.items():
                    s, e = ind.offsets[tid], ind.offsets[tid + 1]
                    acc[ind.post_docs[s:e]] += wq * self._w_tfidf[s:e]
                acc = np.minimum(acc / norma_q, 1.0)
        return acc

    def buscar(self, consulta, modelo="bm25", top_k=10):
        """Devuelve [(document_id, score)] ordenado por score desc. y document_id asc."""
        if modelo not in MODELOS:
            raise ValueError(f"Modelo desconocido: {modelo!r}")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k debe ser un entero positivo")
        tokens = preprocesar(consulta, self.stopwords, self.stem)
        if not tokens:
            raise ConsultaInvalida("La consulta no contiene términos indexables (vacía o solo stopwords).")
        acc = self._puntuar(tokens, modelo)
        orden = np.lexsort((self._rango_id, -acc))[: min(top_k, self.indice.n_docs)]
        return [(self.indice.doc_ids[i], float(acc[i])) for i in orden]
