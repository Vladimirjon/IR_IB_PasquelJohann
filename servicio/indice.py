"""Índice invertido propio (formato CSR de NumPy)."""
import json
from collections import Counter

import numpy as np

FORMATO = 1


class IndiceInvertido:
    def __init__(self, doc_ids, terminos, offsets, post_docs, post_tf, doc_len, meta=None):
        self.doc_ids = list(doc_ids)              # posición interna -> id original
        self.terminos = list(terminos)            # id de término -> texto
        self.vocab = {t: i for i, t in enumerate(self.terminos)}
        self.offsets = offsets                    # postings del término i: [offsets[i], offsets[i+1])
        self.post_docs = post_docs                # posiciones internas de documento
        self.post_tf = post_tf                    # frecuencia del término en ese documento
        self.doc_len = doc_len
        self.meta = dict(meta or {})
        self.df = np.diff(offsets).astype(np.int64)
        self.n_docs = len(self.doc_ids)
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0

    @classmethod
    def construir(cls, doc_ids, docs_tokens, meta=None):
        vocab, tids, dids, tfs = {}, [], [], []
        doc_len = np.zeros(len(doc_ids), dtype=np.int32)
        for d, tokens in enumerate(docs_tokens):
            doc_len[d] = len(tokens)
            for termino, f in Counter(tokens).items():
                tids.append(vocab.setdefault(termino, len(vocab)))
                dids.append(d)
                tfs.append(f)
        tids = np.asarray(tids, dtype=np.int64)
        orden = np.argsort(tids, kind="stable")      # agrupa por término; dentro, documentos ascendentes
        df = np.bincount(tids, minlength=len(vocab))
        offsets = np.concatenate(([0], np.cumsum(df))).astype(np.int64)
        terminos = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        return cls(doc_ids, terminos, offsets,
                   np.asarray(dids, dtype=np.int32)[orden], np.asarray(tfs, dtype=np.int32)[orden],
                   doc_len, meta)

    def postings(self, termino):
        tid = self.vocab.get(termino)
        if tid is None:
            return np.empty(0, np.int32), np.empty(0, np.int32)
        s, e = self.offsets[tid], self.offsets[tid + 1]
        return self.post_docs[s:e], self.post_tf[s:e]

    def memoria_bytes(self):
        return sum(a.nbytes for a in (self.offsets, self.post_docs, self.post_tf, self.doc_len))

    def guardar(self, ruta):
        meta = {**self.meta, "formato": FORMATO, "n_docs": self.n_docs}
        np.savez_compressed(ruta, doc_ids=np.array(self.doc_ids, dtype=str),
                            terminos=np.array(self.terminos, dtype=str), offsets=self.offsets,
                            post_docs=self.post_docs, post_tf=self.post_tf, doc_len=self.doc_len,
                            meta=np.array(json.dumps(meta)))

    @classmethod
    def cargar(cls, ruta):
        z = np.load(ruta, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("formato") != FORMATO:
            raise ValueError("Formato de índice no soportado")
        return cls(z["doc_ids"].tolist(), z["terminos"].tolist(), z["offsets"], z["post_docs"],
                   z["post_tf"], z["doc_len"], meta)
