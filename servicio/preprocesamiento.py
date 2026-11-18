"""Preprocesamiento de texto compartido por la evaluación y el servicio web."""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()


@lru_cache(maxsize=None)
def _raiz(token: str) -> str:
    return _STEMMER.stem(token)


def cargar_stopwords(ruta) -> frozenset:
    lineas = Path(ruta).read_text(encoding="utf-8").splitlines()
    return frozenset(linea.strip() for linea in lineas if linea.strip())


def tokenizar(texto: str) -> list:
    """Minúsculas, sin acentos, tokens alfanuméricos de al menos 2 caracteres."""
    texto = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode("ascii")
    return [t for t in TOKEN_RE.findall(texto) if len(t) > 1]


def preprocesar(texto: str, stopwords=frozenset(), stem: bool = True) -> list:
    """Tokeniza, quita stopwords y (opcionalmente) aplica el stemmer de Porter."""
    tokens = [t for t in tokenizar(texto) if t not in stopwords]
    return [_raiz(t) for t in tokens] if stem else tokens
