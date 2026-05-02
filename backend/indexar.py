"""
indexar.py — Ejecutar UNA SOLA VEZ localmente para generar chroma_db/

Pasos:
  1. Pon todos tus PDFs dentro de la carpeta  backend/documentos/
  2. Desde la carpeta backend/ corre:
       set GOOGLE_API_KEY=tu_clave        (Windows CMD)
       $env:GOOGLE_API_KEY="tu_clave"     (Windows PowerShell)
  3. python indexar.py
  4. Sube chroma_db/ al repositorio y haz deploy en Render
"""

import os
import sys
import time
import requests

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"ERROR de importacion: {e}")
    print("Instala con:")
    print("  python -m pip install langchain-community langchain-text-splitters langchain-chroma chromadb pypdf")
    sys.exit(1)

DOCS_DIR   = "documentos"
CHROMA_DIR = "chroma_db"

# ── Leer clave de Google ─────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("ERROR: Define la variable de entorno GOOGLE_API_KEY antes de correr este script.")
    print("  PowerShell:  $env:GOOGLE_API_KEY=\"tu_clave_aqui\"")
    print("  CMD:         set GOOGLE_API_KEY=tu_clave_aqui")
    sys.exit(1)

# ── Clase de embeddings via REST (compatible con Python 3.14) ────────────────
class GoogleEmbeddingsREST:
    """Llama la API de Google directamente con requests, sin SDK."""

    def __init__(self, api_key):
        self.api_key  = api_key
        self.model    = "gemini-embedding-001"
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )

    def _embed_one(self, text):
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]}
        }
        for intento in range(3):
            try:
                resp = requests.post(self.base_url, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()["embedding"]["values"]
            except Exception as e:
                if intento < 2:
                    time.sleep(2)
                else:
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"ERROR detalle: {e.response.text}")
                    raise e

    def embed_documents(self, texts):
        embeddings = []
        for i, text in enumerate(texts):
            if i % 20 == 0:
                print(f"  Procesando fragmento {i+1}/{len(texts)}...")
            embeddings.append(self._embed_one(text))
            time.sleep(0.1)  # respetar límite de la API
        return embeddings

    def embed_query(self, text):
        return self._embed_one(text)


# ── 1. Cargar PDFs ───────────────────────────────────────────────────────────
if not os.path.exists(DOCS_DIR):
    print(f"ERROR: No existe la carpeta '{DOCS_DIR}/'. Creala y pon tus PDFs ahi.")
    sys.exit(1)

print(f"Leyendo PDFs desde '{DOCS_DIR}/' ...")
loader = PyPDFDirectoryLoader(DOCS_DIR)
docs   = loader.load()

if not docs:
    print("No se encontraron PDFs. Agrega archivos .pdf a documentos/ e intenta de nuevo.")
    sys.exit(1)

archivos = len(set(d.metadata['source'] for d in docs))
print(f"  {len(docs)} paginas encontradas en {archivos} archivos.")

# ── 2. Dividir en fragmentos ─────────────────────────────────────────────────
print("Dividiendo en fragmentos ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits   = splitter.split_documents(docs)
print(f"  {len(splits)} fragmentos generados.")

# ── 3. Generar embeddings y guardar en chroma_db/ ────────────────────────────
print("Generando embeddings con Google text-embedding-004 via REST ...")
print("(Esto puede tardar unos minutos)")

embeddings = GoogleEmbeddingsREST(api_key=GOOGLE_API_KEY)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print()
print("Indexacion completada exitosamente.")
print(f"  BD vectorial guardada en: {os.path.abspath(CHROMA_DIR)}")
print()
print("Siguiente paso: sube chroma_db/ a git y haz push:")
print("  git add backend/chroma_db backend/documentos/.gitkeep")
print("  git commit -m 'feat: add RAG vector index'")
print("  git push")
