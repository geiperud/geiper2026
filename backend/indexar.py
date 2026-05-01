"""
indexar.py — Ejecutar UNA SOLA VEZ localmente para generar chroma_db/

Pasos:
  1. Pon todos tus PDFs dentro de la carpeta  backend/documentos/
  2. Desde la carpeta backend/ corre:  python indexar.py
  3. Se creará la carpeta  backend/chroma_db/  con el índice vectorial listo
  4. Sube chroma_db/ al repositorio y haz deploy en Render
"""

import os
import sys

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"ERROR de importación: {e}")
    print("Instala con:")
    print("  python -m pip install langchain langchain-community langchain-text-splitters langchain-huggingface chromadb pypdf sentence-transformers")
    sys.exit(1)

DOCS_DIR = "documentos"
CHROMA_DIR = "chroma_db"

# ── 1. Cargar PDFs ──────────────────────────────────────────────────────────
if not os.path.exists(DOCS_DIR):
    print(f"ERROR: No existe la carpeta '{DOCS_DIR}/'. Créala y pon tus PDFs ahí.")
    sys.exit(1)

print(f"Leyendo PDFs desde '{DOCS_DIR}/' ...")
loader = PyPDFDirectoryLoader(DOCS_DIR)
docs = loader.load()

if not docs:
    print("No se encontraron PDFs. Agrega archivos .pdf a la carpeta documentos/ e intenta de nuevo.")
    sys.exit(1)

print(f"  {len(docs)} páginas encontradas en {len(set(d.metadata['source'] for d in docs))} archivos.")

# ── 2. Dividir en fragmentos ─────────────────────────────────────────────────
print("Dividiendo en fragmentos (chunk_size=1000, overlap=200) ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"  {len(splits)} fragmentos generados.")

# ── 3. Generar embeddings y guardar en chroma_db/ ───────────────────────────
print("Generando embeddings y guardando en chroma_db/ (puede tardar unos minutos) ...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
vectorstore.persist()

print()
print(f"Indexación completada exitosamente.")
print(f"  BD vectorial guardada en: {os.path.abspath(CHROMA_DIR)}")
print()
print("Siguiente paso: agrega chroma_db/ a git y haz push para desplegar en Render.")
print("  git add backend/chroma_db backend/documentos/.gitkeep")
print("  git commit -m 'feat: add RAG vector index'")
print("  git push")
