"""
indexar.py — Ejecutar UNA SOLA VEZ localmente para generar chroma_db/

Pasos:
  1. Pon todos tus PDFs dentro de la carpeta  backend/documentos/
  2. Desde la carpeta backend/ corre:  python indexar.py
  3. Se creara la carpeta  backend/chroma_db/  con el indice vectorial listo
  4. Sube chroma_db/ al repositorio y haz deploy en Render
"""

import os
import sys

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError as e:
    print(f"ERROR de importacion: {e}")
    print("Instala con:")
    print("  python -m pip install langchain-community langchain-text-splitters langchain-chroma langchain-google-genai chromadb pypdf google-generativeai")
    sys.exit(1)

DOCS_DIR   = "documentos"
CHROMA_DIR = "chroma_db"

# ── Leer clave de Google ─────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("ERROR: Define la variable de entorno GOOGLE_API_KEY antes de correr este script.")
    print("  En Windows:  set GOOGLE_API_KEY=tu_clave_aqui")
    print("  En Mac/Linux: export GOOGLE_API_KEY=tu_clave_aqui")
    sys.exit(1)

# ── 1. Cargar PDFs ───────────────────────────────────────────────────────────
if not os.path.exists(DOCS_DIR):
    print(f"ERROR: No existe la carpeta '{DOCS_DIR}/'. Creala y pon tus PDFs ahi.")
    sys.exit(1)

print(f"Leyendo PDFs desde '{DOCS_DIR}/' ...")
loader = PyPDFDirectoryLoader(DOCS_DIR)
docs = loader.load()

if not docs:
    print("No se encontraron PDFs. Agrega archivos .pdf a la carpeta documentos/ e intenta de nuevo.")
    sys.exit(1)

print(f"  {len(docs)} paginas encontradas en {len(set(d.metadata['source'] for d in docs))} archivos.")

# ── 2. Dividir en fragmentos ─────────────────────────────────────────────────
print("Dividiendo en fragmentos (chunk_size=1000, overlap=200) ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"  {len(splits)} fragmentos generados.")

# ── 3. Generar embeddings con Google y guardar en chroma_db/ ─────────────────
print("Generando embeddings con Google text-embedding-004 ...")
print("(Esto puede tardar un momento segun el numero de documentos)")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print()
print("Indexacion completada exitosamente.")
print(f"  BD vectorial guardada en: {os.path.abspath(CHROMA_DIR)}")
print()
print("Siguiente paso: agrega chroma_db/ a git y haz push para desplegar en Render.")
print("  git add backend/chroma_db backend/documentos/.gitkeep")
print("  git commit -m 'feat: add RAG vector index with Google embeddings'")
print("  git push")
