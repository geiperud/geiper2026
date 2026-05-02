"""
indexar.py — Ejecutar UNA SOLA VEZ localmente para generar chroma_db/

Formatos soportados:
  - PDF     (.pdf)
  - Word    (.docx)
  - Texto   (.txt)
  - CSV     (.csv)
  - Markdown(.md)

Pasos:
  1. Pon tus archivos dentro de la carpeta  backend/documentos/
     (puedes crear subcarpetas: documentos/articulos/, documentos/datos/, etc.)
  2. Desde la carpeta backend/ corre:
       $env:GOOGLE_API_KEY="tu_clave"   (PowerShell)
       set GOOGLE_API_KEY=tu_clave      (CMD)
  3. python indexar.py
  4. Sube chroma_db/ al repositorio y haz deploy en Render
"""

import os
import sys
import time
import requests
import glob

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
        CSVLoader,
        UnstructuredMarkdownLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"ERROR de importacion: {e}")
    print("Instala con:")
    print("  python -m pip install langchain-community langchain-text-splitters langchain-chroma chromadb pypdf docx2txt unstructured")
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

# ── Clase de embeddings via REST ─────────────────────────────────────────────
class GoogleEmbeddingsREST:
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
            time.sleep(0.1)
        return embeddings

    def embed_query(self, text):
        return self._embed_one(text)


# ── 1. Cargar documentos (todos los formatos) ────────────────────────────────
if not os.path.exists(DOCS_DIR):
    print(f"ERROR: No existe la carpeta '{DOCS_DIR}/'.")
    sys.exit(1)

# Extensiones soportadas y sus loaders
LOADERS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
}

docs = []
archivos_cargados = {"pdf": 0, "docx": 0, "txt": 0, "csv": 0, "md": 0}

print(f"Buscando documentos en '{DOCS_DIR}/' (incluye subcarpetas)...")

# Buscar archivos recursivamente en todas las subcarpetas
for ext, LoaderClass in LOADERS.items():
    patron = os.path.join(DOCS_DIR, "**", f"*{ext}")
    archivos = glob.glob(patron, recursive=True)
    for archivo in archivos:
        try:
            if ext == ".txt" or ext == ".md":
                loader = LoaderClass(archivo, encoding="utf-8")
            else:
                loader = LoaderClass(archivo)
            paginas = loader.load()
            docs.extend(paginas)
            tipo = ext.replace(".", "")
            archivos_cargados[tipo] = archivos_cargados.get(tipo, 0) + 1
            print(f"  OK: {os.path.basename(archivo)} ({len(paginas)} paginas/secciones)")
        except Exception as e:
            print(f"  ADVERTENCIA: No se pudo cargar {archivo}: {e}")

# CSV por separado (cada fila es un documento)
patron_csv = os.path.join(DOCS_DIR, "**", "*.csv")
archivos_csv = glob.glob(patron_csv, recursive=True)
for archivo in archivos_csv:
    try:
        loader = CSVLoader(archivo, encoding="utf-8")
        filas = loader.load()
        docs.extend(filas)
        archivos_cargados["csv"] += 1
        print(f"  OK: {os.path.basename(archivo)} ({len(filas)} filas)")
    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo cargar {archivo}: {e}")

if not docs:
    print()
    print("No se encontraron documentos. Formatos soportados:")
    print("  .pdf  .docx  .txt  .csv  .md")
    sys.exit(1)

print()
print("Resumen de archivos cargados:")
for tipo, cantidad in archivos_cargados.items():
    if cantidad > 0:
        print(f"  .{tipo}: {cantidad} archivo(s)")
print(f"  Total de paginas/secciones: {len(docs)}")

# ── 2. Dividir en fragmentos ─────────────────────────────────────────────────
print()
print("Dividiendo en fragmentos ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits   = splitter.split_documents(docs)
print(f"  {len(splits)} fragmentos generados.")

# ── 3. Generar embeddings y guardar en chroma_db/ ────────────────────────────
print()
print("Generando embeddings con Google gemini-embedding-001 via REST ...")
print("(Esto puede tardar unos minutos segun la cantidad de documentos)")
print()

embeddings  = GoogleEmbeddingsREST(api_key=GOOGLE_API_KEY)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print()
print("Indexacion completada exitosamente.")
print(f"  BD vectorial guardada en: {os.path.abspath(CHROMA_DIR)}")
print(f"  Total de fragmentos indexados: {len(splits)}")
print()
print("Siguiente paso: sube chroma_db/ a git y haz push:")
print("  git add backend/chroma_db backend/documentos/.gitkeep")
print("  git commit -m 'feat: actualizar indice RAG'")
print("  git push")
