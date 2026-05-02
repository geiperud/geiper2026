import os
import time
import logging
import traceback
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings
    HAS_DEPS = True
except ImportError as e:
    logger.error(f"ImportError: {e}")
    HAS_DEPS = False

app = FastAPI(title="GEIPER AI Cloud Backend")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://geiperud.github.io"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

CHROMA_DIR   = "chroma_db"
GEMINI_MODEL = "gemini-2.0-flash"
EMBED_MODEL  = "gemini-embedding-001"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
EMBED_URL    = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"

vectorstore = None
api_token   = None

# ── Embeddings via REST ──────────────────────────────────────────────────────
class GoogleEmbeddingsREST(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        self.url     = EMBED_URL + f"?key={api_key}"

    def _embed_one(self, text):
        payload = {
            "model": f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": text}]}
        }
        for intento in range(3):
            try:
                resp = requests.post(self.url, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()["embedding"]["values"]
            except Exception as e:
                if intento < 2:
                    time.sleep(2)
                else:
                    raise e

    def embed_documents(self, texts):
        result = []
        for text in texts:
            result.append(self._embed_one(text))
            time.sleep(0.05)
        return result

    def embed_query(self, text):
        return self._embed_one(text)


# ── LLM via REST ─────────────────────────────────────────────────────────────
def gemini_generate(prompt, api_key):
    url = GEMINI_URL + f"?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": "Eres un asistente del grupo de investigacion GEIPER. Responde SIEMPRE en español, sin excepcion."}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.3
        }
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def init_services():
    global vectorstore, api_token

    api_token = os.environ.get("GOOGLE_API_KEY", "")
    if not api_token:
        logger.warning("No se encontro GOOGLE_API_KEY.")
        return

    logger.info("Google API Key encontrada.")

    if not HAS_DEPS:
        logger.warning("Faltan dependencias de LangChain/ChromaDB.")
        return

    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = GoogleEmbeddingsREST(api_key=api_token)
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            logger.info("BD Vectorial cargada.")
        except Exception as e:
            logger.error(f"Error cargando ChromaDB: {e}")

@app.on_event("startup")
def on_startup():
    init_services()

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: str = Field(pattern=r"^(investigacion|tematico)$")

@app.get("/status")
def status():
    return {"status": "ok", "cloud_ready": bool(api_token)}

@app.post("/chat")
def chat(request: ChatRequest):
    if not api_token:
        raise HTTPException(status_code=500, detail="Sin configuracion de Google API.")

    try:
        if request.mode == "investigacion":
            user_prompt = (
                f"Eres el Asistente de Investigacion del grupo GEIPER. "
                f"Responde en español de forma profesional.\n\n"
                f"Pregunta: {request.query}"
            )
        else:
            contexto = ""
            if vectorstore is not None:
                try:
                    docs = vectorstore.similarity_search(request.query, k=3)
                    contexto = "\n\n".join([d.page_content for d in docs])
                    logger.info(f"RAG: {len(docs)} fragmentos encontrados.")
                except Exception as e:
                    logger.warning(f"RAG fallo: {e}")

            if contexto:
                user_prompt = (
                    f"Eres el Asistente Tematico del semillero GEIPER. "
                    f"Usa el siguiente contexto de los documentos para responder en español.\n\n"
                    f"Contexto:\n{contexto}\n\n"
                    f"Pregunta: {request.query}"
                )
            else:
                user_prompt = (
                    f"Eres el Asistente Tematico del semillero GEIPER. "
                    f"Responde en español sobre el grupo GEIPER.\n\n"
                    f"Pregunta: {request.query}"
                )

        logger.info(f"Enviando a Gemini (modo: {request.mode})")
        respuesta = gemini_generate(user_prompt, api_token)
        logger.info("Respuesta recibida de Gemini.")
        return {"response": respuesta}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
