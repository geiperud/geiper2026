import os
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
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from langchain_chroma import Chroma
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

CHROMA_DIR = "chroma_db"
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
vectorstore = None
api_token = None

def init_services():
    global vectorstore, api_token

    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not api_token:
        logger.warning("No se encontró HUGGINGFACEHUB_API_TOKEN.")
        return

    logger.info("Token HuggingFace encontrado.")

    if not HAS_DEPS:
        logger.warning("Faltan dependencias de LangChain/ChromaDB.")
        return

    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=api_token,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
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
        raise HTTPException(status_code=500, detail="Sin token HuggingFace.")

    try:
        # Construir prompt según modo
        if request.mode == "investigacion":
            user_prompt = (
                f"Eres el Asistente de Investigación del grupo GEIPER. "
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
                    logger.warning(f"RAG falló, respondiendo sin contexto: {e}")
                    contexto = ""
            user_prompt = (
                f"Eres el Asistente Temático del semillero GEIPER. "
                f"Usa el contexto para responder en español.\n\n"
                f"Contexto:\n{contexto}\n\nPregunta: {request.query}"
            )

        # Llamada directa a HuggingFace API (formato OpenAI /v1/chat/completions)
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [
                {"role": "system", "content": "Eres un asistente del grupo GEIPER. Responde siempre en español."},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        logger.info(f"HF status: {response.status_code}")
        logger.info(f"HF body: {response.text[:300]}")

        response.raise_for_status()
        data = response.json()
        respuesta = data["choices"][0]["message"]["content"]
        return {"response": respuesta}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error API: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
