import os
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
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
DOCS_DIR = "documentos"
vectorstore = None
hf_client = None

def init_services():
    global vectorstore, hf_client

    if not HAS_DEPS:
        logger.warning("Faltan dependencias.")
        return

    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not api_token:
        logger.warning("No se encontró HUGGINGFACEHUB_API_TOKEN.")
        return

    try:
        hf_client = InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",
            token=api_token
        )
        logger.info("Cliente HuggingFace listo.")
    except Exception as e:
        logger.error(f"Error creando cliente HF: {e}")
        return

    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=api_token,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            logger.info("BD Vectorial cargada desde disco.")
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
    return {"status": "ok", "cloud_ready": hf_client is not None}

@app.post("/chat")
def chat(request: ChatRequest):
    if hf_client is None:
        raise HTTPException(status_code=500, detail="Backend sin conexión a HuggingFace.")

    try:
        if request.mode == "investigacion":
            prompt = (
                f"Eres el Asistente de Investigación del grupo GEIPER. "
                f"Responde en español de forma profesional.\n\n"
                f"Pregunta: {request.query}\nRespuesta:"
            )
        elif request.mode == "tematico":
            contexto = ""
            if vectorstore is not None:
                docs = vectorstore.similarity_search(request.query, k=3)
                contexto = "\n\n".join([d.page_content for d in docs])

            prompt = (
                f"Eres el Asistente Temático del semillero GEIPER. "
                f"Usa el contexto para responder amablemente en español.\n\n"
                f"Contexto:\n{contexto}\n\n"
                f"Pregunta: {request.query}\nRespuesta:"
            )
        else:
            raise HTTPException(status_code=400, detail="Modo inválido.")

        result = hf_client.text_generation(
            prompt,
            max_new_tokens=500,
            temperature=0.3,
            return_full_text=False
        )
        return {"response": result}

    except Exception as e:
        logger.error(f"Error API: {e}")
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
