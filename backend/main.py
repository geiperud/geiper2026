import os
import logging
import traceback
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
vectorstore = None
gemini_model = None
api_token = None

def init_services():
    global vectorstore, gemini_model, api_token

    api_token = os.environ.get("GOOGLE_API_KEY", "")
    if not api_token:
        logger.warning("No se encontro GOOGLE_API_KEY.")
        return

    logger.info("Google API Key encontrada.")

    # Configurar Gemini
    try:
        genai.configure(api_key=api_token)
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=genai.GenerationConfig(
                max_output_tokens=500,
                temperature=0.3,
            ),
            system_instruction="Eres un asistente del grupo de investigacion GEIPER. Responde SIEMPRE en español, sin excepcion."
        )
        logger.info("Modelo Gemini 1.5 Flash listo.")
    except Exception as e:
        logger.error(f"Error configurando Gemini: {e}")
        return

    # Cargar ChromaDB con embeddings de Google
    if not HAS_DEPS:
        logger.warning("Faltan dependencias de LangChain/ChromaDB.")
        return

    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_token
            )
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            logger.info("BD Vectorial cargada con Google Embeddings.")
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
    return {"status": "ok", "cloud_ready": bool(api_token and gemini_model)}

@app.post("/chat")
def chat(request: ChatRequest):
    if not api_token or not gemini_model:
        raise HTTPException(status_code=500, detail="Sin configuracion de Google API.")

    try:
        if request.mode == "investigacion":
            user_prompt = (
                f"Eres el Asistente de Investigacion del grupo GEIPER. "
                f"Responde en español de forma profesional y detallada.\n\n"
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
                    logger.warning(f"RAG fallo, respondiendo sin contexto: {e}")
                    contexto = ""

            if contexto:
                user_prompt = (
                    f"Eres el Asistente Tematico del semillero GEIPER. "
                    f"Usa el siguiente contexto extraido de los documentos del grupo para responder en español.\n\n"
                    f"Contexto:\n{contexto}\n\n"
                    f"Pregunta: {request.query}"
                )
            else:
                user_prompt = (
                    f"Eres el Asistente Tematico del semillero GEIPER. "
                    f"Responde en español sobre el grupo GEIPER y sus temas de investigacion.\n\n"
                    f"Pregunta: {request.query}"
                )

        logger.info(f"Enviando prompt a Gemini (modo: {request.mode})")
        response = gemini_model.generate_content(user_prompt)
        respuesta = response.text
        logger.info("Respuesta recibida de Gemini.")
        return {"response": respuesta}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error API: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
