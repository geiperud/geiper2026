import os
import time
import logging
import traceback
import requests
from fastapi import FastAPI, HTTPException, Request, Response
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
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

GLM_MODEL = "glm-4.5-flash"
GLM_URL   = "https://api.z.ai/api/paas/v4/chat/completions"

SALUDOS = {"hola", "hi", "hello", "buenas", "buen día", "buen dia", "buenos días",
           "buenos dias", "hey", "saludos", "qué tal", "que tal", "ola"}

# Referencias APA 7ª edición de los documentos indexados
REFERENCIAS_APA = {
    "cai2005.pdf": (
        "Cai, G., Wang, H., MacEachren, A. M., & Fuhrmann, S. (2005). "
        "Natural conversational interfaces to geospatial databases. "
        "Transactions in GIS, 9(2), 199–221."
    ),
    "wang2008.pdf": (
        "Wang, H., Cai, G., & MacEachren, A. M. (2008). "
        "GeoDialogue: A software agent enabling collaborative dialogues between a user and a conversational GIS. "
        "En Proceedings of the 20th IEEE International Conference on Tools with Artificial Intelligence. IEEE."
    ),
    "GeoLLM-A-specialized-large-language-model-framework-for-intelligent-geotechnical-design.pdf": (
        "Xu, H.-R., Zhang, N., Yin, Z.-Y., & Atangana Njock, P. G. (2025). "
        "GeoLLM: A specialized large language model framework for intelligent geotechnical design. "
        "Computers and Geotechnics, 177, 106849. https://doi.org/10.1016/j.compgeo.2025.106849"
    ),
}

vectorstore = None
api_token   = None
glm_token   = None

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


# ── LLM GLM-4.5-Flash via Z.ai (primario) ───────────────────────────────────
def glm_generate(prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres el Asistente Académico del semillero de investigación GEIPER. "
                    "NORMAS ABSOLUTAS:\n"
                    "- Responde SIEMPRE en español, sin excepción.\n"
                    "- Sé preciso, académico y coherente.\n"
                    "- NUNCA inventes información que no esté en el contexto proporcionado.\n"
                    "- Si no tienes información suficiente, dilo claramente."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.1,
        "stream": False
    }
    resp = requests.post(GLM_URL, json=payload, headers=headers, timeout=90)
    if resp.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="El servicio de IA está temporalmente saturado. Por favor, espera unos segundos e intenta de nuevo."
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── LLM Gemini via REST (fallback) ───────────────────────────────────────────
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
    MAX_REINTENTOS = 2
    for intento in range(MAX_REINTENTOS):
        resp = requests.post(url, json=payload, timeout=25)
        if resp.status_code == 429:
            if intento < MAX_REINTENTOS - 1:
                espera = 5 * (intento + 1)
                logger.warning(f"Rate limit (429), esperando {espera}s antes de reintentar...")
                time.sleep(espera)
                continue
            else:
                logger.error("Rate limit (429) agotado tras reintentos.")
                raise HTTPException(
                    status_code=429,
                    detail="El servicio de IA está temporalmente saturado. Por favor, espera unos segundos e intenta de nuevo."
                )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ── Búsqueda web via DuckDuckGo ──────────────────────────────────────────────
def web_search(query, max_results=3):
    if not HAS_DDG:
        return []
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=max_results, region="es-es"))
        return resultados
    except Exception as e:
        logger.warning(f"Web search fallo: {e}")
        return []


def init_services():
    global vectorstore, api_token, glm_token

    glm_token = os.environ.get("GLM_API_KEY", "")
    if glm_token:
        logger.info("GLM API Key encontrada (modelo primario).")
    else:
        logger.warning("No se encontro GLM_API_KEY.")

    api_token = os.environ.get("GOOGLE_API_KEY", "")
    if api_token:
        logger.info("Google API Key encontrada (modelo fallback).")
    else:
        logger.warning("No se encontro GOOGLE_API_KEY.")

    if not glm_token and not api_token:
        logger.error("No hay ninguna API Key configurada.")
        return

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
    return {"status": "ok", "cloud_ready": bool(glm_token or api_token)}

@app.post("/chat")
def chat(request: ChatRequest):
    if not glm_token and not api_token:
        raise HTTPException(status_code=500, detail="Sin configuracion de API.")

    try:
        if request.mode == "investigacion":
            user_prompt = (
                f"Eres el Asistente de Investigacion del grupo GEIPER. "
                f"Responde en español de forma profesional.\n\n"
                f"Pregunta: {request.query}"
            )
        else:
            # ── Detección de saludo: respuesta directa sin RAG ni web ────────
            es_saludo = request.query.strip().lower().rstrip("!?.") in SALUDOS
            if es_saludo:
                user_prompt = (
                    f"Eres el Asistente Académico del semillero GEIPER. "
                    f"El usuario te saludó. Responde con un saludo breve y amigable en español, "
                    f"y pregúntale sobre cuál de los siguientes temas desea consultar:\n"
                    f"1. Interfaces conversacionales con SIG (Cai et al., 2005; Wang et al., 2008)\n"
                    f"2. Modelos de lenguaje para geotecnia (Xu et al., 2025)\n"
                    f"Sé conciso, no más de 3 líneas."
                )
                if glm_token:
                    try:
                        respuesta = glm_generate(user_prompt, glm_token)
                        return {"response": respuesta}
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.warning(f"GLM fallo en saludo: {e}")
                if api_token:
                    respuesta = gemini_generate(user_prompt, api_token)
                    return {"response": respuesta}

            contexto_docs = ""
            contexto_web  = ""
            fuentes_log   = []

            # ── RAG: documentos indexados ────────────────────────────────────
            if vectorstore is not None:
                try:
                    docs_scores = vectorstore.similarity_search_with_score(request.query, k=3)
                    relevantes  = [(doc, score) for doc, score in docs_scores if score < 1.2]
                    if relevantes:
                        bloques = []
                        for doc, score in relevantes:
                            fuente = os.path.basename(doc.metadata.get("source", "documento"))
                            apa    = REFERENCIAS_APA.get(fuente, fuente)
                            fuentes_log.append(f"{fuente} (score:{score:.2f})")
                            bloques.append(f"[Referencia APA: {apa}]\n{doc.page_content[:500]}")
                        contexto_docs = "\n\n---\n\n".join(bloques)
                        logger.info(f"RAG: {len(relevantes)} fragmentos relevantes: {', '.join(fuentes_log)}")
                    else:
                        logger.info("RAG: ningún fragmento superó el umbral de relevancia.")
                except Exception as e:
                    logger.warning(f"RAG fallo: {e}")

            # ── Web search: DuckDuckGo ───────────────────────────────────────
            resultados_web = web_search(request.query, max_results=2)
            if resultados_web:
                bloques_web = []
                for r in resultados_web:
                    titulo = r.get("title", "")
                    cuerpo = r.get("body", "")[:250]
                    url    = r.get("href", "")
                    bloques_web.append(f"[Web: {titulo} — {url}]\n{cuerpo}")
                contexto_web = "\n\n---\n\n".join(bloques_web)
                logger.info(f"Web search: {len(resultados_web)} resultados encontrados.")

            # ── Construir prompt analítico ───────────────────────────────────
            secciones = []
            if contexto_docs:
                secciones.append(f"DOCUMENTOS ACADÉMICOS INDEXADOS:\n{contexto_docs}")
            if contexto_web:
                secciones.append(f"RESULTADOS DE LA WEB (contexto complementario):\n{contexto_web}")

            if secciones:
                user_prompt = (
                    f"Eres el Asistente Académico del semillero GEIPER.\n\n"
                    f"Tienes acceso a documentos académicos propios del semillero y a resultados "
                    f"actuales de la web. Usa ambas fuentes para dar una respuesta completa, "
                    f"analítica y contextualizada a la pregunta del usuario.\n\n"
                    f"INSTRUCCIONES:\n"
                    f"1. Analiza la pregunta en profundidad y responde de forma clara y académica.\n"
                    f"2. Integra la información de los documentos y la web de manera coherente.\n"
                    f"3. Cita los documentos académicos en formato APA 7ª edición.\n"
                    f"4. Cita las fuentes web con su título y URL.\n"
                    f"5. Al final incluye una sección 'Referencias:' con todas las fuentes usadas.\n"
                    f"6. Responde siempre en español.\n\n"
                    f"{chr(10).join(secciones)}\n\n"
                    f"PREGUNTA: {request.query}"
                )
            else:
                user_prompt = (
                    f"Eres el Asistente Académico del semillero GEIPER.\n\n"
                    f"No encontraste información específica en los documentos ni en la web "
                    f"para esta consulta. Responde en español indicando esto y sugiere "
                    f"preguntar sobre los temas del semillero: interfaces conversacionales "
                    f"con SIG, razonamiento en LLMs y geotecnia con inteligencia artificial.\n\n"
                    f"PREGUNTA: {request.query}"
                )

        # Intentar con GLM-4.5-Flash primero
        if glm_token:
            try:
                logger.info(f"Enviando a GLM-4.5-Flash (modo: {request.mode})")
                respuesta = glm_generate(user_prompt, glm_token)
                logger.info("Respuesta recibida de GLM.")
                return {"response": respuesta}
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"GLM fallo, intentando con Gemini: {e}")

        # Fallback: Gemini
        if not api_token:
            raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")

        logger.info(f"Enviando a Gemini fallback (modo: {request.mode})")
        respuesta = gemini_generate(user_prompt, api_token)
        logger.info("Respuesta recibida de Gemini.")
        return {"response": respuesta}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
