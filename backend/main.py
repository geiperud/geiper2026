import os
import re
import time
import logging
import traceback
import requests
from typing import List
from fastapi import FastAPI, HTTPException, Request, Response
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings
    HAS_DEPS = True
except ImportError as e:
    logger.error(f"ImportError: {e}")
    HAS_DEPS = False

app = FastAPI(title="GEIPER AI Cloud Backend")

# ── Rate limiting: máximo de peticiones por IP para evitar abuso/saturación ──
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://geiperud.github.io"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

FAISS_DIR = "faiss_index"
FAISS_DIR_INVESTIGACION = "faiss_index_investigacion"
GEMINI_MODEL = "gemini-2.0-flash"
EMBED_MODEL  = "gemini-embedding-001"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
EMBED_URL    = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

SALUDOS = {"hola", "hi", "hello", "buenas", "buen día", "buen dia", "buenos días",
           "buenos dias", "hey", "saludos", "qué tal", "que tal", "ola"}

# ── Filtro de contenido: obscenidades y alcance temático ─────────────────────
PALABRAS_BLOQUEADAS = {
    # Insultos y groserías comunes (español, con variantes usadas en Colombia)
    "puta", "puto", "putas", "putos", "putica", "putico",
    "hijueputa", "hijoeputa", "hijo de puta", "hpta", "gonorrea",
    "malparido", "malparida", "malparidos", "malparidas",
    "marica", "maricon", "maricón", "mariconcito",
    "mierda", "mierdas", "mierdero", "mierdera",
    "coño", "cono e", "pendejo", "pendeja", "pendejada",
    "gilipollas", "capullo", "imbecil", "imbécil", "idiota",
    "verga", "vergas", "pinche", "cabron", "cabrón", "cabrona",
    "zorra", "zorras", "perra", "perras",

    # Contenido sexual explícito
    "follar", "sexo explicito", "sexo explícito",
    "pornografia", "pornografía", "porno", "desnudo", "desnuda", "desnudos",
    "desnudas", "masturbar", "masturbacion", "masturbación",
    "prostituta", "prostitucion", "prostitución",

    # Lenguaje discriminatorio / de odio (por origen, orientación, discapacidad, etc.)
    "negro de mierda", "indio de mierda", "sudaca",
    "retrasado mental", "retrasada mental", "mongolico", "mongólico",

    # Drogas ilícitas (fuera del alcance del semillero, riesgo de mal uso)
    "como fabricar droga", "como hacer droga", "receta de droga",
}

LINEAS_TEMATICAS_DESC = (
    "las líneas de investigación oficiales del semillero GEIPER (Sistemas de "
    "Información Geográfica -SIG-, geomática y percepción remota), y en general "
    "cualquier tema propio de la carrera de Ingeniería Catastral y Geodesia "
    "(geodesia, cartografía, fotogrametría, topografía, agrimensura, catastro "
    "multipropósito, avalúos, ordenamiento territorial, teledetección, "
    "procesamiento digital de imágenes satelitales, modelos de datos "
    "geoespaciales como LADM_COL); los trabajos de tesis e investigaciones ya "
    "realizados dentro del semillero GEIPER; temas académicos de ingeniería "
    "geoespacial, GeoIA y geomática (incluyendo modelos de lenguaje aplicados a "
    "estas áreas); metodología de la investigación, normativa y procedimientos "
    "de la Oficina de Investigaciones de la Universidad Distrital Francisco "
    "José de Caldas, formatos y trámites de trabajos de grado, y estructura "
    "organizativa de la universidad; e información sobre el propio semillero "
    "GEIPER como organización (integrantes, líneas de investigación, eventos, "
    "repositorios) y sobre el sitio web y sus secciones en general"
)

def _normalizar_texto(texto):
    """Normaliza para atrapar variaciones simples: minusculas, sin acentos, espacios colapsados."""
    texto = texto.strip().lower()
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for tilde, plano in reemplazos.items():
        texto = texto.replace(tilde, plano)
    return " ".join(texto.split())

# Precalculamos la version normalizada, y compilamos un patron con limites de
# palabra (\b) para evitar falsos positivos por coincidencia de subcadena
# (ej. "coger" dentro de "recoger" o "escoger").
_PALABRAS_BLOQUEADAS_NORM = sorted(
    {_normalizar_texto(p) for p in PALABRAS_BLOQUEADAS},
    key=len, reverse=True
)
_PATRON_BLOQUEADO = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in _PALABRAS_BLOQUEADAS_NORM) + r')\b'
)

def contiene_contenido_bloqueado(texto):
    texto_norm = _normalizar_texto(texto)
    return bool(_PATRON_BLOQUEADO.search(texto_norm))

# Referencias APA 7ª edición de los documentos indexados
REFERENCIAS_APA = {
    "ValbuenaGaonaMarthaPatricia2020.pdf": (
        "Valbuena Gaona, M. P. (2020). Propuesta metodológica para la "
        "estandarización e incorporación de información ráster y LiDAR a la "
        "plataforma del Sistema de Información Territorial para la Construcción "
        "y Operación-SITCO [Trabajo de grado, Universidad Distrital Francisco "
        "José de Caldas]."
    ),
    "PalominoEscobarDanielFernando2020.pdf": (
        "Palomino Escobar, D. F., & Guerrero Guio, Y. F. (2020). Evaluación del "
        "método Split-Spectrum con imágenes InSAR para la estimación de "
        "diferenciales ionosféricos (ΔTEC) en los departamentos de Cundinamarca "
        "y Boyacá entre los años 2007 y 2010 [Trabajo de grado, Universidad "
        "Distrital Francisco José de Caldas]."
    ),
    "MeloCristanchoJimyAndersson2015.pdf": (
        "Melo Cristancho, J. A. (2015). Metodología detallada de estructuración "
        "y adecuación de cartografía para planes de manejo de perforación de "
        "pozos exploratorios, perforación de pozos de desarrollo o producción y "
        "sus líneas de flujo, en la explotación de hidrocarburos [Trabajo de "
        "grado, Universidad Distrital Francisco José de Caldas]."
    ),
    "ForeroZapataSebastian2024.pdf": (
        "Forero Zapata, S. (2024). Evaluación de redes convolucionales para la "
        "segmentación de objetos geográficos: Un insumo para la cartografía "
        "básica a escala 1:2000 basado en el catálogo del IGAC [Monografía, "
        "Universidad Distrital Francisco José de Caldas]."
    ),
    "ChávezBustosAndrésGuillermo2020.pdf": (
        "Vargas Rodríguez, L. M., & Chávez Bustos, A. G. (2020). Protección de "
        "bosques en el marco de la metodología Fit for Purpose. Caso de "
        "estudio: vereda Termales, Vista Hermosa, Meta [Trabajo de grado, "
        "Universidad Distrital Francisco José de Caldas]."
    ),
}

# Referencias de los documentos del Asistente de Investigación (metodología,
# normativa institucional). Se completa igual que REFERENCIAS_APA cuando
# tengas los PDFs reales en GEIPER_Documentos_Investigacion.
REFERENCIAS_APA_INVESTIGACION = {}

vectorstore = None
vectorstore_investigacion = None
api_token   = None
groq_token   = None

# ── Embeddings via REST ──────────────────────────────────────────────────────
class GoogleEmbeddingsREST(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        self.url     = EMBED_URL + f"?key={api_key}"

    def _embed_one(self, text, task_type):
        payload = {
            "model": f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
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
            result.append(self._embed_one(text, "RETRIEVAL_DOCUMENT"))
            time.sleep(0.05)
        return result

    def embed_query(self, text):
        return self._embed_one(text, "RETRIEVAL_QUERY")


# ── LLM Groq (Llama 3.3 70B) vía API REST (primario) ────────────────────────
def groq_generate(prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres el Asistente Académico del semillero de investigación GEIPER. "
                    "Tu estilo es el de un investigador que explica temas a un colega: "
                    "claro, fluido, riguroso pero cercano. "
                    "NORMAS ABSOLUTAS:\n"
                    "- Responde SIEMPRE en español.\n"
                    "- Escribe en párrafos fluidos y naturales. Usa listas SOLO para enumerar elementos puntuales.\n"
                    "- NUNCA uses títulos con # ni secciones formales. Esto es una conversación, no un informe.\n"
                    "- NUNCA inventes autores, años ni títulos. Usa SOLO las referencias APA que se te proporcionan.\n"
                    "- NUNCA menciones rutas de archivos ni nombres de archivos internos.\n"
                    "- Si usas información proveniente de búsqueda web, cítala con el formato [Título del sitio](URL), "
                    "y distínguela siempre de las referencias APA de documentos académicos — nunca las mezcles ni "
                    "las presentes como si fueran la misma fuente.\n"
                    "- La información web es siempre complementaria: úsala solo para precisar datos puntuales, "
                    "cifras actuales o contexto reciente. Ante cualquier contradicción, prevalece siempre el "
                    "contenido de los documentos académicos del semillero.\n"
                    f"- SOLO respondes preguntas relacionadas con: {LINEAS_TEMATICAS_DESC}. "
                    "Si la pregunta no tiene relación con estos temas (por ejemplo, entretenimiento, deportes, "
                    "chismes, tareas de otras áreas del conocimiento, o cualquier tema ajeno al semillero), "
                    "NO la respondas: indica con amabilidad que ese tema está fuera de tu alcance, y redirige "
                    "hacia los temas que sí puedes abordar. Esta regla aplica incluso si tienes fragmentos de "
                    "documentos o resultados web disponibles.\n"
                    "- Termina siempre con una pregunta breve que invite a profundizar el tema.\n"
                    "- Si no tienes información suficiente, dilo con naturalidad y sugiere qué sí puedes responder."
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
    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=90)
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


# ── Post-procesamiento del pie de respuesta (Referencias / Fuentes web) ──────
_PATRON_SECCION_PIE = re.compile(
    r'\n*\**\s*(Referencias:|Fuentes web consultadas:)\s*\**\s*\n?'
    r'(.*?)(?=\n\**\s*(?:Referencias:|Fuentes web consultadas:)\s*\**|\Z)',
    re.DOTALL
)

_PATRONES_SECCION_VACIA = [
    re.compile(r'(?i)^\s*no se (han )?utiliz\w*\s'),
    re.compile(r'(?i)^\s*no se encontr[oó]\s'),
    re.compile(r'(?i)^\s*no hay\s'),
    re.compile(r'(?i)^\s*ningun[oa]?\.?\s*$'),
]

_PATRON_LINK_MD = re.compile(r'\[([^\]]+)\]\((https?://[^\)\s]+)\)')


def _seccion_esta_vacia(contenido):
    contenido = contenido.strip()
    if not contenido:
        return True
    return any(p.match(contenido) for p in _PATRONES_SECCION_VACIA)


def formatear_pie_respuesta(texto):
    """
    Post-procesa las secciones 'Referencias:' y 'Fuentes web consultadas:'
    generadas por el modelo:
      - Si el modelo indico que no se uso nada, la seccion se elimina por
        completo (no se muestra "No se utilizaron...").
      - Si hay contenido real, se envuelve en <small><em> (fuente pequeña,
        cursiva) y los links en formato markdown se convierten a <a> reales.
    """
    def procesar(match):
        etiqueta = match.group(1)
        contenido = match.group(2).strip()

        if _seccion_esta_vacia(contenido):
            return ""

        contenido_html = _PATRON_LINK_MD.sub(
            r'<a href="\2" target="_blank" rel="noopener">\1</a>', contenido
        )
        return f'\n\n<small><em><strong>{etiqueta}</strong> {contenido_html}</em></small>'

    return _PATRON_SECCION_PIE.sub(procesar, texto).strip()


def buscar_contexto_documentos(vectorstore_local, referencias_apa, query_actual, consulta_efectiva):
    """
    Busca en el índice FAISS dado y arma el bloque de contexto con citas APA.

    Prueba DOS versiones de la búsqueda: la pregunta tal cual la escribió el
    usuario, y la version combinada con el turno anterior (consulta_efectiva).
    Esto evita que preguntas claras y autosuficientes ("monografia") se vean
    perjudicadas por quedar mezcladas con contexto previo no relacionado,
    mientras que preguntas ambiguas con pronombres ("¿lo menciona algun
    trabajo?") siguen beneficiándose de la combinación.
    """
    if vectorstore_local is None:
        return ""
    try:
        candidatos = {}
        consultas_a_probar = [query_actual]
        if consulta_efectiva != query_actual:
            consultas_a_probar.append(consulta_efectiva)

        for consulta in consultas_a_probar:
            docs_scores = vectorstore_local.similarity_search_with_score(consulta, k=5)
            for doc, score in docs_scores:
                clave = f"{doc.metadata.get('source', '')}|{doc.metadata.get('page', '')}|{doc.page_content[:50]}"
                if clave not in candidatos or score < candidatos[clave][1]:
                    candidatos[clave] = (doc, score)

        relevantes = sorted(
            [(doc, score) for doc, score in candidatos.values() if score < 1.6],
            key=lambda par: par[1]
        )[:5]

        if not relevantes:
            logger.info("RAG: ningún fragmento superó el umbral de relevancia (en ninguna de las dos consultas).")
            return ""

        bloques = []
        fuentes_log = []
        for doc, score in relevantes:
            fuente = os.path.basename(doc.metadata.get("source", "documento"))
            apa = referencias_apa.get(fuente, fuente)
            fuentes_log.append(f"{fuente} (score:{score:.2f})")
            bloques.append(f"[Referencia APA: {apa}]\n{doc.page_content[:500]}")
        logger.info(f"RAG: {len(relevantes)} fragmentos relevantes: {', '.join(fuentes_log)}")
        return "\n\n---\n\n".join(bloques)
    except Exception as e:
        logger.warning(f"RAG falló: {e}")
        return ""


def buscar_contexto_web(consulta):
    """Busca en la web (DuckDuckGo) y arma el bloque de contexto complementario."""
    resultados_web = web_search(consulta, max_results=3)
    if not resultados_web:
        return ""
    bloques_web = []
    for r in resultados_web:
        titulo = r.get("title", "Fuente web")
        url = r.get("href", "")
        snippet = (r.get("body", "") or "")[:300]
        if url:
            bloques_web.append(f"[Fuente web: {titulo} — {url}]\n{snippet}")
    if bloques_web:
        logger.info(f"Web search: {len(resultados_web)} resultados encontrados.")
    return "\n\n---\n\n".join(bloques_web)


def construir_prompt_conversacional(contexto_docs, contexto_web, transcripcion, query):
    """
    Arma el prompt final combinando documentos (fuente principal) + web
    (complemento), con las instrucciones de cita correspondientes. Se usa
    igual para el Asistente Temático y el Asistente de Investigación —
    cada uno le pasa su propio contexto ya buscado en su propio índice.
    """
    partes_contexto = []
    instrucciones_citas = []

    if contexto_docs:
        partes_contexto.append(f"FRAGMENTOS DE DOCUMENTOS DEL SEMILLERO:\n{contexto_docs}")
        instrucciones_citas.append(
            "Para los fragmentos de documentos, usa la referencia APA exacta que aparece "
            "entre corchetes, sin modificarla. Si alguno de los fragmentos no es realmente "
            "relevante para responder la pregunta, ignóralo por completo y no lo cites."
        )

    if contexto_web:
        partes_contexto.append(
            f"RESULTADOS DE BÚSQUEDA WEB (respaldo opcional, no fuente principal):\n{contexto_web}"
        )
        instrucciones_citas.append(
            "Responde principalmente con los fragmentos de documentos. Usa la información web "
            "ÚNICAMENTE si de verdad hace falta para completar la respuesta de forma fluida — por "
            "ejemplo, si los documentos no cubren un dato puntual o una cifra actual que la pregunta "
            "necesita. Si los documentos ya responden bien por sí solos, ignora la búsqueda web por "
            "completo y no la menciones. Cuando sí la uses, cita el sitio con el formato "
            "[Título del sitio](URL), y nunca la uses para contradecir a los documentos."
        )

    if partes_contexto:
        contexto_total = "\n\n===\n\n".join(partes_contexto)
        return (
            f"Tienes acceso a fragmentos de documentos del semillero GEIPER "
            f"y, de forma complementaria, a resultados de búsqueda web relacionados con la pregunta.\n\n"
            f"{' '.join(instrucciones_citas)} "
            f"Prioriza siempre los documentos como fuente principal.\n\n"
            f"Responde de forma conversacional y académica: párrafos fluidos, sin títulos con #, "
            f"listas solo cuando sean estrictamente necesarias. Integra la información con análisis propio. "
            f"Cierra tu explicación con una pregunta breve que invite a seguir conversando. "
            f"DESPUÉS de esa pregunta, y solo al final de todo el mensaje, agrega dos apartados en "
            f"texto plano (sin negrita, sin asteriscos): 'Referencias:' seguido de las citas APA, y "
            f"'Fuentes web consultadas:' seguido de los enlaces en formato [Título](URL). Estos dos "
            f"apartados SIEMPRE deben ser lo último del mensaje, nunca antes de la pregunta de cierre. "
            f"IMPORTANTE: si no usaste documentos, OMITE por completo el apartado 'Referencias:' "
            f"— no lo escribas ni indiques que no se usó nada. Lo mismo aplica para "
            f"'Fuentes web consultadas:' si no usaste ninguna fuente web: simplemente no la incluyas.\n\n"
            f"{transcripcion}"
            f"{contexto_total}\n\n"
            f"PREGUNTA: {query}"
        )

    return (
        f"No se encontró información relevante ni en los documentos ni en la búsqueda web "
        f"para esta consulta. Responde con naturalidad indicando que no tienes información "
        f"específica sobre eso, y sugiere qué temas sí puedes abordar dentro de tu especialidad. "
        f"Sé breve y amigable.\n\n"
        f"{transcripcion}"
        f"PREGUNTA: {query}"
    )


def init_services():
    global vectorstore, vectorstore_investigacion, api_token, groq_token

    groq_token  = os.environ.get("GROQ_API_KEY", "")
    if groq_token:
        logger.info("Groq API Key encontrada (modelo primario).")
    else:
        logger.warning("No se encontro GROQ_API_KEY.")

    api_token = os.environ.get("GOOGLE_API_KEY", "")
    if api_token:
        logger.info("Google API Key encontrada (modelo fallback + embeddings RAG).")
    else:
        logger.warning("No se encontro GOOGLE_API_KEY.")

    if not groq_token and not api_token:
        logger.error("No hay ninguna API Key configurada.")
        return

    if not HAS_DEPS:
        logger.warning("Faltan dependencias de LangChain/FAISS.")
        return

    if not api_token:
        logger.warning("GOOGLE_API_KEY no configurada: no se puede cargar FAISS (embeddings via Gemini REST).")
        return

    embeddings = GoogleEmbeddingsREST(api_key=api_token)

    # ── Índice del Asistente Temático ─────────────────────────────────────
    if os.path.exists(FAISS_DIR):
        try:
            vectorstore = FAISS.load_local(
                FAISS_DIR, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("BD Vectorial FAISS (Temático) cargada.")
        except Exception as e:
            logger.error(f"Error cargando FAISS (Temático): {e}")
    else:
        logger.warning(f"No se encontró la carpeta '{FAISS_DIR}'. El Asistente Temático no tendrá documentos.")

    # ── Índice del Asistente de Investigación ─────────────────────────────
    if os.path.exists(FAISS_DIR_INVESTIGACION):
        try:
            vectorstore_investigacion = FAISS.load_local(
                FAISS_DIR_INVESTIGACION, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("BD Vectorial FAISS (Investigación) cargada.")
        except Exception as e:
            logger.error(f"Error cargando FAISS (Investigación): {e}")
    else:
        logger.info(
            f"No se encontró la carpeta '{FAISS_DIR_INVESTIGACION}'. "
            "El Asistente de Investigación aún no tiene índice (normal hasta que lo generes)."
        )

@app.on_event("startup")
def on_startup():
    init_services()

class Turno(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: str = Field(pattern=r"^(investigacion|tematico)$")
    # Ventana deslizante: el frontend solo manda los ultimos 3 intercambios
    # (6 turnos). Esto acota el tamaño de cada peticion sin importar que tan
    # larga sea la conversacion completa en pantalla.
    historial: List[Turno] = Field(default_factory=list, max_length=6)


def construir_transcripcion(historial):
    """Convierte el historial reciente en texto plano para dar contexto al LLM."""
    if not historial:
        return ""
    lineas = []
    for turno in historial:
        etiqueta = "Usuario" if turno.role == "user" else "Asistente"
        lineas.append(f"{etiqueta}: {turno.content}")
    return "HISTORIAL RECIENTE DE LA CONVERSACIÓN (para contexto, no la repitas):\n" + "\n".join(lineas) + "\n\n"


def construir_consulta_efectiva(chat_request):
    """
    Combina el ultimo turno del usuario con la pregunta actual, para que las
    preguntas de seguimiento con pronombres ('lo', 'eso', 'ese tema') sigan
    encontrando contexto relevante al buscar en documentos y en la web.
    """
    ultimo_turno_usuario = next(
        (t.content for t in reversed(chat_request.historial) if t.role == "user"),
        None
    )
    if ultimo_turno_usuario:
        return f"{ultimo_turno_usuario} {chat_request.query}"
    return chat_request.query

@app.get("/status")
def status():
    return {"status": "ok", "cloud_ready": bool(groq_token or api_token)}

@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, chat_request: ChatRequest):
    if not groq_token and not api_token:
        raise HTTPException(status_code=500, detail="Sin configuracion de API.")

    # ── Filtro de obscenidades: corte inmediato, sin gastar API ni web ───────
    if contiene_contenido_bloqueado(chat_request.query):
        logger.info("Consulta bloqueada por filtro de contenido.")
        return {
            "response": (
                "Prefiero que mantengamos un tono respetuoso en esta conversación. "
                "Estoy aquí para ayudarte con temas del semillero GEIPER: SIG, geomática, "
                "percepción remota, catastro multipropósito y áreas afines. "
                "¿En qué puedo ayudarte dentro de esos temas?"
            )
        }

    try:
        transcripcion = construir_transcripcion(chat_request.historial)
        consulta_efectiva = construir_consulta_efectiva(chat_request)

        # ── Detección de saludo: respuesta directa sin RAG ni web ────────────
        es_saludo = chat_request.query.strip().lower().rstrip("!?.") in SALUDOS
        if es_saludo:
            if chat_request.mode == "investigacion":
                saludo_prompt = (
                    f"Eres el Asistente de Investigación del semillero GEIPER. "
                    f"El usuario te saludó. Responde con un saludo breve y amigable en español, "
                    f"y pregúntale sobre qué tema de metodología de investigación, normativa "
                    f"institucional, trámites de trabajos de grado o estructura de la universidad "
                    f"desea consultar. Sé conciso, no más de 3 líneas."
                )
            else:
                saludo_prompt = (
                    f"Eres el Asistente Temático del semillero GEIPER. "
                    f"El usuario te saludó. Responde con un saludo breve y amigable en español, "
                    f"y pregúntale sobre cuál de los siguientes temas desea consultar:\n"
                    f"1. Interfaces conversacionales con SIG (Cai et al., 2005; Wang et al., 2008)\n"
                    f"2. Modelos de lenguaje para geotecnia (Xu et al., 2025)\n"
                    f"Sé conciso, no más de 3 líneas."
                )
            if groq_token:
                try:
                    respuesta = groq_generate(saludo_prompt, groq_token)
                    return {"response": respuesta}
                except HTTPException:
                    raise
                except Exception as e:
                    logger.warning(f"Groq fallo en saludo: {e}")
            if api_token:
                respuesta = gemini_generate(saludo_prompt, api_token)
                return {"response": respuesta}

        # ── Selecciona el índice y las referencias según el asistente activo ─
        if chat_request.mode == "investigacion":
            vectorstore_activo   = vectorstore_investigacion
            referencias_activas  = REFERENCIAS_APA_INVESTIGACION
        else:
            vectorstore_activo   = vectorstore
            referencias_activas  = REFERENCIAS_APA

        # ── Primero documentos (fuente principal), web como complemento fluido ─
        # El orden de ejecucion ya garantiza que los documentos se consultan
        # antes que la web. La prioridad real (documentos > web) la aplica el
        # modelo via las instrucciones del prompt, no un bloqueo binario aqui,
        # para que la web siga pudiendo aportar datos puntuales o actuales
        # incluso cuando los documentos ya respondieron parte de la pregunta.
        contexto_docs = buscar_contexto_documentos(
            vectorstore_activo, referencias_activas, chat_request.query, consulta_efectiva
        )
        contexto_web = buscar_contexto_web(consulta_efectiva)
        user_prompt = construir_prompt_conversacional(contexto_docs, contexto_web, transcripcion, chat_request.query)

        # ── Generar respuesta (Groq primero, Gemini fallback) ─────────────────
        respuesta = None
        if groq_token:
            try:
                logger.info(f"Enviando a Groq (Llama 3.3 70B) (modo: {chat_request.mode})")
                respuesta = groq_generate(user_prompt, groq_token)
                logger.info("Respuesta recibida de Groq.")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Groq fallo, intentando con Gemini: {e}")

        if respuesta is None:
            if not api_token:
                raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
            logger.info(f"Enviando a Gemini fallback (modo: {chat_request.mode})")
            respuesta = gemini_generate(user_prompt, api_token)
            logger.info("Respuesta recibida de Gemini.")

        # ── Formatear el pie de respuesta (Referencias / Fuentes web) ────────
        respuesta = formatear_pie_respuesta(respuesta)

        return {"response": respuesta}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
