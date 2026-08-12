import os
import re
import time
import logging
import traceback
import unicodedata
import requests
import numpy as np
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
GEMINI_MODEL = "gemini-2.5-flash"
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
    "estas áreas); metodología de la investigación, resoluciones y acuerdos de "
    "grado actualizados de la Facultad de Ingeniería, normativa y procedimientos "
    "de la ODI (Oficina de Investigaciones) de la Universidad Distrital Francisco "
    "José de Caldas incluyendo convocatorias de financiación para proyectos o "
    "estancias de investigación, formatos y trámites de trabajos de grado, "
    "estructura organizativa de la universidad, y apoyo en la estructuración de "
    "artículos académicos, ensayos y escritura académica en general; e "
    "información sobre el propio semillero GEIPER como organización (integrantes, "
    "líneas de investigación, eventos, repositorios) y sobre el sitio web y sus "
    "secciones en general"
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


# ── Detección de preguntas tipo "¿qué documentos tienes?" ────────────────────
# Estas preguntas NUNCA se pueden responder bien con busqueda semantica: ningun
# fragmento de un PDF dice "estos son todos los documentos indexados" -- esa es
# informacion sobre el sistema, no contenido de los documentos. Se responden
# aparte, enumerando directamente el indice, sin pasar por el LLM.
_PATRON_LISTAR_DOCS = re.compile(
    r'(que|cuales?)\s+(documentos?|pdfs?|archivos?)|'
    r'sobre que (documentos?|archivos?|informacion)|'
    r'listad?o? de documentos|'
    r'que (tienes|conoces) (indexado|cargado|disponible)',
    re.IGNORECASE
)


def es_pregunta_de_listado(query):
    return bool(_PATRON_LISTAR_DOCS.search(_normalizar_texto(query)))


def _normalizar_nombre_archivo(nombre):
    """Normaliza tildes/eñes a forma NFC. Un mismo archivo puede llegar con
    codificacion Unicode distinta (NFC vs NFD) segun por donde haya pasado
    (Windows, Colab, GitHub), y eso rompe comparaciones exactas de texto
    aunque el nombre se vea identico a simple vista."""
    return unicodedata.normalize("NFC", nombre or "")


def obtener_referencia_apa(referencias_dict, nombre_archivo):
    """Busca la cita APA de un archivo, robusto a diferencias de codificación
    Unicode en tildes/eñes. Si no la encuentra, devuelve el nombre tal cual."""
    nombre_norm = _normalizar_nombre_archivo(nombre_archivo)
    for clave, valor in referencias_dict.items():
        if _normalizar_nombre_archivo(clave) == nombre_norm:
            return valor
    return nombre_archivo


_PATRON_CORCHETES_FINALES = re.compile(r'\s*\[[^\]]*\]\.?\s*$')


def _quitar_corchetes_finales(cita):
    """Quita el bloque final entre corchetes de una cita APA (ej. '[Trabajo
    de grado, Universidad Distrital...]'), para que el listado de documentos
    se vea mas limpio. Solo se usa en el listado, no en las citas dentro de
    una respuesta normal, donde el corchete si aporta contexto."""
    return _PATRON_CORCHETES_FINALES.sub('.', cita).strip()


def listar_documentos_indexados(vectorstore_local, referencias_apa):
    """Enumera, directamente desde el índice, los documentos realmente indexados."""
    if vectorstore_local is None:
        return None
    try:
        fuentes = set()
        for doc in vectorstore_local.docstore._dict.values():
            fuente = os.path.basename(doc.metadata.get("source", ""))
            if fuente:
                fuentes.add(fuente)
        if not fuentes:
            return None
        lineas = [
            f"- {_quitar_corchetes_finales(obtener_referencia_apa(referencias_apa, fuente))}"
            for fuente in sorted(fuentes)
        ]
        return "\n".join(lineas)
    except Exception as e:
        logger.warning(f"No se pudo listar documentos indexados: {e}")
        return None

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
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_ARCHIVOS_PROMPT = {
    "Asistente Temático": "system_tematico.txt",
    "Asistente de Investigación": "system_investigacion.txt",
}
_PROMPT_CACHE = {}


def _cargar_plantilla_prompt(nombre_asistente):
    archivo = _ARCHIVOS_PROMPT.get(nombre_asistente, "system_tematico.txt")
    if archivo in _PROMPT_CACHE:
        return _PROMPT_CACHE[archivo]
    ruta = os.path.join(_PROMPTS_DIR, archivo)
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
        _PROMPT_CACHE[archivo] = contenido
        return contenido
    except Exception as e:
        logger.warning(f"No se pudo cargar la plantilla de prompt '{archivo}': {e}")
        return None


def construir_system_prompt(nombre_asistente):
    """
    System prompt compartido por Groq y Gemini (evita que el respaldo de
    Gemini pierda las reglas al fallar Groq). Se carga desde un archivo .txt
    editable en backend/prompts/ — uno por asistente — para que el rol, el
    tono y las reglas se puedan ajustar sin tocar el código Python.
    """
    plantilla = _cargar_plantilla_prompt(nombre_asistente)
    if plantilla:
        try:
            return plantilla.format(
                nombre_asistente=nombre_asistente,
                lineas_tematicas=LINEAS_TEMATICAS_DESC,
            )
        except Exception as e:
            logger.warning(f"Error formateando la plantilla de prompt: {e}")

    # Respaldo minimo por si el archivo .txt no existe o falla al cargar,
    # para que el servicio nunca se caiga por esto.
    return (
        f"Eres el {nombre_asistente} del semillero de investigación GEIPER. "
        f"Responde siempre en español, con rigor académico y tono cercano. "
        f"Solo respondes temas relacionados con: {LINEAS_TEMATICAS_DESC}."
    )


def groq_generate(prompt, api_key, nombre_asistente="Asistente Académico"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": construir_system_prompt(nombre_asistente)
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
def gemini_generate(prompt, api_key, nombre_asistente="Asistente Académico"):
    url = GEMINI_URL + f"?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": construir_system_prompt(nombre_asistente)}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.2
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
# Dominios genericos que no son fuentes serias para citar en un contexto
# academico (enciclopedias colaborativas, diccionarios, foros, redes
# sociales, sitios de descargas). Se excluyen de los resultados de busqueda.
_DOMINIOS_EXCLUIDOS_WEB = {
    "wikipedia.org", "wikimedia.org", "wiktionary.org", "wikihow.com",
    "rae.es", "dle.rae.es",
    "quora.com", "reddit.com", "pinterest.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "youtube.com", "youtu.be",
    "ccm.net", "commentcamarche.net", "forums.commentcamarche.net",
}


def _dominio_excluido(url):
    try:
        dominio = re.sub(r'^www\.', '', url.split("/")[2].lower())
        return any(dominio == d or dominio.endswith("." + d) for d in _DOMINIOS_EXCLUIDOS_WEB)
    except Exception:
        return False


def web_search(query, max_results=3):
    if not HAS_DDG:
        return []
    try:
        # Se piden mas resultados de los necesarios porque algunos se van a
        # filtrar despues; asi no se queda corto si varios eran de dominios
        # excluidos.
        with DDGS() as ddgs:
            resultados_brutos = list(ddgs.text(query, max_results=max_results * 3, region="es-es"))
        resultados_filtrados = [r for r in resultados_brutos if not _dominio_excluido(r.get("href", ""))]
        return resultados_filtrados[:max_results]
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
    re.compile(r'(?i)no se (han )?utiliz\w*'),
    re.compile(r'(?i)no se encontr[oó]'),
    re.compile(r'(?i)no hay referencias'),
    re.compile(r'(?i)no hay fuentes'),
    re.compile(r'(?i)no se citaron'),
    re.compile(r'(?i)^\s*no hay\s'),
    re.compile(r'(?i)^\s*ningun[oa]?\.?\s*$'),
]

_PATRON_LINK_MD = re.compile(r'\[([^\]]+)\]\((https?://[^\)\s]+)\)')


def _seccion_esta_vacia(contenido):
    contenido = contenido.strip()
    if not contenido:
        return True
    return any(p.search(contenido) for p in _PATRONES_SECCION_VACIA)


# ── Detección de citas inventadas en TODO el cuerpo de la respuesta ──────────
# No solo en 'Referencias:' -- un modelo puede alucinar un autor/año dentro
# del parrafo mismo ("...segun Hernandez et al. (2019)..."), lo cual no se
# puede limpiar de forma quirurgica con regex sin arriesgar romper el texto.
# Por eso, si se detecta AUNQUE SEA UNA cita no verificable en cualquier
# parte del texto, se descarta la respuesta completa por seguridad.
_PATRON_CITA_INLINE = re.compile(
    r'\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\'-]+)'
    r'(?:\s+et\s+al\.|(?:,\s*[A-ZÁÉÍÓÚÑ]\.){1,3}(?:,?\s*&\s*[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+(?:,\s*[A-ZÁÉÍÓÚÑ]\.){1,3})?)'
    r'\s*\((\d{4})\)'
)


def detectar_citas_no_verificadas(texto, referencias_validas):
    """
    Busca patrones tipo 'Apellido et al. (2020)' o 'Apellido, A. (2020)' en
    cualquier parte del texto, y verifica que el apellido y el año
    coincidan con alguna cita real conocida. Devuelve la lista de
    coincidencias que NO se pudieron verificar (posibles alucinaciones).
    """
    if not referencias_validas:
        return []
    referencias_texto = " ".join(referencias_validas).lower()
    sospechosas = []
    for m in _PATRON_CITA_INLINE.finditer(texto):
        cita_completa = m.group(0)
        apellido = m.group(1).lower()
        anio = m.group(2)
        if apellido not in referencias_texto or anio not in referencias_texto:
            sospechosas.append(cita_completa)
    return sospechosas


def formatear_pie_respuesta(texto, referencias_validas=None):
    """
    Post-procesa las secciones 'Referencias:' y 'Fuentes web consultadas:'
    generadas por el modelo:
      - Si el modelo indico que no se uso nada, la seccion se elimina por
        completo (no se muestra "No se utilizaron...").
      - Si 'Referencias:' no coincide con NINGUNA cita real conocida (las que
        estan en REFERENCIAS_APA / REFERENCIAS_APA_INVESTIGACION), se asume
        que el modelo la inventó (autor/año/título que no existen en el
        corpus) y se elimina esa sección por completo -- no se confía
        únicamente en que el modelo siga la instrucción de no inventar.
      - Si hay contenido real, se envuelve en <small><em> (fuente pequeña,
        cursiva) y los links en formato markdown se convierten a <a> reales.
    """
    def procesar(match):
        etiqueta = match.group(1)
        contenido = match.group(2).strip()

        if _seccion_esta_vacia(contenido):
            return ""

        if etiqueta.strip() == "Referencias:" and referencias_validas:
            coincide_con_cita_real = any(
                cita[:35].lower() in contenido.lower()
                for cita in referencias_validas
                if len(cita) >= 35
            )
            if not coincide_con_cita_real:
                logger.warning(
                    f"Cita en 'Referencias:' no coincide con ninguna conocida "
                    f"(posible alucinación), se elimina: {contenido[:200]}"
                )
                return ""

        contenido_html = _PATRON_LINK_MD.sub(
            r'<a href="\2" target="_blank" rel="noopener">\1</a>', contenido
        )
        return f'\n\n<small><em><strong>{etiqueta}</strong> {contenido_html}</em></small>'

    return _PATRON_SECCION_PIE.sub(procesar, texto).strip()


# Umbral de similitud coseno para aceptar un fragmento de documento como
# relevante. Rango real: -1 (opuesto) a 1 (idéntico). 0.70 es un punto de
# partida razonable para embeddings de recuperación (gemini-embedding-001
# con taskType RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY): en la práctica, un
# fragmento genuinamente relevante suele caer en 0.70+, uno tangencial en
# 0.50-0.65, y algo sin relación real por debajo de eso. Se deja como
# constante para poder subirlo o bajarlo sin tocar el resto del código.
# IMPORTANTE: hay que calibrarlo con datos reales — cada llamada deja un log
# "RAG: candidatos evaluados" con la similitud de TODOS los candidatos
# (pasen o no el umbral); revisa esos logs en Render tras un rato de uso
# real y ajusta este número si hace falta (ver explicación completa aparte).
UMBRAL_SIMILITUD_COSENO = 0.70


def _buscar_con_similitud_coseno(vectorstore_local, consulta, k=8):
    """
    Busca los k vecinos más cercanos en el índice FAISS y devuelve, para cada
    uno, la similitud coseno REAL entre la consulta y el fragmento — no la
    distancia euclidiana cruda que devuelve FAISS por defecto.

    Por qué hace falta esto: el índice se construye con FAISS.from_documents(),
    que usa por defecto un IndexFlatL2 (distancia euclidiana) y NO normaliza
    los vectores de gemini-embedding-001 antes de guardarlos. Eso vuelve el
    "score" original una distancia euclidiana cruda, sin un límite superior
    fijo e interpretable — de ahí que el umbral anterior (score < 1.6) dejara
    pasar prácticamente cualquier fragmento.

    Esta función reconstruye el vector original de cada resultado directamente
    desde el índice (index.reconstruct), normaliza ese vector y el de la
    consulta a longitud 1, y calcula la similitud coseno real entre ambos:
    coseno = (consulta · doc) / (‖consulta‖ · ‖doc‖), que siempre cae en el
    rango [-1, 1] y es directamente comparable entre búsquedas.
    """
    query_vec = np.array(vectorstore_local._embed_query(consulta), dtype="float32")
    norma_query = np.linalg.norm(query_vec) or 1.0
    query_norm = query_vec / norma_query

    _, indices = vectorstore_local.index.search(np.array([query_vec], dtype="float32"), k)

    resultados = []
    for idx in indices[0]:
        if idx == -1:
            continue
        doc_vec = np.array(vectorstore_local.index.reconstruct(int(idx)), dtype="float32")
        norma_doc = np.linalg.norm(doc_vec) or 1.0
        doc_norm = doc_vec / norma_doc
        similitud_coseno = float(np.dot(query_norm, doc_norm))

        docstore_id = vectorstore_local.index_to_docstore_id[int(idx)]
        doc = vectorstore_local.docstore.search(docstore_id)
        resultados.append((doc, similitud_coseno))

    return resultados


def buscar_contexto_documentos(vectorstore_local, referencias_apa, query_actual, consulta_efectiva):
    """
    Busca en el índice FAISS dado y arma el bloque de contexto con citas APA,
    filtrando por similitud coseno real (ver UMBRAL_SIMILITUD_COSENO).

    Prueba DOS versiones de la búsqueda: la pregunta tal cual la escribió el
    usuario, y la version combinada con el turno anterior (consulta_efectiva).
    Esto evita que preguntas claras y autosuficientes ("monografia") se vean
    perjudicadas por quedar mezcladas con contexto previo no relacionado,
    mientras que preguntas ambiguas con pronombres ("¿lo menciona algun
    trabajo?") siguen beneficiándose de la combinación.

    Devuelve una tupla (contexto_texto, mejor_similitud). mejor_similitud es
    la similitud coseno del fragmento más relevante encontrado (0.0 si no se
    encontró nada por encima del umbral) -- se usa más adelante para decidir
    si vale la pena además buscar en la web, o si el RAG ya respondió tan
    bien que buscar en la web sería redundante.
    """
    if vectorstore_local is None:
        return "", 0.0
    try:
        candidatos = {}
        consultas_a_probar = [query_actual]
        if consulta_efectiva != query_actual:
            consultas_a_probar.append(consulta_efectiva)

        for consulta in consultas_a_probar:
            docs_similitudes = _buscar_con_similitud_coseno(vectorstore_local, consulta, k=8)
            for doc, similitud in docs_similitudes:
                clave = f"{doc.metadata.get('source', '')}|{doc.metadata.get('page', '')}|{doc.page_content[:50]}"
                if clave not in candidatos or similitud > candidatos[clave][1]:
                    candidatos[clave] = (doc, similitud)

        # Log de TODOS los candidatos evaluados (pasen o no el umbral), para
        # poder calibrar UMBRAL_SIMILITUD_COSENO con datos reales de producción.
        if candidatos:
            resumen_todos = ", ".join(
                f"{os.path.basename(doc.metadata.get('source', '?'))}:{sim:.2f}"
                for doc, sim in sorted(candidatos.values(), key=lambda par: -par[1])
            )
            logger.info(f"RAG: candidatos evaluados (similitud coseno) -> {resumen_todos}")

        relevantes = sorted(
            [(doc, sim) for doc, sim in candidatos.values() if sim >= UMBRAL_SIMILITUD_COSENO],
            key=lambda par: -par[1]
        )[:5]

        if not relevantes:
            logger.info(
                f"RAG: ningún fragmento superó el umbral de similitud coseno "
                f"({UMBRAL_SIMILITUD_COSENO}) en ninguna de las dos consultas."
            )
            return "", 0.0

        bloques = []
        fuentes_log = []
        for doc, sim in relevantes:
            fuente = os.path.basename(doc.metadata.get("source", "documento"))
            apa = obtener_referencia_apa(referencias_apa, fuente)
            fuentes_log.append(f"{fuente} (similitud:{sim:.2f})")
            bloques.append(f"[Referencia APA: {apa}]\n{doc.page_content[:500]}")
        logger.info(f"RAG: {len(relevantes)} fragmentos relevantes: {', '.join(fuentes_log)}")
        mejor_similitud = relevantes[0][1]
        return "\n\n---\n\n".join(bloques), mejor_similitud
    except Exception as e:
        logger.warning(f"RAG falló: {e}")
        return "", 0.0


# ── Detección de preguntas sobre el semillero mismo o el sitio ──────────────
_PATRON_SOBRE_SEMILLERO = re.compile(
    r'\b(semillero|geiper|nide)\b|\blider(azgo)?\b|\bintegrantes?\b|\bmiembros?\b|'
    r'\bdirector(a)?\b|\bcoordinador(a)?\b|\bequipo\b|\bmision\b|\bvision\b|'
    r'\bhistoria del semillero\b|\bcuando (se )?fundo\b|\bminciencias\b|'
    r'\bque es geiper\b|\bsobre geiper\b|\blineas de investigacion (activas|oficiales|del semillero)\b',
    re.IGNORECASE
)


def es_pregunta_sobre_semillero(query):
    return bool(_PATRON_SOBRE_SEMILLERO.search(_normalizar_texto(query)))


# ── Detección de preguntas personales/emocionales (no académicas) ───────────
# Estas preguntas NUNCA deben forzar citas de tesis del semillero (buscar
# empleo, un estado de animo, una relacion personal no tiene nada que ver con
# cartografia de pozos petroleros o proteccion de bosques). Se detectan para:
#   1. Saltar el RAG por completo (ver uso en /chat).
#   2. Preferir la busqueda web como apoyo, si aplica.
#   3. Pedirle al modelo una respuesta breve y humana, sin el aparataje
#      academico habitual (ver construir_prompt_conversacional).
# Es una lista heuristica por patrones, no exhaustiva -- se puede ampliar con
# el tiempo segun lo que se vea en uso real.
_PATRON_PERSONAL_EMOCIONAL = re.compile(
    r'\bno encuentro trabajo\b|\bno consigo (trabajo|empleo)\b|\bbusco trabajo\b|'
    r'\bbusco empleo\b|\bperdi mi trabajo\b|\bme despidieron\b|'
    r'\bcomo consigo (trabajo|empleo|novi[oa]|pareja)\b|'
    # Se admite un intensificador opcional ("muy", "bastante", "tan") entre
    # el verbo y el adjetivo, para cubrir frases naturales como "estoy MUY
    # triste" o "me siento BASTANTE mal" -- sin esto, esas variantes tan
    # comunes no coincidian con el patron.
    r'\bestoy (muy |bastante |tan )?(triste|mal|deprimid[oa]|estresad[oa]|ansios[oa]|perdid[oa])\b|'
    r'\bme siento (muy |bastante |tan )?(mal|solo|sola|triste|deprimid[oa]|perdid[oa]|estresad[oa])\b|'
    r'\btengo ansiedad\b|\bno se que hacer con mi vida\b|\bconsejo de vida\b|'
    r'\bproblemas? (personales?|de pareja|familiares?)\b|'
    r'\btermine con mi (pareja|novi[oa])\b|\bme dejo mi (pareja|novi[oa])\b',
    re.IGNORECASE
)


def es_pregunta_personal_emocional(query):
    return bool(_PATRON_PERSONAL_EMOCIONAL.search(_normalizar_texto(query)))


# Palabras de alerta de crisis/autolesión: si aparecen, NUNCA se acorta la
# respuesta ni se le pide brevedad al modelo -- una respuesta de crisis
# necesita poder incluir lineas de ayuda completas, no quedar recortada por
# una instruccion de "se breve". Se sigue saltando el RAG (no tiene sentido
# citar tesis de geomatica aqui), pero sin el modo "respuesta_corta".
_PATRON_POSIBLE_CRISIS = re.compile(
    r'\bsuicid\w*\b|\bmatarme\b|\bquitarme la vida\b|\bno quiero vivir\b|'
    r'\bacabar con (mi vida|todo)\b|\bautolesion\w*\b|\bhacerme dano\b|'
    r'\bcortarme\b|\bya no aguanto\b|\bno vale la pena vivir\b',
    re.IGNORECASE
)


def es_posible_crisis(query):
    return bool(_PATRON_POSIBLE_CRISIS.search(_normalizar_texto(query)))


# Similitud coseno a partir de la cual el RAG ya respondió tan bien que
# buscar en la web sería redundante (ver logica de decision en /chat). Se
# deja mas alto que UMBRAL_SIMILITUD_COSENO (0.70) a proposito: 0.70 ya es
# "suficiente para citar", pero 0.80 es "tan bueno que no hace falta nada
# mas". Igual que el otro umbral, es un punto de partida a calibrar con los
# logs reales de produccion.
UMBRAL_SALTAR_WEB = 0.80


# ── Hechos verificados sobre el semillero y el sitio web ─────────────────────
# Extraidos directamente del HTML real del sitio (index.html, pages/lineas.html,
# pages/integrantes.html) -- no de busqueda web ni RAG sobre PDFs, que no
# cubren esta informacion de forma confiable.
# IMPORTANTE: actualizar manualmente este bloque cuando cambie el contenido
# real de esas paginas.
HECHOS_SEMILLERO = (
    "Nombre completo: GEIPER = Grupo Especializado en Investigación en Percepción "
    "Remota y Sistemas de Información Geográfica.\n"
    "Adscrito a: Universidad Distrital Francisco José de Caldas, Facultad de "
    "Ingeniería, programa de Ingeniería Catastral y Geodesia, Bogotá, Colombia.\n"
    "Pertenece al grupo de investigación NIDE, avalado por MinCiencias con "
    "calificación A. Activo desde 2007.\n"
    "Misión: espacio creado para estudiantes, docentes y egresados que buscan "
    "formarse como investigadores a través de la generación, concreción y "
    "aprovechamiento de sus ideas de investigación, fortaleciendo las líneas de "
    "trabajo del Grupo de Investigación e impactando positivamente a la comunidad.\n"
    "Visión (al 2026): ser un referente nacional e internacional en investigación, "
    "impulsando iniciativas alineadas con RedCOLSI, desarrollando habilidades "
    "interdisciplinarias, liderazgo y actitudes proactivas en sus miembros.\n"
    "Líneas de investigación oficiales:\n"
    "  1. Procesamiento digital y análisis de imágenes — procesamiento de imágenes "
    "pasivas y activas de percepción remota para monitoreo y análisis territorial.\n"
    "  2. Implementación de modelos de GeoIA — aplicaciones de inteligencia "
    "artificial para detección de cambios en la superficie terrestre.\n"
    "  3. Análisis espacial — análisis avanzado de imágenes satelitales y firmas "
    "espectrales.\n"
    "  4. Detección de cambios en objetos y fenómenos geográficos — modelado "
    "territorial, evaluación de riesgos y análisis geoespacial.\n"
    "  5. Infraestructuras de datos espaciales y estandarización — estandarización "
    "e interoperabilidad de información geoespacial (ráster, LiDAR, modelos de "
    "datos como LADM_COL) para su integración en plataformas territoriales.\n"
    "  6. Cartografía aplicada a la gestión de recursos y ordenamiento territorial "
    "— estructuración y adecuación de cartografía para la gestión de recursos "
    "naturales, catastro multipropósito y planificación del territorio.\n"
    "Contacto: geiper@udistrital.edu.co, Instagram/redes @semillerogeiper.\n"
    "La líder del semillero (mujer) es Laura Dayana Díaz Beltrán (estudiante, investigación en percepción remota).\n"
    "Profesores vinculados: José Luis Herrera Escorcia, Carlos Germán Ramírez Ramos, "
    "Maykol Camilo Delgado Correal, Paulo César Coronado Sánchez.\n"
    "Estudiantes integrantes (además de la líder): Argenis Alexandra Daza Roa, Mayra Ibeth Pérez "
    "Rodríguez, Roxxane Brigith Rozo Romero, Fabian Enrique Rodríguez Agatón, Laura Natalia Ramírez "
    "Aguilera, Haessier Joan Ortiz Moncada, Isabel Semanate Rivera, Camilo Arévalo Sánchez, Martín "
    "Porras Sierra, Marie Anne López Poveda, Juana Valentina León Upegui, Silvana Castillo Meneses, "
    "Yan Sebastián Muñoz Gamba, Anamaria Zamudio Arias, Johan Smit Pérez Muñoz, Luis Esteban Pinto "
    "Casique, Daniel Alejandro Afanador Bolívar, Martha Patricia Valbuena Gaona, Sebastian Forero "
    "Zapata, Brian Stiffenn Luna Bolívar."
)


def construir_consulta_web(chat_request, consulta_efectiva):
    """
    Decide qué texto se le manda a DuckDuckGo. A diferencia del RAG (donde
    combinar con el turno anterior ayuda a resolver pronombres contra el
    corpus de tesis), para un motor de búsqueda externo esa combinación es
    contraproducente: si el turno anterior trataba de un tema distinto,
    el resultado es una frase de dos temas pegados que no es buscable
    (ej. "que ha hecho te matare IA"), y DuckDuckGo devuelve resultados sin
    relación real con lo que se preguntó.

    Por eso, salvo en confirmaciones breves ("sí, dale", donde el mensaje
    actual por sí solo no dice nada y sí hace falta el contexto), se busca
    únicamente el mensaje actual, limpio.
    """
    if es_confirmacion_breve(chat_request.query) and chat_request.historial:
        return consulta_efectiva
    return chat_request.query


def buscar_contexto_web(consulta, priorizar_geiper=False):
    """
    Busca en la web (DuckDuckGo) y arma el bloque de contexto complementario.
    Si priorizar_geiper=True (preguntas sobre el semillero mismo: líder,
    integrantes, etc.), primero busca especificamente dentro del sitio de
    GEIPER (site:geiperud.github.io), ya que una busqueda abierta rara vez
    encuentra un sitio pequeño frente a resultados mas populares. Si esa
    busqueda no encuentra nada, cae de vuelta a la busqueda abierta normal.
    """
    resultados_web = []
    if priorizar_geiper:
        resultados_web = web_search(f"site:geiperud.github.io {consulta}", max_results=3)
        if resultados_web:
            logger.info("Búsqueda web priorizada al sitio de GEIPER (site:geiperud.github.io).")

    if not resultados_web:
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



def construir_prompt_conversacional(contexto_docs, contexto_web, transcripcion, query, respuesta_corta=False):
    """
    Arma el prompt final combinando documentos (fuente principal) + web
    (complemento), con las instrucciones de cita correspondientes. Se usa
    igual para el Asistente Temático y el Asistente de Investigación —
    cada uno le pasa su propio contexto ya buscado en su propio índice.

    respuesta_corta=True se usa para preguntas personales/emocionales (ver
    es_pregunta_personal_emocional): en ese caso contexto_docs normalmente
    viene vacío a propósito (se salta el RAG para no forzar citas de tesis
    que no vienen al caso), y se le pide al modelo una respuesta breve y
    humana en vez del formato académico habitual de varios párrafos con
    cierre en pregunta de seguimiento.
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
            "completo y no la menciones. Si entre los resultados web hay varias fuentes, prioriza las "
            "más serias y confiables (artículos académicos, revistas especializadas, libros, sitios "
            "institucionales o gubernamentales) sobre fuentes genéricas. Cuando sí la uses, cita el "
            "sitio con el formato [Título del sitio](URL), y nunca la uses para contradecir a los documentos. "
            "IMPORTANTE: si mencionas un curso, empresa, institución o programa específico por nombre "
            "propio, ese nombre debe aparecer literalmente en uno de estos resultados web, con su URL "
            "citada — nunca nombres una entidad externa concreta que no esté en esta lista de resultados."
        )
    else:
        instrucciones_citas.append(
            "No se obtuvieron resultados de búsqueda web reales para esta consulta (la búsqueda no "
            "encontró nada o falló). Por lo tanto, NO menciones cursos, empresas, instituciones, "
            "misiones ni programas externos específicos por nombre propio, y NO afirmes haber "
            "'encontrado información en internet' o en el sitio de alguna organización — eso sería "
            "presentar como verificado algo que no lo es. Responde solo con lo que sí esté en los "
            "fragmentos de documentos, o en términos generales sin nombres propios ni datos puntuales "
            "no verificables."
        )

    if respuesta_corta:
        instrucciones_extension = (
            "Esta es una pregunta personal o emocional, no una consulta académica del semillero. "
            "NO fuerces temas de GEIPER, geomática, percepción remota ni SIG si no vienen al caso, y "
            "NO cites tesis del semillero solo por tener algo que citar — es preferible responder sin "
            "ninguna referencia académica a forzar una que no aplica de verdad. Responde con calidez, "
            "cercanía y brevedad: máximo 2-3 párrafos cortos. No cierres con una pregunta de "
            "seguimiento tipo plantilla — solo hazlo si se siente genuinamente natural."
        )
    else:
        instrucciones_extension = (
            "Responde de forma conversacional y académica: párrafos fluidos, sin títulos con #, "
            "listas solo cuando sean estrictamente necesarias — EXCEPTO si el usuario pidió "
            "explícitamente una lista, listado o enumeración (ej. 'lista los integrantes', 'haz un "
            "listado de...'), en cuyo caso responde con viñetas o numeración real en markdown, no en "
            "prosa. Integra la información con análisis propio. "
            "Cierra tu explicación con una pregunta breve que invite a seguir conversando."
        )

    if partes_contexto:
        contexto_total = "\n\n===\n\n".join(partes_contexto)
        return (
            f"Tienes acceso a fragmentos de documentos del semillero GEIPER "
            f"y, de forma complementaria, a resultados de búsqueda web relacionados con la pregunta.\n\n"
            f"{' '.join(instrucciones_citas)} "
            f"Prioriza siempre los documentos como fuente principal, salvo que la pregunta sea "
            f"personal/emocional (ver instrucción de extensión abajo), en cuyo caso los documentos "
            f"académicos casi nunca aplican.\n\n"
            f"{instrucciones_extension} "
            f"DESPUÉS de tu respuesta, y solo al final de todo el mensaje, agrega dos apartados en "
            f"texto plano (sin negrita, sin asteriscos): 'Referencias:' seguido de las citas APA, y "
            f"'Fuentes web consultadas:' seguido de los enlaces en formato [Título](URL). Estos dos "
            f"apartados SIEMPRE deben ser lo último del mensaje. "
            f"IMPORTANTE: si no usaste documentos, OMITE por completo el apartado 'Referencias:' "
            f"— no lo escribas ni indiques que no se usó nada. Lo mismo aplica para "
            f"'Fuentes web consultadas:' si no usaste ninguna fuente web: simplemente no la incluyas.\n\n"
            f"{transcripcion}"
            f"{contexto_total}\n\n"
            f"PREGUNTA: {query}"
        )

    if respuesta_corta:
        return (
            f"No se encontró información web específica para esta consulta. "
            f"{instrucciones_extension} "
            f"No inventes datos externos, estadísticas ni cifras que no tengas certeza de que sean "
            f"reales — responde desde el acompañamiento y el sentido común, no desde datos que no "
            f"tienes.\n\n"
            f"{transcripcion}"
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

    api_token = os.environ.get("GOOGLE_API_KEY", "").strip()
    if api_token:
        logger.info(
            f"Google API Key encontrada (modelo fallback + embeddings RAG). "
            f"len={len(api_token)} inicio='{api_token[:6]}' fin='{api_token[-4:]}'"
        )
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


_PALABRAS_CONFIRMACION = (
    r'si+|s[ií]+|dale|adelante|claro|ok(ay)?|vale|va|de una|continua(mos)?|continuemos|'
    r'sigamos|hazlo|obvio|correcto|exacto|por favor|que'
)
_PATRON_CONFIRMACION_BREVE = re.compile(
    rf'^({_PALABRAS_CONFIRMACION})([\s,]+({_PALABRAS_CONFIRMACION}))*[\s,\.!¡¿?]*$',
    re.IGNORECASE
)


def es_confirmacion_breve(query):
    return bool(_PATRON_CONFIRMACION_BREVE.match(query.strip()))


def construir_consulta_efectiva(chat_request):
    """
    Combina turnos previos con la pregunta actual, para que las preguntas de
    seguimiento sigan encontrando contexto relevante al buscar en documentos
    y en la web.

    - Si la pregunta actual es una confirmación breve ('sí, adelante', 'dale',
      'continuemos'), el tema real casi siempre vive en la ÚLTIMA PREGUNTA QUE
      HIZO EL PROPIO ASISTENTE (la pregunta de cierre que el usuario está
      confirmando) — no en el último mensaje del usuario, que puede ser de
      un tema completamente distinto y ya resuelto. Se usa ese turno.
    - En cualquier otro caso, se combina con el último turno del usuario, para
      resolver pronombres ('lo', 'eso', 'ese tema').
    """
    query = chat_request.query

    if es_confirmacion_breve(query) and chat_request.historial:
        ultimo_turno_asistente = next(
            (t.content for t in reversed(chat_request.historial) if t.role == "assistant"),
            None
        )
        if ultimo_turno_asistente:
            return f"{ultimo_turno_asistente} {query}"

    ultimo_turno_usuario = next(
        (t.content for t in reversed(chat_request.historial) if t.role == "user"),
        None
    )
    if ultimo_turno_usuario:
        return f"{ultimo_turno_usuario} {query}"
    return query

@app.get("/status")
def status():
    return {"status": "ok", "cloud_ready": bool(groq_token or api_token)}

@app.get("/documentos")
def documentos(modo: str = "tematico"):
    """Devuelve la lista de documentos indexados para el asistente dado, para
    que el frontend pueda mostrar contexto antes de que el usuario pregunte."""
    if modo == "investigacion":
        vs, referencias = vectorstore_investigacion, REFERENCIAS_APA_INVESTIGACION
    else:
        vs, referencias = vectorstore, REFERENCIAS_APA

    listado = listar_documentos_indexados(vs, referencias)
    if not listado:
        return {"documentos": []}
    return {"documentos": listado.split("\n")}

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

        # ── Selecciona el índice, referencias y nombre según el asistente activo ─
        if chat_request.mode == "investigacion":
            vectorstore_activo   = vectorstore_investigacion
            referencias_activas  = REFERENCIAS_APA_INVESTIGACION
            nombre_asistente     = "Asistente de Investigación"
        else:
            vectorstore_activo   = vectorstore
            referencias_activas  = REFERENCIAS_APA
            nombre_asistente     = "Asistente Temático"

        # ── Detección de saludo: respuesta directa sin RAG ni web ────────────
        es_saludo = chat_request.query.strip().lower().rstrip("!?.") in SALUDOS
        if es_saludo:
            if chat_request.mode == "investigacion":
                saludo_prompt = (
                    f"El usuario te saludó. Responde con un saludo breve y amigable en español, "
                    f"y pregúntale sobre qué tema de metodología de investigación, normativa "
                    f"institucional, trámites de trabajos de grado o estructura de la universidad "
                    f"desea consultar. Sé conciso, no más de 3 líneas."
                )
            else:
                saludo_prompt = (
                    f"El usuario te saludó. Responde con un saludo breve y amigable en español, "
                    f"y pregúntale sobre cuál de los siguientes temas desea consultar:\n"
                    f"1. Percepción remota\n"
                    f"2. Geomática\n"
                    f"Sé conciso, no más de 3 líneas."
                )
            if groq_token:
                try:
                    respuesta = groq_generate(saludo_prompt, groq_token, nombre_asistente)
                    return {"response": respuesta}
                except Exception as e:
                    # Antes, un HTTPException (ej. 429 de Groq saturado) se
                    # relanzaba directo al usuario sin darle oportunidad al
                    # respaldo de Gemini de abajo. Ahora CUALQUIER falla de
                    # Groq (429, error de red, lo que sea) cae al respaldo.
                    logger.warning(f"Groq fallo en saludo: {e}")
            if api_token:
                respuesta = gemini_generate(saludo_prompt, api_token, nombre_asistente)
                return {"response": respuesta}

        # ── "¿Qué documentos tienes?": se responde directo, sin pasar por el LLM ─
        if es_pregunta_de_listado(chat_request.query):
            listado = listar_documentos_indexados(vectorstore_activo, referencias_activas)
            if listado:
                return {
                    "response": (
                        f"Estos son los documentos que tengo indexados actualmente como "
                        f"{nombre_asistente}:\n\n{listado}\n\n"
                        f"¿Sobre cuál de ellos quieres que profundicemos?"
                    )
                }

        pregunta_sobre_semillero = es_pregunta_sobre_semillero(chat_request.query)
        pregunta_personal = es_pregunta_personal_emocional(chat_request.query)
        posible_crisis = es_posible_crisis(chat_request.query)
        # Se pide respuesta breve para preguntas personales/emocionales, EXCEPTO
        # si hay señales de crisis/autolesión: ahí el modelo necesita poder dar
        # una respuesta completa (lineas de ayuda, etc.) sin que una
        # instrucción de "sé breve" se lo impida.
        respuesta_corta = pregunta_personal and not posible_crisis

        # ── RAG: se omite por completo en dos casos ───────────────────────────
        # 1. Preguntas sobre el semillero mismo (lider, mision, lineas
        #    oficiales, etc.): los hechos verificados de mas abajo ya
        #    responden esto de forma confiable, y buscar en los PDFs solo
        #    arriesga coincidencias falsas que terminarian citandose sin
        #    venir a cuento.
        # 2. Preguntas personales/emocionales (buscar empleo, estado de animo,
        #    relaciones, o alguna señal de crisis): no tiene sentido forzar
        #    una cita de una tesis de cartografia de pozos petroleros en una
        #    pregunta sobre por que alguien no consigue trabajo.
        if pregunta_sobre_semillero or pregunta_personal or posible_crisis:
            contexto_docs, mejor_similitud_docs = "", 0.0
        else:
            contexto_docs, mejor_similitud_docs = buscar_contexto_documentos(
                vectorstore_activo, referencias_activas, chat_request.query, consulta_efectiva
            )

        # ── Búsqueda web: ya no se dispara en cada mensaje sin condición ─────
        # Se busca en la web cuando:
        #   - la pregunta es sobre el semillero mismo (ya existía, se mantiene)
        #   - es una pregunta personal/emocional (nuevo: aqui es donde SI
        #     queremos que la web ayude a dar una respuesta util, en vez de
        #     forzar tesis que no aplican)
        #   - el RAG no encontró nada relevante (el corpus no cubre el tema:
        #     la web es exactamente el respaldo pensado para este caso)
        #   - el RAG encontró algo, pero no tan fuerte como para bastar por
        #     si solo (por debajo de UMBRAL_SALTAR_WEB)
        # y se omite solo cuando el RAG ya encontró un fragmento tan bueno
        # (similitud >= UMBRAL_SALTAR_WEB) que buscar en la web sería
        # redundante -- ahorra una llamada a DuckDuckGo y evita el riesgo de
        # que un resultado web de bajo valor distraiga al modelo.
        debe_buscar_web = (
            pregunta_sobre_semillero
            or pregunta_personal
            or not contexto_docs
            or mejor_similitud_docs < UMBRAL_SALTAR_WEB
        )

        if debe_buscar_web:
            consulta_web = construir_consulta_web(chat_request, consulta_efectiva)
            contexto_web = buscar_contexto_web(consulta_web, priorizar_geiper=pregunta_sobre_semillero)
        else:
            logger.info(
                f"Búsqueda web omitida: el RAG ya encontró un fragmento con "
                f"similitud {mejor_similitud_docs:.2f} (>= {UMBRAL_SALTAR_WEB})."
            )
            contexto_web = ""

        if pregunta_sobre_semillero:
            # Hechos verificados directamente de las paginas propias del sitio
            # (index, lineas, integrantes): mas confiable que RAG (los PDFs no
            # cubren esto) o busqueda web abierta (poco confiable para este
            # dato, ya que DuckDuckGo no tiene bien indexado el sitio actual).
            # Se formatea como fuente web para aprovechar el mismo mecanismo
            # de cita con link real.
            bloque_hechos = (
                "[Fuente web: Sitio oficial del semillero GEIPER — "
                "https://geiperud.github.io/]\n"
                f"{HECHOS_SEMILLERO}"
            )
            contexto_web = f"{bloque_hechos}\n\n---\n\n{contexto_web}" if contexto_web else bloque_hechos

        user_prompt = construir_prompt_conversacional(
            contexto_docs, contexto_web, transcripcion, chat_request.query, respuesta_corta=respuesta_corta
        )

        # ── Generar respuesta (Groq primero, Gemini fallback) ─────────────────
        respuesta = None
        if groq_token:
            try:
                logger.info(f"Enviando a Groq (Llama 3.3 70B) (modo: {chat_request.mode})")
                respuesta = groq_generate(user_prompt, groq_token, nombre_asistente)
                logger.info("Respuesta recibida de Groq.")
            except Exception as e:
                # Antes, un HTTPException (ej. 429 de Groq saturado) se
                # relanzaba directo al usuario sin darle oportunidad al
                # respaldo de Gemini de abajo. Ahora CUALQUIER falla de
                # Groq (429, error de red, lo que sea) cae al respaldo.
                logger.warning(f"Groq falló, intentando con Gemini: {e}")

        if respuesta is None:
            if not api_token:
                raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
            logger.info(f"Enviando a Gemini fallback (modo: {chat_request.mode})")
            respuesta = gemini_generate(user_prompt, api_token, nombre_asistente)
            logger.info("Respuesta recibida de Gemini.")

        # ── Verificación anti-alucinación: descarta toda la respuesta si hay ──
        # una cita no verificable en cualquier parte del texto, no solo en
        # 'Referencias:'. Prioridad absoluta: nunca mostrar una cita inventada.
        citas_sospechosas = detectar_citas_no_verificadas(respuesta, referencias_activas.values())
        if citas_sospechosas:
            logger.warning(
                f"Respuesta descartada por citas no verificables: {citas_sospechosas} "
                f"(modo: {chat_request.mode}, pregunta: {chat_request.query!r})"
            )
            listado_docs = listar_documentos_indexados(vectorstore_activo, referencias_activas)
            sugerencia_docs = (
                f"\n\nEstos son los documentos que sí tengo disponibles:\n\n{listado_docs}"
                if listado_docs else ""
            )
            return {
                "response": (
                    f"No tengo un documento específico sobre eso en mi corpus actual, "
                    f"y prefiero decírtelo con claridad en vez de arriesgarme a darte "
                    f"una referencia incorrecta.{sugerencia_docs}\n\n"
                    f"¿Quieres que busquemos algo relacionado en lo que sí tengo, o "
                    f"prefieres reformular la pregunta?"
                )
            }

        # ── Formatear el pie de respuesta (Referencias / Fuentes web) ────────
        respuesta = formatear_pie_respuesta(respuesta, referencias_validas=referencias_activas.values())

        return {"response": respuesta}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
