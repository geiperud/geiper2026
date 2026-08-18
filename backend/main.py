import os
import re
import math
import time
import logging
import traceback
import unicodedata
import requests
import numpy as np
from datetime import datetime, timedelta
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
glm_token   = None
sh_client_id     = None
sh_client_secret = None
_sh_token_cache = {"token": None, "expira": 0.0}  # cache en memoria del access_token OAuth de Sentinel Hub

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


# ── LLM GLM via Zhipu AI / BigModel (fallback, formato OpenAI-compatible) ────
GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4.5-flash"


def glm_generate(prompt, api_key, nombre_asistente="Asistente Académico"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": construir_system_prompt(nombre_asistente)},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }
    MAX_REINTENTOS = 2
    for intento in range(MAX_REINTENTOS):
        resp = requests.post(GLM_URL, headers=headers, json=payload, timeout=25)
        if resp.status_code == 429:
            if intento < MAX_REINTENTOS - 1:
                espera = 5 * (intento + 1)
                logger.warning(f"Rate limit GLM (429), esperando {espera}s antes de reintentar...")
                time.sleep(espera)
                continue
            else:
                logger.error("Rate limit GLM (429) agotado tras reintentos.")
                raise HTTPException(
                    status_code=429,
                    detail="El servicio de IA está temporalmente saturado. Por favor, espera unos segundos e intenta de nuevo."
                )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


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

    # ── Caso 1: la consulta nombra un documento directamente por autor ──────
    # (ej. "hablemos del trabajo de forero zapata", "resume el de valbuena
    # gaona"). Se resuelve por coincidencia textual + retrieval filtrado por
    # fuente, NO por similitud coseno global (ver detectar_documento_nombrado
    # para el porque). Se prueba con query_actual y, si no hay match ahi, con
    # consulta_efectiva (cubre el caso "resume ESE trabajo" referido al turno
    # anterior, donde el nombre del autor ya no esta en el mensaje actual).
    archivo_nombrado = detectar_documento_nombrado(query_actual, referencias_apa)
    if archivo_nombrado is None and consulta_efectiva != query_actual:
        archivo_nombrado = detectar_documento_nombrado(consulta_efectiva, referencias_apa)
    if archivo_nombrado:
        contexto = buscar_contexto_por_documento(
            vectorstore_local, archivo_nombrado, consulta_efectiva, referencias_apa
        )
        if contexto:
            # similitud sintetica alta: es una fuente identificada con
            # certeza por nombre, no una coincidencia semantica aproximada.
            # Ademas, al ser >= UMBRAL_SALTAR_WEB, evita una busqueda web
            # redundante para una pregunta que ya se resolvio con la fuente
            # exacta que el usuario pidio.
            return contexto, 1.0

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


def _extraer_apellidos_autores(cita_apa):
    """Extrae los apellidos de autor(es) de una cita APA (la parte antes del
    año entre parentesis), descartando las iniciales sueltas de nombre (ej.
    'S.', 'L. M.'). Soporta apellidos compuestos ('Forero Zapata',
    'Vargas Rodríguez') y multiples autores unidos por '&'."""
    parte_autores = cita_apa.split(" (")[0]
    autores = parte_autores.split("&")
    apellidos = []
    for autor in autores:
        apellido = autor.split(",")[0].strip()
        apellido = re.sub(r'^(y|and)\s+', '', apellido, flags=re.IGNORECASE)
        if apellido:
            apellidos.append(apellido)
    return apellidos


def detectar_documento_nombrado(query, referencias_apa):
    """
    Detecta si la consulta nombra directamente a un documento indexado por su
    autor (ej. "hablemos del trabajo de forero zapata", "resume el de
    valbuena gaona", "que dice el de melo cristancho"). Este tipo de
    preguntas NO son preguntas de contenido semántico -- son una referencia
    directa a una fuente por nombre -- así que compararlas contra los chunks
    via similitud coseno (ver UMBRAL_SIMILITUD_COSENO) falla sistemáticamente:
    "hablemos del trabajo de X" esta lejos, en el espacio de embeddings, de
    un fragmento que habla de segmentacion semantica o redes convolucionales,
    aunque el documento SI este indexado. Por eso se detecta aparte, con
    coincidencia textual de apellidos, y se resuelve con
    buscar_contexto_por_documento (bypass del umbral).

    Devuelve el nombre de archivo del mejor match (el apellido mas largo /
    especifico que aparezca en la consulta, para evitar que un apellido corto
    y comun capture el documento equivocado), o None si no se detecta ninguno.
    """
    query_norm = _normalizar_texto(query)
    mejor_archivo = None
    mejor_longitud = 0
    for archivo, cita in referencias_apa.items():
        for apellido in _extraer_apellidos_autores(cita):
            apellido_norm = _normalizar_texto(apellido)
            if not apellido_norm:
                continue
            patron = r'\b' + re.escape(apellido_norm) + r'\b'
            if re.search(patron, query_norm) and len(apellido_norm) > mejor_longitud:
                mejor_archivo = archivo
                mejor_longitud = len(apellido_norm)
    return mejor_archivo


def buscar_contexto_por_documento(vectorstore_local, archivo, consulta_para_ordenar, referencias_apa, k=5):
    """
    Trae fragmentos de UN documento ya identificado por nombre (autor/título),
    sin pasar por UMBRAL_SIMILITUD_COSENO: ese umbral esta calibrado para
    preguntas de contenido, y aqui la pregunta ya trajo la fuente por su
    cuenta. La similitud coseno se usa solo para ORDENAR los fragmentos
    propios del documento entre si (los mas relevantes primero), nunca para
    decidir si el documento entra o no.
    """
    if vectorstore_local is None:
        return ""
    try:
        archivo_norm = _normalizar_nombre_archivo(archivo)
        id_a_indice = {
            docstore_id: idx
            for idx, docstore_id in vectorstore_local.index_to_docstore_id.items()
        }

        candidatos = []
        for docstore_id, doc in vectorstore_local.docstore._dict.items():
            fuente = os.path.basename(doc.metadata.get("source", ""))
            if _normalizar_nombre_archivo(fuente) != archivo_norm:
                continue
            idx = id_a_indice.get(docstore_id)
            if idx is None:
                continue
            candidatos.append((doc, idx))

        if not candidatos:
            logger.warning(
                f"RAG: '{archivo}' se detecto por nombre pero no tiene "
                f"fragmentos en el docstore (¿desalineado con el indice?)."
            )
            return ""

        query_vec = np.array(vectorstore_local._embed_query(consulta_para_ordenar), dtype="float32")
        norma_query = np.linalg.norm(query_vec) or 1.0
        query_norm = query_vec / norma_query

        puntuados = []
        for doc, idx in candidatos:
            doc_vec = np.array(vectorstore_local.index.reconstruct(int(idx)), dtype="float32")
            norma_doc = np.linalg.norm(doc_vec) or 1.0
            doc_norm = doc_vec / norma_doc
            similitud = float(np.dot(query_norm, doc_norm))
            puntuados.append((doc, similitud))

        puntuados.sort(key=lambda par: -par[1])
        seleccionados = puntuados[:k]

        apa = obtener_referencia_apa(referencias_apa, archivo)
        bloques = [f"[Referencia APA: {apa}]\n{doc.page_content[:500]}" for doc, _ in seleccionados]
        logger.info(
            f"RAG: documento nombrado detectado -> {archivo}, "
            f"{len(seleccionados)}/{len(candidatos)} fragmentos traidos directamente (bypass umbral)."
        )
        return "\n\n---\n\n".join(bloques)
    except Exception as e:
        logger.warning(f"No se pudo recuperar contexto por documento nombrado '{archivo}': {e}")
        return ""


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
    global vectorstore, vectorstore_investigacion, api_token, groq_token, glm_token

    groq_token  = os.environ.get("GROQ_API_KEY", "")
    if groq_token:
        logger.info("Groq API Key encontrada (modelo primario).")
    else:
        logger.warning("No se encontro GROQ_API_KEY.")

    glm_token = os.environ.get("GLM_API_KEY", "").strip()
    if glm_token:
        logger.info("GLM API Key encontrada (modelo fallback).")
    else:
        logger.warning("No se encontro GLM_API_KEY.")


    api_token = os.environ.get("GOOGLE_API_KEY", "").strip()
    if api_token:
        logger.info(
            f"Google API Key encontrada (modelo fallback + embeddings RAG). "
            f"len={len(api_token)} inicio='{api_token[:6]}' fin='{api_token[-4:]}'"
        )
    else:
        logger.warning("No se encontro GOOGLE_API_KEY.")

    global sh_client_id, sh_client_secret
    sh_client_id = os.environ.get("SENTINELHUB_CLIENT_ID", "").strip()
    sh_client_secret = os.environ.get("SENTINELHUB_CLIENT_SECRET", "").strip()
    if sh_client_id and sh_client_secret:
        logger.info("Credenciales de Sentinel Hub (Copernicus Data Space Ecosystem) encontradas.")
    else:
        logger.warning("No se encontraron SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET. "
                        "La herramienta de detección de cambios SAR quedará deshabilitada.")

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
    return {"status": "ok", "cloud_ready": bool(groq_token or glm_token)}

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
    if not groq_token and not glm_token:
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
                    # respaldo de GLM de abajo. Ahora CUALQUIER falla de
                    # Groq (429, error de red, lo que sea) cae al respaldo.
                    logger.warning(f"Groq fallo en saludo: {e}")
            if glm_token:
                respuesta = glm_generate(saludo_prompt, glm_token, nombre_asistente)
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

        # ── Generar respuesta (Groq primero, GLM fallback) ─────────────────
        respuesta = None
        if groq_token:
            try:
                logger.info(f"Enviando a Groq (Llama 3.3 70B) (modo: {chat_request.mode})")
                respuesta = groq_generate(user_prompt, groq_token, nombre_asistente)
                logger.info("Respuesta recibida de Groq.")
            except Exception as e:
                # Antes, un HTTPException (ej. 429 de Groq saturado) se
                # relanzaba directo al usuario sin darle oportunidad al
                # respaldo de GLM de abajo. Ahora CUALQUIER falla de
                # Groq (429, error de red, lo que sea) cae al respaldo.
                logger.warning(f"Groq falló, intentando con GLM: {e}")

        if respuesta is None:
            if not glm_token:
                raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
            logger.info(f"Enviando a GLM fallback (modo: {chat_request.mode})")
            respuesta = glm_generate(user_prompt, glm_token, nombre_asistente)
            logger.info("Respuesta recibida de GLM.")

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


# ═══════════════════════════════════════════════════════════════════════════
# ── Asistente del geovisor (function calling) ────────────────────────────
# Distinto del /chat (RAG sobre documentos): este endpoint no responde con
# texto fundamentado en PDFs, sino que traduce lenguaje natural en acciones
# ejecutables sobre el mapa (buscar un municipio, activar una herramienta de
# análisis espacial, cambiar el basemap, etc.), siguiendo el patrón de
# asistente geoespacial con function calling descrito en Dorobantu y Badea
# (2026a) para arquitecturas WebGIS: el LLM decide qué función invocar, el
# backend valida y estructura la llamada, y es el FRONTEND (no el backend)
# quien la ejecuta sobre el mapa real —el backend nunca toca datos
# geoespaciales, solo interpreta la intención del usuario.
# ═══════════════════════════════════════════════════════════════════════════

GEOVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_municipio",
            "description": (
                "Busca un municipio colombiano por nombre, lo resalta en el mapa "
                "y hace zoom hacia él. Úsalo cuando el usuario pida ver, ubicar, "
                "buscar o centrar el mapa en un municipio específico."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre": {
                        "type": "string",
                        "description": "Nombre del municipio, tal como lo escribió el usuario (ej. 'Sopó', 'Bogotá', 'Villa de Leyva')."
                    }
                },
                "required": ["nombre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "limpiar_seleccion_municipios",
            "description": "Quita del mapa todos los municipios actualmente resaltados/seleccionados.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "activar_herramienta_analisis",
            "description": (
                "Activa una herramienta de análisis espacial (Turf.js) sobre las tesis "
                "del semillero mostradas en el mapa. Úsala cuando el usuario pida calcular "
                "una distancia, un área, un buffer, un centroide, el vecino más cercano, "
                "si un punto está dentro de un polígono, una intersección, o celdas de Voronoi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "herramienta": {
                        "type": "string",
                        "enum": ["dist", "area", "buff", "ctrd", "near", "pip", "intx", "voro"],
                        "description": (
                            "dist=distancia entre dos puntos, area=área de cobertura (convex hull), "
                            "buff=buffer/zona de influencia, ctrd=centroide, near=vecino más cercano, "
                            "pip=punto en polígono, intx=intersección, voro=celdas de Voronoi."
                        )
                    }
                },
                "required": ["herramienta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "centrar_vista_colombia",
            "description": "Centra y ajusta el zoom del mapa para mostrar todo el territorio colombiano.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cambiar_basemapa",
            "description": "Cambia la capa base (basemap) del mapa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo": {
                        "type": "string",
                        "enum": ["osm", "sat", "relieve", "blank"],
                        "description": "osm=callejero OpenStreetMap, sat=satelital (EOX Sentinel-2), relieve=sombreado de terreno/alturas (AWS Terrain Tiles), blank=sin fondo."
                    }
                },
                "required": ["tipo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_coordenadas",
            "description": (
                "Obtiene las coordenadas geográficas (latitud/longitud) de un lugar en "
                "Colombia y centra el mapa ahí. Úsala para CUALQUIER lugar que el usuario "
                "pida ubicar o del que pida coordenadas: municipios, veredas, barrios, "
                "sitios de interés, direcciones aproximadas, ríos, o cualquier topónimo "
                "colombiano — no solo municipios."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lugar": {
                        "type": "string",
                        "description": "Nombre del lugar tal como lo escribió el usuario, con el departamento si lo mencionó (ej. 'Guatapé, Antioquia')."
                    }
                },
                "required": ["lugar"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filtrar_por_departamento",
            "description": (
                "Selecciona y resalta en el mapa TODOS los municipios que pertenecen a "
                "un departamento colombiano. Úsala cuando el usuario pida ver, resaltar, "
                "contar o seleccionar un departamento completo (no un municipio individual)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "departamento": {
                        "type": "string",
                        "description": "Nombre del departamento (ej. 'Cundinamarca', 'Boyacá', 'Valle del Cauca')."
                    }
                },
                "required": ["departamento"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "comparar_municipios",
            "description": (
                "Compara el área de dos municipios colombianos y resalta ambos en el "
                "mapa. Úsala cuando el usuario pida comparar, o preguntar cuál de dos "
                "municipios es más grande/pequeño."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "municipio1": {"type": "string", "description": "Nombre del primer municipio."},
                    "municipio2": {"type": "string", "description": "Nombre del segundo municipio."}
                },
                "required": ["municipio1", "municipio2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_proyectos_linea",
            "description": (
                "Lista los trabajos de tesis/proyectos del Semillero GEIPER que pertenecen "
                "a una línea de investigación (Percepción Remota, SIG, Geomática, etc.). "
                "Úsala cuando el usuario pregunte qué proyectos hay, cuántos, o pida "
                "ejemplos de una línea de investigación del semillero."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "linea": {
                        "type": "string",
                        "description": "Nombre o palabra clave de la línea de investigación mencionada por el usuario."
                    }
                },
                "required": ["linea"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perfil_elevacion_entre_puntos",
            "description": (
                "Calcula un perfil de elevación APROXIMADO en línea recta entre dos "
                "lugares (NO sigue carreteras reales, es la línea recta entre ambos "
                "puntos): ganancia y pérdida de altura acumuladas, pendiente promedio, "
                "y el punto más alto/más bajo del trayecto. Úsala cuando el usuario "
                "pregunte por el desnivel, el perfil de elevación, o qué tan empinado "
                "es el camino ENTRE DOS lugares específicos. NO la uses para pedir la "
                "pendiente o inclinación en un solo punto exacto -- eso no se puede "
                "calcular de forma confiable con los datos disponibles (ver instrucciones "
                "del sistema)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lugar_origen":  {"type": "string", "description": "Punto de partida del trayecto."},
                    "lugar_destino": {"type": "string", "description": "Punto de llegada del trayecto."}
                },
                "required": ["lugar_origen", "lugar_destino"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rango_elevacion_zona",
            "description": (
                "Estima el RANGO de elevación (diferencia entre el punto más alto y "
                "más bajo, entre varios puntos muestreados) dentro del área aproximada "
                "de un lugar (municipio, vereda, región), como indicador aproximado de "
                "qué tan accidentado es el terreno. Es una ESTIMACIÓN basada en pocos "
                "puntos muestreados dentro del área -- no un análisis raster completo "
                "del relieve. Úsala cuando el usuario pregunte qué tan accidentado, "
                "montañoso o plano es el terreno de una zona en general, o cuál es el "
                "desnivel de una zona (no de un punto)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lugar": {"type": "string", "description": "Nombre del municipio, vereda o zona."}
                },
                "required": ["lugar"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_elevacion",
            "description": (
                "Obtiene la elevación (altitud sobre el nivel del mar) de un lugar en "
                "Colombia y centra el mapa ahí. Úsala cuando el usuario pregunte qué tan "
                "alto está, la altitud o la elevación de un lugar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lugar": {"type": "string", "description": "Nombre del lugar del que se pide la elevación."}
                },
                "required": ["lugar"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_lugares_cercanos",
            "description": (
                "Busca lugares reales de una categoría (hospitales, universidades, colegios, "
                "farmacias, estaciones de policía, ríos) cerca de un punto, usando datos "
                "abiertos de OpenStreetMap, y los muestra como marcadores en el mapa. Úsala "
                "cuando el usuario pida encontrar o mostrar lugares cercanos a algo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "categoria": {
                        "type": "string",
                        "enum": ["hospital", "universidad", "colegio", "farmacia", "policia", "rio"],
                        "description": "Tipo de lugar a buscar."
                    },
                    "lugar": {
                        "type": "string",
                        "description": "Lugar de referencia alrededor del cual buscar (ej. 'Sopó', 'centro de Bogotá')."
                    },
                    "radio_km": {
                        "type": "number",
                        "description": "Radio de búsqueda en kilómetros. Si el usuario no lo especifica, usa 3."
                    }
                },
                "required": ["categoria", "lugar"]
            }
        }
    },
]

GEOVISOR_SYSTEM_PROMPT = (
    "Eres el asistente del geovisor del Semillero GEIPER (UDFJC). Tu única función "
    "es traducir lo que pide el usuario en llamadas a las funciones disponibles para "
    "controlar el mapa: buscar municipios, obtener coordenadas o elevación de cualquier "
    "lugar, perfil de elevación o rango de elevación aproximado, filtrar por departamento, "
    "comparar municipios, buscar lugares cercanos (hospitales, universidades, ríos, etc.), "
    "listar proyectos del semillero por línea de investigación, activar herramientas de "
    "análisis espacial, cambiar el basemap, o centrar la vista. No respondes preguntas de "
    "investigación de fondo ni buscas en documentos del corpus académico —para eso existen "
    "los Asistentes Temático y de Investigación del geoportal, indícaselo al usuario si "
    "pregunta algo de ese tipo. Si el usuario pide algo que no corresponde a ninguna función "
    "disponible (ej. dibujar polígonos libremente, generar datos ficticios), dilo con claridad "
    "y sin inventar una acción ni datos que no tienes. Si la petición es ambigua, pide una "
    "aclaración breve antes de llamar a una función.\n\n"
    "REGLA ESTRICTA sobre elevación y relieve: NUNCA inventes ni estimes de memoria un valor "
    "de pendiente, inclinación, altitud o perfil de terreno -- todo dato de elevación que "
    "des debe venir SIEMPRE de una llamada a una función (obtener_elevacion, "
    "perfil_elevacion_entre_puntos, rango_elevacion_zona). No existe ninguna función para "
    "calcular la pendiente exacta en un solo punto ni para seguir una carretera real (el "
    "geovisor no tiene un modelo digital de elevación propio ni un servicio de ruteo "
    "integrado): si el usuario pide eso, dile con claridad que esa medición no está "
    "disponible con los datos del geovisor y por qué, en vez de aproximarla, adivinarla, o "
    "responder 'aproximadamente' con un número. Esto aplica incluso si el usuario insiste, "
    "pide 'solo un estimado', reformula la pregunta de otra forma, o dice que es solo para "
    "referencia -- la respuesta sigue siendo que no se puede calcular con precisión."
)


# ── Bloqueo duro (pre-LLM) para preguntas de pendiente/relieve que el ────────
# geovisor NO puede responder con datos reales: pendiente exacta en un solo
# punto (requeriria un DEM local de alta resolucion, no solo puntos de
# Open-Elevation) y perfiles que sigan una via real (requeriria una API de
# ruteo que el proyecto no integra). Se resuelven ANTES de tocar al LLM --
# igual que es_pregunta_de_listado -- porque un texto fijo es la unica forma
# de garantizar CERO alucinacion, sin importar como se reformule la pregunta
# o cuanto insista el usuario. El texto del propio prompt de sistema refuerza
# lo mismo por si una variante de la pregunta no cae en este patron.
_PATRON_PENDIENTE_O_RELIEVE = re.compile(
    r'\b(pendiente|inclinacion|declive|grado de inclinacion|angulo de la pendiente|'
    r'que tan (inclinado|empinado))\b',
    re.IGNORECASE
)
_PATRON_ENTRE_DOS_LUGARES = re.compile(
    r'\bentre\b.+\by\b|\bde\b.+\ba\b|\btrayecto\b|\bcamino de\b|\bruta de\b',
    re.IGNORECASE
)
_PATRON_RUTA_REAL = re.compile(
    r'\b(carretera real|via real|ruta real|camino real|siguiendo la (carretera|via|ruta)|'
    r'por la (carretera|via)(?! recta))\b',
    re.IGNORECASE
)


def es_pregunta_imposible_de_relieve(query):
    """Detecta preguntas de pendiente/relieve que el geovisor NO puede
    responder con garantías (pendiente puntual exacta, o un perfil que siga
    una via real) para bloquearlas ANTES de que lleguen al LLM. Las
    preguntas de pendiente/desnivel ENTRE DOS lugares SI se dejan pasar --
    esas las resuelve perfil_elevacion_entre_puntos con datos reales."""
    query_norm = _normalizar_texto(query)
    pide_ruta_real = bool(_PATRON_RUTA_REAL.search(query_norm))
    pide_pendiente = bool(_PATRON_PENDIENTE_O_RELIEVE.search(query_norm))
    es_entre_dos_lugares = bool(_PATRON_ENTRE_DOS_LUGARES.search(query_norm))
    return pide_ruta_real or (pide_pendiente and not es_entre_dos_lugares)


MENSAJE_RELIEVE_NO_DISPONIBLE = (
    "No puedo calcular eso con precisión: el geovisor no tiene un modelo digital de "
    "elevación (DEM) propio, solo puede consultar la altitud de puntos individuales "
    "(vía Open-Elevation, resolución ~30 m) y calcular un perfil aproximado en línea "
    "recta entre dos lugares -- no la pendiente exacta en un punto puntual, y no "
    "siguiendo una carretera real (no hay un servicio de ruteo integrado). Prefiero "
    "decírtelo claramente a darte un número aproximado que parezca exacto. Si te sirve, "
    "puedo darte la elevación de un lugar específico, o un perfil aproximado en línea "
    "recta entre dos lugares (ej. \"perfil de elevación entre Bogotá y Villavicencio\")."
)



# ── Geocodificación vía Nominatim (OpenStreetMap) ────────────────────────
# Servicio gratuito y abierto, coherente con la arquitectura de costo cero
# del proyecto. Se respeta su política de uso: un User-Agent identificable
# y una sola petición por consulta (sin llamadas en lote).
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {
    "User-Agent": "GEIPER-Geoportal/1.0 (UDFJC; contacto: laddiazb@udistrital.edu.co)"
}


def geocodificar_lugar(lugar):
    """Resuelve un topónimo colombiano a coordenadas usando Nominatim.
    Devuelve None si no encuentra nada o si el servicio falla — nunca
    inventa una coordenada aproximada."""
    try:
        params = {
            "q": lugar,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "co",
        }
        resp = requests.get(NOMINATIM_URL, params=params, headers=NOMINATIM_HEADERS, timeout=8)
        resp.raise_for_status()
        resultados = resp.json()
        if not resultados:
            return None
        r = resultados[0]
        return {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "nombre_encontrado": r.get("display_name", lugar)
        }
    except Exception as e:
        logger.warning(f"Geocodificación falló para '{lugar}': {e}")
        return None


# ── Elevación vía Open-Elevation (proyecto open source, sin API key) ─────
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


def obtener_elevacion_metros(lat, lon):
    """Consulta la elevación en metros sobre el nivel del mar para un punto.
    Devuelve None si el servicio falla."""
    try:
        params = {"locations": f"{lat},{lon}"}
        resp = requests.get(OPEN_ELEVATION_URL, params=params, timeout=10)
        resp.raise_for_status()
        resultados = resp.json().get("results", [])
        if not resultados:
            return None
        return resultados[0].get("elevation")
    except Exception as e:
        logger.warning(f"Consulta de elevación falló para ({lat},{lon}): {e}")
        return None


def obtener_elevaciones_lote(puntos):
    """Consulta la elevación de VARIOS puntos en una sola llamada a
    Open-Elevation (POST por lotes), en vez de una petición por punto.
    Devuelve una lista de elevaciones en el mismo orden que 'puntos', o
    None si el servicio falla o la respuesta no trae la misma cantidad de
    resultados que de puntos pedidos -- nunca se rellenan ni se interpolan
    valores inventados para los que falten."""
    try:
        body = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in puntos]}
        resp = requests.post(OPEN_ELEVATION_URL, json=body, timeout=20)
        resp.raise_for_status()
        resultados = resp.json().get("results", [])
        if len(resultados) != len(puntos):
            return None
        return [r.get("elevation") for r in resultados]
    except Exception as e:
        logger.warning(f"Consulta de elevación en lote falló para {len(puntos)} puntos: {e}")
        return None


def interpolar_puntos_linea(lat1, lon1, lat2, lon2, n=12):
    """Genera n puntos igualmente espaciados en LÍNEA RECTA entre dos
    coordenadas (interpolación lineal simple en lat/lon). Esto NO sigue
    carreteras reales -- seguir una vía real requeriría una API de ruteo
    (ej. OSRM), que este proyecto no integra. Es una aproximación
    geométrica, suficiente para un perfil de elevación referencial."""
    puntos = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        puntos.append((lat, lon))
    return puntos


def distancia_haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta (geodésica aproximada, fórmula de
    Haversine) entre dos coordenadas, en kilómetros."""
    radio_tierra_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radio_tierra_km * math.asin(math.sqrt(a))


def obtener_boundingbox_lugar(lugar):
    """Igual que geocodificar_lugar, pero además devuelve el bounding box
    que reporta Nominatim para el lugar (limites sur/norte/oeste/este).
    Util para muestrear una ZONA en vez de un solo punto, sin depender de
    un DEM o de la geometría exacta del polígono (que solo vive en el
    TopoJSON del frontend, no en el backend). Devuelve None si falla."""
    try:
        params = {
            "q": lugar,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "co",
        }
        resp = requests.get(NOMINATIM_URL, params=params, headers=NOMINATIM_HEADERS, timeout=8)
        resp.raise_for_status()
        resultados = resp.json()
        if not resultados:
            return None
        r = resultados[0]
        bbox = r.get("boundingbox")  # Nominatim: [sur, norte, oeste, este] como strings
        if not bbox or len(bbox) != 4:
            return None
        return {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "nombre_encontrado": r.get("display_name", lugar),
            "sur": float(bbox[0]), "norte": float(bbox[1]),
            "oeste": float(bbox[2]), "este": float(bbox[3]),
        }
    except Exception as e:
        logger.warning(f"Consulta de bounding box falló para '{lugar}': {e}")
        return None


def muestrear_grilla_bbox(sur, norte, oeste, este, filas=3, columnas=3):
    """Genera una grilla de puntos (filas x columnas) dentro de un bounding
    box, para muestrear la elevación en varios puntos de una zona. No es un
    análisis raster completo -- es una muestra dispersa, pensada solo para
    dar un ESTIMADO del rango de elevación (relieve más o menos accidentado),
    no un valor exacto punto a punto."""
    puntos = []
    for i in range(filas):
        t_lat = (i / (filas - 1)) if filas > 1 else 0.5
        lat = sur + (norte - sur) * t_lat
        for j in range(columnas):
            t_lon = (j / (columnas - 1)) if columnas > 1 else 0.5
            lon = oeste + (este - oeste) * t_lon
            puntos.append((lat, lon))
    return puntos


# ── Lugares cercanos vía Overpass API (datos abiertos de OpenStreetMap) ──
# Mismo tipo de fuente que usa download_osm_to_geojson en el paper de
# Dorobantu y Badea (2026a): datos vectoriales reales, gratuitos, sin key.
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

CATEGORIA_OSM = {
    "hospital":    ("amenity", "hospital"),
    "universidad": ("amenity", "university"),
    "colegio":     ("amenity", "school"),
    "farmacia":    ("amenity", "pharmacy"),
    "policia":     ("amenity", "police"),
    "rio":         ("waterway", "river"),
}


def buscar_pois_cercanos(categoria, lat, lon, radio_km=3):
    """Busca elementos de OpenStreetMap de una categoría dada dentro de un
    radio (en km) alrededor de un punto. Devuelve una lista de hasta 15
    lugares con nombre y coordenadas, o [] si no encuentra nada o el
    servicio falla."""
    tag = CATEGORIA_OSM.get(categoria)
    if not tag:
        return []
    clave, valor = tag
    radio_m = max(200, min(int((radio_km or 3) * 1000), 15000))  # tope de seguridad: 15 km

    query = f"""
    [out:json][timeout:20];
    (
      node["{clave}"="{valor}"](around:{radio_m},{lat},{lon});
      way["{clave}"="{valor}"](around:{radio_m},{lat},{lon});
      relation["{clave}"="{valor}"](around:{radio_m},{lat},{lon});
    );
    out center 15;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=20)
        resp.raise_for_status()
        elementos = resp.json().get("elements", [])
        lugares = []
        for el in elementos:
            if el.get("type") == "node":
                elat, elon = el.get("lat"), el.get("lon")
            else:
                centro = el.get("center") or {}
                elat, elon = centro.get("lat"), centro.get("lon")
            if elat is None or elon is None:
                continue
            nombre = (el.get("tags") or {}).get("name", f"{categoria} sin nombre registrado")
            lugares.append({"lat": elat, "lon": elon, "nombre": nombre})
        return lugares[:15]
    except Exception as e:
        logger.warning(f"Búsqueda Overpass falló para categoría '{categoria}': {e}")
        return []


class GeovisorChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    historial: List[Turno] = Field(default_factory=list, max_length=6)


def groq_generate_with_tools(mensajes, api_key):
    """Variante de groq_generate que expone las herramientas del geovisor al
    modelo. A diferencia de /chat (RAG puro), aquí la respuesta puede venir
    como texto libre O como una o más llamadas a función (tool_calls)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": mensajes,
        "tools": GEOVISOR_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 400,
        "temperature": 0.1,
        "stream": False
    }
    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="El servicio de IA está temporalmente saturado. Por favor, espera unos segundos e intenta de nuevo."
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


@app.post("/geovisor-chat")
@limiter.limit("15/minute")
def geovisor_chat(request: Request, chat_request: GeovisorChatRequest):
    if not groq_token:
        raise HTTPException(status_code=500, detail="Sin configuración de API.")

    if contiene_contenido_bloqueado(chat_request.query):
        return {"response": "Prefiero que mantengamos un tono respetuoso. ¿Qué necesitas ver en el mapa?", "actions": []}

    # Bloqueo duro: pendiente puntual exacta o ruta siguiendo via real. Se
    # resuelve ANTES del LLM -- ver es_pregunta_imposible_de_relieve -- para
    # que sea imposible que el modelo invente un numero, sin importar como
    # el usuario reformule o insista en la misma pregunta.
    if es_pregunta_imposible_de_relieve(chat_request.query):
        return {"response": MENSAJE_RELIEVE_NO_DISPONIBLE, "actions": []}

    mensajes = [{"role": "system", "content": GEOVISOR_SYSTEM_PROMPT}]
    for turno in chat_request.historial:
        mensajes.append({"role": turno.role, "content": turno.content})
    mensajes.append({"role": "user", "content": chat_request.query})

    try:
        mensaje_modelo = groq_generate_with_tools(mensajes, groq_token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en geovisor-chat: {e}")
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")

    tool_calls = mensaje_modelo.get("tool_calls") or []
    acciones = []
    texto_extra = ""
    for tc in tool_calls:
        try:
            import json as _json
            nombre_fn = tc["function"]["name"]
            args = _json.loads(tc["function"]["arguments"] or "{}")

            # perfil_elevacion_entre_puntos: geocodifica ambos extremos,
            # interpola puntos en línea recta entre ellos, consulta su
            # elevación en UNA sola llamada por lotes a Open-Elevation, y
            # calcula estadísticas básicas. Es una aproximación GEOMÉTRICA
            # (línea recta), no sigue vías reales -- eso requeriría una API
            # de ruteo que el proyecto no integra, y se lo decimos siempre
            # al usuario en el propio texto de respuesta.
            if nombre_fn == "perfil_elevacion_entre_puntos":
                origen = geocodificar_lugar(args.get("lugar_origen", ""))
                destino = geocodificar_lugar(args.get("lugar_destino", ""))
                if not origen or not destino:
                    faltante = args.get("lugar_origen") if not origen else args.get("lugar_destino")
                    texto_extra += f"No pude ubicar \"{faltante}\" para calcular el perfil de elevación. "
                    continue

                n_puntos = 12
                puntos = interpolar_puntos_linea(
                    origen["lat"], origen["lon"], destino["lat"], destino["lon"], n_puntos
                )
                elevaciones = obtener_elevaciones_lote(puntos)
                if elevaciones is None or any(e is None for e in elevaciones):
                    texto_extra += (
                        f"Encontré {args.get('lugar_origen')} y {args.get('lugar_destino')} pero "
                        f"no pude consultar la elevación del trayecto en este momento. "
                    )
                    continue

                distancia_km = distancia_haversine_km(
                    origen["lat"], origen["lon"], destino["lat"], destino["lon"]
                )
                ganancia = sum(max(0, elevaciones[i + 1] - elevaciones[i]) for i in range(len(elevaciones) - 1))
                perdida  = sum(max(0, elevaciones[i] - elevaciones[i + 1]) for i in range(len(elevaciones) - 1))
                idx_max, idx_min = elevaciones.index(max(elevaciones)), elevaciones.index(min(elevaciones))
                pendiente_promedio_pct = (
                    abs(elevaciones[-1] - elevaciones[0]) / (distancia_km * 1000) * 100
                    if distancia_km > 0 else 0.0
                )

                texto_extra += (
                    f"Perfil de elevación en línea recta ({distancia_km:.1f} km) entre "
                    f"{args.get('lugar_origen')} ({elevaciones[0]:.0f} msnm) y "
                    f"{args.get('lugar_destino')} ({elevaciones[-1]:.0f} msnm): ganancia "
                    f"acumulada {ganancia:.0f} m, pérdida acumulada {perdida:.0f} m, pendiente "
                    f"promedio {pendiente_promedio_pct:.1f}%. Punto más alto: "
                    f"{elevaciones[idx_max]:.0f} msnm. Punto más bajo: {elevaciones[idx_min]:.0f} "
                    f"msnm. (Este perfil sigue la línea recta entre los dos puntos, no la "
                    f"carretera real -- no se integra una API de ruteo en este geovisor). "
                )
                acciones.append({
                    "name": "mostrar_perfil_elevacion",
                    "args": {
                        "puntos": [
                            {"lat": lat, "lon": lon, "elevacion": elev}
                            for (lat, lon), elev in zip(puntos, elevaciones)
                        ]
                    }
                })
                continue

            # rango_elevacion_zona: geocodifica el lugar y usa el bounding
            # box que devuelve Nominatim para muestrear una grilla de
            # puntos dentro del área -- una ESTIMACIÓN aproximada de cuánto
            # varía la elevación en la zona, siempre presentada como tal.
            if nombre_fn == "rango_elevacion_zona":
                zona = obtener_boundingbox_lugar(args.get("lugar", ""))
                if not zona:
                    texto_extra += f"No pude ubicar el área de \"{args.get('lugar')}\" para estimar su relieve. "
                    continue

                puntos = muestrear_grilla_bbox(zona["sur"], zona["norte"], zona["oeste"], zona["este"])
                elevaciones = obtener_elevaciones_lote(puntos)
                if elevaciones is None or any(e is None for e in elevaciones):
                    texto_extra += f"Encontré {args.get('lugar')} pero no pude consultar la elevación de la zona en este momento. "
                    continue

                rango = max(elevaciones) - min(elevaciones)
                texto_extra += (
                    f"En {args.get('lugar')}, la elevación de los {len(elevaciones)} puntos "
                    f"muestreados dentro del área va de {min(elevaciones):.0f} a "
                    f"{max(elevaciones):.0f} msnm (rango de {rango:.0f} m). Esta es una "
                    f"ESTIMACIÓN aproximada basada en pocos puntos muestreados en el área, no "
                    f"un análisis completo del relieve. "
                )
                acciones.append({
                    "name": "centrar_coordenadas",
                    "args": {"lat": zona["lat"], "lon": zona["lon"], "lugar": args.get("lugar")}
                })
                continue

            # obtener_coordenadas se resuelve aquí mismo, en el backend, en
            # vez de dejarle la geocodificación al frontend: es una consulta
            # de datos, no una manipulación del DOM del mapa.
            if nombre_fn == "obtener_coordenadas":
                geo = geocodificar_lugar(args.get("lugar", ""))
                if geo:
                    texto_extra += (
                        f"{args.get('lugar')}: latitud {geo['lat']:.5f}, "
                        f"longitud {geo['lon']:.5f}. "
                    )
                    acciones.append({
                        "name": "centrar_coordenadas",
                        "args": {"lat": geo["lat"], "lon": geo["lon"], "lugar": args.get("lugar")}
                    })
                else:
                    texto_extra += f"No pude encontrar coordenadas para \"{args.get('lugar')}\". "
                continue

            # obtener_elevacion: primero geocodifica (Nominatim), luego
            # consulta la elevación de ese punto (Open-Elevation).
            if nombre_fn == "obtener_elevacion":
                geo = geocodificar_lugar(args.get("lugar", ""))
                if not geo:
                    texto_extra += f"No pude encontrar \"{args.get('lugar')}\" para consultar su elevación. "
                    continue
                elevacion = obtener_elevacion_metros(geo["lat"], geo["lon"])
                if elevacion is not None:
                    texto_extra += f"{args.get('lugar')} está a {elevacion:.0f} msnm. "
                    acciones.append({
                        "name": "centrar_coordenadas",
                        "args": {"lat": geo["lat"], "lon": geo["lon"], "lugar": args.get("lugar")}
                    })
                else:
                    texto_extra += f"Encontré {args.get('lugar')} pero no pude consultar su elevación en este momento. "
                continue

            # buscar_lugares_cercanos: geocodifica el punto de referencia y
            # luego consulta Overpass por la categoría pedida alrededor de él.
            if nombre_fn == "buscar_lugares_cercanos":
                geo = geocodificar_lugar(args.get("lugar", ""))
                if not geo:
                    texto_extra += f"No pude ubicar \"{args.get('lugar')}\" para buscar alrededor. "
                    continue
                lugares = buscar_pois_cercanos(
                    args.get("categoria", ""), geo["lat"], geo["lon"], args.get("radio_km", 3)
                )
                if lugares:
                    texto_extra += (
                        f"Encontré {len(lugares)} resultado(s) de tipo "
                        f"'{args.get('categoria')}' cerca de {args.get('lugar')}. "
                    )
                    acciones.append({
                        "name": "mostrar_pois",
                        "args": {"lugares": lugares, "lat": geo["lat"], "lon": geo["lon"]}
                    })
                else:
                    texto_extra += (
                        f"No encontré resultados de tipo '{args.get('categoria')}' "
                        f"cerca de {args.get('lugar')} en ese radio. "
                    )
                continue

            acciones.append({"name": nombre_fn, "args": args})
        except Exception:
            continue

    texto_respuesta = texto_extra or mensaje_modelo.get("content") or (
        "Hecho." if acciones else "No entendí bien qué querías hacer en el mapa, ¿puedes reformularlo?"
    )

    return {"response": texto_respuesta.strip(), "actions": acciones}


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN DE CAMBIOS SAR (Sentinel-1) — Copernicus Data Space Ecosystem
#
# Compara la retrodispersión de radar (banda VV) entre dos periodos de
# tiempo sobre una misma área, usando la Process API de Sentinel Hub
# (https://dataspace.copernicus.eu). A diferencia de Sentinel-2, el radar
# atraviesa nubes y funciona de noche, lo que lo hace especialmente útil
# para detectar cambios recientes (inundaciones, deforestación, obras)
# en zonas con cobertura de nubes persistente -- muy común en Colombia.
#
# Técnica: log-ratio de amplitud VV (10*log10(después/antes)), el método
# estándar en teledetección SAR para resaltar cambios reales por encima
# del ruido speckle inherente al radar. Se visualiza con una escala
# divergente: rojo = aumento de retrodispersión (p. ej. nueva
# construcción, superficie más rugosa), azul = disminución (p. ej.
# inundación, tala), gris = sin cambio significativo.
# ══════════════════════════════════════════════════════════════════

SH_TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
SH_CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"

SAR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [
      { datasource: "antes",   bands: ["VV"] },
      { datasource: "despues", bands: ["VV"] }
    ],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(samples) {
  let antes   = samples.antes[0].VV;
  let despues = samples.despues[0].VV;

  // Sentinel-1 GRD usa 0 como valor de "sin dato" fuera de la franja de
  // barrido del satélite (la franja de una pasada no siempre cubre el
  // rectángulo completo elegido). Sin este chequeo, dividir por un valor
  // casi cero dispara la razón al máximo y esa zona se pinta de rojo
  // sólido como si fuera un cambio enorme -- siendo en realidad ausencia
  // de dato, no una detección real. Se pinta transparente en su lugar,
  // dejando ver el mapa base debajo, para no confundir "sin dato" con
  // "cambio detectado".
  if (antes <= 0 || despues <= 0) {
    return [0, 0, 0, 0];
  }

  let ratioDb = 10 * Math.log10(despues / antes);

  // Rango tipico de cambio real en SAR: -6 dB a +6 dB. Se recorta (clamp)
  // para que el ruido speckle fuera de ese rango no sature el color.
  let t = Math.max(-6, Math.min(6, ratioDb));
  t = (t + 6) / 12; // normalizado 0..1

  if (t < 0.5) {
    // disminución de retrodispersión -> azul (posible inundación / tala)
    let f = t * 2;
    return [0.85 * f, 0.85 * f, 1, 1];
  } else {
    // aumento de retrodispersión -> rojo (posible construcción / superficie nueva)
    let f = (t - 0.5) * 2;
    return [1, 0.85 * (1 - f), 0.85 * (1 - f), 1];
  }
}
"""


def get_sentinelhub_token():
    """Obtiene (y cachea en memoria) el access_token OAuth de Sentinel Hub
    vía client_credentials. Los tokens de Sentinel Hub duran ~1 hora; se
    reutiliza mientras no esté vencido para no gastar cuota de más."""
    if not sh_client_id or not sh_client_secret:
        raise HTTPException(
            status_code=503,
            detail="La detección de cambios SAR no está configurada en el servidor todavía "
                   "(faltan credenciales de Copernicus Data Space Ecosystem)."
        )

    ahora = time.time()
    if _sh_token_cache["token"] and ahora < _sh_token_cache["expira"]:
        return _sh_token_cache["token"]

    resp = requests.post(
        SH_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": sh_client_id,
            "client_secret": sh_client_secret,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _sh_token_cache["token"]  = data["access_token"]
    # se resta un margen de 60s para renovar antes de que expire de verdad
    _sh_token_cache["expira"] = ahora + data.get("expires_in", 3600) - 60
    return _sh_token_cache["token"]


def _ventana_de_dias(fecha_iso: str, margen_dias: int = 4):
    """Convierte una fecha puntual (YYYY-MM-DD) en una ventana [inicio, fin]
    de +/- margen_dias, para aumentar la probabilidad de encontrar una
    escena Sentinel-1 real dentro del rango (revisita ~6 días)."""
    base = datetime.strptime(fecha_iso, "%Y-%m-%d")
    inicio = (base - timedelta(days=margen_dias)).strftime("%Y-%m-%dT00:00:00Z")
    fin    = (base + timedelta(days=margen_dias)).strftime("%Y-%m-%dT23:59:59Z")
    return inicio, fin


class SarCambiosRequest(BaseModel):
    bbox: List[float] = Field(min_length=4, max_length=4)  # [minLon, minLat, maxLon, maxLat] EPSG:4326
    fecha_inicio: str  # YYYY-MM-DD ("antes")
    fecha_fin: str     # YYYY-MM-DD ("después")


class SarDisponibilidadRequest(BaseModel):
    bbox: List[float] = Field(min_length=4, max_length=4)  # [minLon, minLat, maxLon, maxLat] EPSG:4326


@app.post("/sar-disponibilidad")
@limiter.limit("15/minute")
def sar_disponibilidad(request: Request, body: SarDisponibilidadRequest):
    """Consulta el catálogo STAC de Sentinel Hub (Copernicus Data Space
    Ecosystem) para averiguar qué escenas Sentinel-1 GRD de los últimos
    2 años cubren el área solicitada COMPLETA -- es decir, filtra de
    antemano las fechas que dejarían huecos sin dato en el análisis,
    en vez de dejar que el usuario elija una fecha a ciegas y descubra
    el hueco después de haber esperado el procesamiento completo."""
    if body.bbox[0] >= body.bbox[2] or body.bbox[1] >= body.bbox[3]:
        raise HTTPException(status_code=400, detail="bbox inválido.")

    try:
        token = get_sentinelhub_token()

        hoy = datetime.utcnow()
        desde = hoy - timedelta(days=730)  # últimos 2 años

        payload = {
            "bbox": body.bbox,
            "datetime": desde.strftime("%Y-%m-%dT00:00:00Z") + "/" + hoy.strftime("%Y-%m-%dT23:59:59Z"),
            "collections": ["sentinel-1-grd"],
            "limit": 100,
        }
        resp = requests.post(
            SH_CATALOG_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        fechas_completas = set()
        for item in data.get("features", []):
            item_bbox = item.get("bbox")
            if not item_bbox or len(item_bbox) < 4:
                continue
            # La escena debe cubrir el bbox solicitado COMPLETO (los 4
            # bordes), no solo tocarlo -- comparación de cajas
            # (aproximación conservadora: no usa la geometría exacta,
            # que suele ser un paralelogramo rotado siguiendo la órbita,
            # pero evita depender de una librería de geometría adicional
            # y es más que suficiente para áreas del tamaño de un
            # municipio).
            cubre_completo = (
                item_bbox[0] <= body.bbox[0] and item_bbox[1] <= body.bbox[1]
                and item_bbox[2] >= body.bbox[2] and item_bbox[3] >= body.bbox[3]
            )
            if cubre_completo:
                fecha_iso = (item.get("properties", {}) or {}).get("datetime", "")
                if fecha_iso:
                    fechas_completas.add(fecha_iso[:10])  # YYYY-MM-DD

        fechas_ordenadas = sorted(fechas_completas, reverse=True)[:30]
        return {"fechas_disponibles": fechas_ordenadas}

    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de red en sar-disponibilidad: {e}")
        raise HTTPException(status_code=502, detail="No se pudo contactar el catálogo de Copernicus Data Space Ecosystem.")
    except Exception as e:
        logger.error(f"Error en sar-disponibilidad: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error interno consultando el catálogo.")


@app.post("/sar-cambios")
@limiter.limit("10/minute")
def sar_cambios(request: Request, body: SarCambiosRequest):
    if body.bbox[0] >= body.bbox[2] or body.bbox[1] >= body.bbox[3]:
        raise HTTPException(status_code=400, detail="bbox inválido.")

    ancho_grados = body.bbox[2] - body.bbox[0]
    alto_grados  = body.bbox[3] - body.bbox[1]
    if ancho_grados > 3 or alto_grados > 3:
        raise HTTPException(
            status_code=400,
            detail="El área seleccionada es demasiado grande para este análisis. "
                   "Elige un municipio o un rectángulo más pequeño."
        )

    try:
        token = get_sentinelhub_token()

        # Margen angosto (±1 día): las fechas ya vienen confirmadas del
        # catálogo STAC (ver /sar-disponibilidad), así que no hace falta
        # un margen amplio -- uno amplio incluso podría hacer que el
        # Process API elija por error una escena vecina con peor
        # cobertura en vez de la exacta que ya se confirmó completa.
        antes_ini, antes_fin     = _ventana_de_dias(body.fecha_inicio, margen_dias=1)
        despues_ini, despues_fin = _ventana_de_dias(body.fecha_fin, margen_dias=1)

        payload = {
            "input": {
                "bounds": {
                    "bbox": body.bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [
                    {
                        "type": "sentinel-1-grd",
                        "id": "antes",
                        "dataFilter": {
                            "timeRange": {"from": antes_ini, "to": antes_fin},
                            "acquisitionMode": "IW",
                            "polarization": "DV"
                        }
                    },
                    {
                        "type": "sentinel-1-grd",
                        "id": "despues",
                        "dataFilter": {
                            "timeRange": {"from": despues_ini, "to": despues_fin},
                            "acquisitionMode": "IW",
                            "polarization": "DV"
                        }
                    }
                ]
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
            },
            "evalscript": SAR_EVALSCRIPT
        }

        resp = requests.post(
            SH_PROCESS_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=45,
        )

        if resp.status_code == 400:
            # Caso más común: no hubo ninguna escena Sentinel-1 en alguna
            # de las dos ventanas de tiempo sobre esa área.
            raise HTTPException(
                status_code=422,
                detail="No se encontraron imágenes de radar Sentinel-1 para alguna de las dos "
                       "fechas en esa área. Prueba con fechas distintas (la revisita del "
                       "satélite es de unos 6 días)."
            )
        resp.raise_for_status()

        import base64
        imagen_b64 = base64.b64encode(resp.content).decode("ascii")

        return {
            "imagen_base64": imagen_b64,
            "bbox": body.bbox,
            "fecha_antes": body.fecha_inicio,
            "fecha_despues": body.fecha_fin,
        }

    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de red en sar-cambios: {e}")
        raise HTTPException(status_code=502, detail="No se pudo contactar el servicio de Copernicus Data Space Ecosystem.")
    except Exception as e:
        logger.error(f"Error en sar-cambios: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error interno procesando la detección de cambios.")

