"""
GEIPER · Reto de las ponencias (Semana ICG 2026, jueves 26 nov).

Router de FastAPI que se monta en main.py con:
    from reto.trivia import router as reto_router
    app.include_router(reto_router)

Variables de entorno (Render → Environment):
    RETO_DATABASE_URL   cadena de conexión de Supabase (Connect → Transaction pooler, puerto 6543)
    RETO_ADMIN_KEY      clave del panel del moderador (cualquier frase larga)

Decisiones:
- El servidor decide todo: entrega la pregunta sin la respuesta correcta,
  mide el tiempo desde que la entregó y calcula los puntos. El navegador
  solo muestra.
- El ranking y el estado público se calculan como máximo una vez cada
  RANKING_TTL segundos y se sirven desde memoria a todos los celulares:
  así la base de datos no se satura aunque 200 personas consulten.
- Al cerrar el reto (botón del moderador o hora de cierre) el servidor
  rechaza cualquier respuesta nueva, aunque alguien tuviera una pregunta abierta.
- No se limita por IP: en la red de la universidad cientos de celulares
  salen por la misma IP pública.
"""
import csv
import hmac
import io
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:  # el resto del backend funciona aunque falte la dependencia
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover
    ConnectionPool = None

router = APIRouter(prefix="/reto", tags=["reto"])

TIEMPO_LIMITE_S = 25          # lo que ve el participante
GRACIA_S = 3                  # latencia de red tolerada en el servidor
PUNTOS_ACIERTO = 100
BONO_MAX = 50
RANKING_TTL = 10              # segundos de caché del ranking/estado
GANADORES_POR_PONENCIA = 3
TOP_PUBLICO = 10
ALIAS_RE = re.compile(r"^[A-Za-z0-9ÁÉÍÓÚáéíóúÑñ_.\- ]{3,24}$")
CORREO_RE = re.compile(r"^[a-z0-9._%+-]+@udistrital\.edu\.co$")

_pool = None
_pool_lock = threading.Lock()
_log = logging.getLogger("reto")


def _sin_clave(texto, url):
    """Quita la contraseña de cualquier mensaje antes de escribirlo en los logs."""
    m = re.match(r"^[a-z]+://[^:/@]+:([^@]*)@", url)
    if m and m.group(1):
        texto = texto.replace(m.group(1), "****")
    return texto


def _db():
    """Pool perezoso: se crea en la primera petición del reto, no al arrancar
    main.py, para no afectar a los asistentes si la base de datos no responde."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                url = (os.environ.get("RETO_DATABASE_URL") or "").strip()
                if not url or ConnectionPool is None:
                    raise HTTPException(503, "Las preguntas no están configuradas todavía.")
                # Prueba de conexión directa: si falla, deja en los logs la causa real
                # (contraseña, host, usuario…), que el pool por sí solo no muestra.
                try:
                    import psycopg
                    psycopg.connect(url, connect_timeout=8, prepare_threshold=None).close()
                except Exception as e:  # noqa: BLE001
                    _log.error("RETO: no se pudo conectar a Supabase: %s: %s", type(e).__name__, _sin_clave(str(e), url))
                    raise HTTPException(503, "No hay conexión con la base de datos de las preguntas.")
                # prepare_threshold=None: necesario con el pooler de Supabase (modo transacción)
                _pool = ConnectionPool(
                    url, min_size=1, max_size=5, open=True, timeout=10,
                    kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
                )
    return _pool


def _q(sql, params=None, one=False):
    with _db().connection() as conn:
        cur = conn.execute(sql, params or {})
        if cur.description is None:
            return None
        return cur.fetchone() if one else cur.fetchall()


# ── Caché en memoria (Render corre un solo proceso) ────────────────
_cache = {"t": 0.0, "data": None}
_cache_lock = threading.Lock()


def _invalidar_cache():
    _cache["t"] = 0.0


def _estado_global():
    """Ponencias, configuración y ranking completo. Máx. 1 lectura cada RANKING_TTL s."""
    ahora = time.monotonic()
    if _cache["data"] is not None and ahora - _cache["t"] < RANKING_TTL:
        return _cache["data"]
    with _cache_lock:
        if _cache["data"] is not None and time.monotonic() - _cache["t"] < RANKING_TTL:
            return _cache["data"]
        cfg = _q("select abierto, cierre_at, cerrado_at from reto_config where id = 1", one=True)
        pon = _q("""select id, hora, sala, titulo, ponentes, abierta,
                           case when abierta then 'abierta'
                                when abierta_at is not null then 'cerrada'
                                else 'proxima' end as fase
                    from reto_ponencias order by id""")
        # Ranking de cada ponencia: más puntos en sus 3 preguntas; desempata el menor tiempo
        # y, después, quien se registró primero.
        rp = _q("""select r.ponencia_id, p.id as participante_id, p.alias,
                          sum(r.puntos) as puntos, coalesce(sum(r.tiempo_ms), 0) as tiempo_ms
                   from reto_respuestas r join reto_participantes p on p.id = r.participante_id
                   group by r.ponencia_id, p.id, p.alias, p.registrado_at
                   having sum(r.puntos) > 0
                   order by r.ponencia_id, puntos desc, tiempo_ms asc, p.registrado_at asc""")
        por_ponencia = {}
        for f in rp:
            lst = por_ponencia.setdefault(f["ponencia_id"], [])
            lst.append({"participante_id": f["participante_id"], "alias": f["alias"],
                        "puntos": f["puntos"], "tiempo_ms": f["tiempo_ms"], "puesto": len(lst) + 1})
        rk = _q("""select participante_id, alias, puntaje, tiempo_ms, ponencias_completas
                   from reto_ranking
                   order by puntaje desc, tiempo_ms asc, registrado_at asc""")
        cerrado = _reto_cerrado(cfg)
        data = {
            "cerrado": cerrado,
            "cierre_at": cfg["cierre_at"].isoformat() if cfg and cfg["cierre_at"] else None,
            "ponencias": [dict(p) for p in pon],
            "por_ponencia": por_ponencia,
            "ranking": [dict(r, puesto=i + 1) for i, r in enumerate(rk)],
            "participantes": len(rk),
        }
        _cache.update(t=time.monotonic(), data=data)
        return data


def _reto_cerrado(cfg):
    if not cfg or not cfg["abierto"]:
        return True
    return bool(cfg["cierre_at"] and datetime.now(timezone.utc) >= cfg["cierre_at"])


def _participante(token):
    if not token or not re.fullmatch(r"[0-9a-fA-F-]{36}", token):
        raise HTTPException(401, "Sesión no válida. Regístrese de nuevo.")
    p = _q("select id, alias from reto_participantes where token = %(t)s", {"t": token}, one=True)
    if not p:
        raise HTTPException(401, "Sesión no válida. Regístrese de nuevo.")
    return p


def _mi_progreso(pid):
    filas = _q("""select ponencia_id, n, puntos, acierto, respondida_at, mostrada_at,
                         (respondida_at is null and mostrada_at < now() - make_interval(secs => %(lim)s)) as vencida
                  from reto_respuestas where participante_id = %(p)s""",
               {"p": pid, "lim": TIEMPO_LIMITE_S + GRACIA_S})
    prog = {}
    for f in filas:
        d = prog.setdefault(f["ponencia_id"], {"puntos": 0, "respondidas": 0})
        d["puntos"] += f["puntos"]
        if f["respondida_at"] is not None or f["vencida"]:
            d["respondidas"] += 1
    return prog


def _top_ponencia(data, pid, n=GANADORES_POR_PONENCIA):
    return [{"puesto": r["puesto"], "alias": r["alias"], "puntos": r["puntos"]}
            for r in data["por_ponencia"].get(pid, [])[:n]]


def _publico_ranking(data, top=TOP_PUBLICO):
    return [{"puesto": r["puesto"], "alias": r["alias"], "puntaje": r["puntaje"]}
            for r in data["ranking"][:top] if r["puntaje"] > 0]


# ════════════════════════════════════════════════════════════════
#  Participantes
# ════════════════════════════════════════════════════════════════
class Registro(BaseModel):
    alias: str = Field(..., min_length=3, max_length=24)
    correo: str = Field(..., max_length=120)
    acepto: bool


@router.post("/registro")
def registro(body: Registro):
    alias = body.alias.strip()
    correo = body.correo.strip().lower()
    if not body.acepto:
        raise HTTPException(400, "Debe autorizar el uso del correo para participar.")
    if not ALIAS_RE.fullmatch(alias):
        raise HTTPException(400, "El alias debe tener entre 3 y 24 letras, números o _ . -")
    if not CORREO_RE.fullmatch(correo):
        raise HTTPException(400, "Use su correo institucional @udistrital.edu.co.")
    previo = _q("select alias, token from reto_participantes where lower(correo) = %(c)s",
                {"c": correo}, one=True)
    if previo:
        # Recuperar la sesión desde otro dispositivo: mismo correo y mismo alias.
        if previo["alias"].lower() == alias.lower():
            return {"token": str(previo["token"]), "alias": previo["alias"], "recuperado": True}
        raise HTTPException(409, "Ese correo ya está registrado con otro alias.")
    if _q("select 1 from reto_participantes where lower(alias) = %(a)s", {"a": alias.lower()}, one=True):
        raise HTTPException(409, "Ese alias ya está en uso. Elija otro.")
    try:
        p = _q("""insert into reto_participantes (alias, correo) values (%(a)s, %(c)s)
                  returning alias, token""", {"a": alias, "c": correo}, one=True)
    except Exception:
        raise HTTPException(409, "No se pudo registrar: el alias o el correo ya existen.")
    _invalidar_cache()
    return {"token": str(p["token"]), "alias": p["alias"], "recuperado": False}


@router.get("/estado")
def estado(x_reto_token: str = Header(None)):
    """Lo que consulta cada celular o PC cada 15 s mientras la página está visible."""
    data = _estado_global()
    p = _participante(x_reto_token)
    prog = _mi_progreso(p["id"])
    yo = next((r for r in data["ranking"] if r["participante_id"] == p["id"]), None)
    ponencias = []
    podios = 0
    for x in data["ponencias"]:
        mp = prog.get(x["id"], {"puntos": 0, "respondidas": 0})
        if mp["respondidas"] >= 3:
            est = "hecha"
        elif x["abierta"] and not data["cerrado"]:
            est = "abierta"
        else:
            est = "bloqueada"
        lista = data["por_ponencia"].get(x["id"], [])
        mio = next((r for r in lista if r["participante_id"] == p["id"]), None)
        final = x["fase"] == "cerrada" or data["cerrado"]
        gano = bool(final and mio and mio["puesto"] <= GANADORES_POR_PONENCIA)
        podios += 1 if gano else 0
        ponencias.append({**{k: x[k] for k in ("id", "hora", "sala", "titulo", "ponentes", "fase")},
                          "estado": est, "puntos": mp["puntos"], "respondidas": mp["respondidas"],
                          "mi_puesto": mio["puesto"] if mio else None, "gano": gano, "final": final,
                          "top": _top_ponencia(data, x["id"])})
    return {
        "alias": p["alias"],
        "cerrado": data["cerrado"],
        "cierre_at": data["cierre_at"],
        "ponencias": ponencias,
        "puntaje": yo["puntaje"] if yo else 0,
        "puesto": yo["puesto"] if yo and yo["puntaje"] > 0 else None,
        "completas": yo["ponencias_completas"] if yo else 0,
        "podios": podios,
        "actualizado_cada_s": 15,
    }


@router.get("/pregunta")
def pregunta(ponencia: str = Query(..., pattern=r"^0[1-6]$"), x_reto_token: str = Header(None)):
    """Entrega la siguiente pregunta pendiente de la ponencia, sin la respuesta correcta.
    El reloj del servidor empieza aquí; pedirla de nuevo no lo reinicia."""
    p = _participante(x_reto_token)
    data = _estado_global()
    if data["cerrado"]:
        raise HTTPException(423, "Las preguntas están cerradas.")
    pon = _q("select abierta from reto_ponencias where id = %(id)s", {"id": ponencia}, one=True)
    if not pon or not pon["abierta"]:
        raise HTTPException(423, "Las preguntas de esta ponencia todavía no están abiertas.")
    lim = TIEMPO_LIMITE_S + GRACIA_S
    filas = _q("""select n, respondida_at, mostrada_at,
                         extract(epoch from now() - mostrada_at) as transcurrido
                  from reto_respuestas
                  where participante_id = %(p)s and ponencia_id = %(id)s order by n""",
               {"p": p["id"], "id": ponencia})
    # ¿Hay una pregunta entregada y aún en tiempo? Se devuelve la misma.
    n = None
    for f in filas:
        if f["respondida_at"] is None and f["transcurrido"] < lim:
            n = f["n"]
            break
    if n is None:
        usadas = {f["n"] for f in filas}
        libres = [k for k in (1, 2, 3) if k not in usadas]
        if not libres:
            return {"terminada": True}
        n = libres[0]
        _q("""insert into reto_respuestas (participante_id, ponencia_id, n)
              values (%(p)s, %(id)s, %(n)s) on conflict do nothing""",
           {"p": p["id"], "id": ponencia, "n": n})
    q = _q("""select r.n, extract(epoch from now() - r.mostrada_at) as transcurrido,
                     g.enunciado, g.opcion_a, g.opcion_b, g.opcion_c, g.opcion_d
              from reto_respuestas r
              join reto_preguntas g on g.ponencia_id = r.ponencia_id and g.n = r.n
              where r.participante_id = %(p)s and r.ponencia_id = %(id)s and r.n = %(n)s""",
           {"p": p["id"], "id": ponencia, "n": n}, one=True)
    if not q:
        raise HTTPException(503, "Las preguntas de esta ponencia aún no están cargadas.")
    restante = max(0.0, TIEMPO_LIMITE_S - float(q["transcurrido"]))
    return {
        "terminada": False,
        "ponencia": ponencia,
        "n": q["n"],
        "enunciado": q["enunciado"],
        "opciones": [{"id": k, "texto": q["opcion_" + k]} for k in "abcd"],
        "segundos_restantes": round(restante, 1),
        "limite_s": TIEMPO_LIMITE_S,
    }


class Respuesta(BaseModel):
    ponencia: str = Field(..., pattern=r"^0[1-6]$")
    n: int = Field(..., ge=1, le=3)
    opcion: str = Field(..., pattern=r"^[abcd]$")


@router.post("/responder")
def responder(body: Respuesta, x_reto_token: str = Header(None)):
    p = _participante(x_reto_token)
    # Comprobación directa en la base (no en caché) para que el cierre sea inmediato
    cfg = _q("select abierto, cierre_at from reto_config where id = 1", one=True)
    if _reto_cerrado(cfg):
        _invalidar_cache()
        raise HTTPException(423, "Las preguntas están cerradas: ya no se reciben respuestas.")
    # Todo en una sentencia: solo actualiza si la pregunta fue entregada, no se ha
    # respondido y no ha vencido. Así no hay doble respuesta ni carreras.
    r = _q("""
        with g as (select correcta from reto_preguntas where ponencia_id = %(id)s and n = %(n)s),
        u as (
          update reto_respuestas r set
              respondida_at = now(),
              opcion        = %(op)s,
              tiempo_ms     = (extract(epoch from now() - r.mostrada_at) * 1000)::int,
              acierto       = (%(op)s = (select correcta from g)),
              puntos        = case when %(op)s = (select correcta from g)
                                   then %(base)s + round(%(bono)s * greatest(0, %(lim)s - extract(epoch from now() - r.mostrada_at)) / %(lim)s)::int
                                   else 0 end
          where r.participante_id = %(p)s and r.ponencia_id = %(id)s and r.n = %(n)s
            and r.respondida_at is null
            and now() - r.mostrada_at <= make_interval(secs => %(lim_gracia)s)
          returning r.acierto, r.puntos, r.tiempo_ms
        )
        select u.*, (select correcta from g) as correcta from u""",
           {"p": p["id"], "id": body.ponencia, "n": body.n, "op": body.opcion,
            "base": PUNTOS_ACIERTO, "bono": BONO_MAX, "lim": TIEMPO_LIMITE_S,
            "lim_gracia": TIEMPO_LIMITE_S + GRACIA_S}, one=True)
    if not r:
        raise HTTPException(409, "Esta pregunta ya fue respondida o se agotó el tiempo.")
    _invalidar_cache()
    prog = _mi_progreso(p["id"]).get(body.ponencia, {"puntos": 0, "respondidas": 0})
    return {"acierto": r["acierto"], "puntos": r["puntos"], "tiempo_ms": r["tiempo_ms"],
            "correcta": r["correcta"], "ponencia_puntos": prog["puntos"],
            "ponencia_respondidas": prog["respondidas"]}


@router.get("/resumen")
def resumen(ponencia: str = Query(..., pattern=r"^0[1-6]$"), x_reto_token: str = Header(None)):
    """Detalle de una ponencia ya terminada (pantalla «Ponencia completada»)."""
    p = _participante(x_reto_token)
    filas = _q("""select r.n, r.acierto, r.puntos, r.respondida_at is not null as respondida,
                         g.enunciado
                  from reto_respuestas r
                  left join reto_preguntas g on g.ponencia_id = r.ponencia_id and g.n = r.n
                  where r.participante_id = %(p)s and r.ponencia_id = %(id)s order by r.n""",
               {"p": p["id"], "id": ponencia})
    return {"preguntas": [dict(f) for f in filas], "total": sum(f["puntos"] for f in filas)}


@router.get("/publico")
def publico():
    """Pantalla proyectada: programa y top 10. No requiere sesión."""
    data = _estado_global()
    return {
        "cerrado": data["cerrado"],
        "cierre_at": data["cierre_at"],
        "participantes": data["participantes"],
        "ponencias": [{**{k: x[k] for k in ("id", "hora", "titulo", "abierta", "fase")},
                       "top": _top_ponencia(data, x["id"])} for x in data["ponencias"]],
    }


# ════════════════════════════════════════════════════════════════
#  Moderador (requiere RETO_ADMIN_KEY)
# ════════════════════════════════════════════════════════════════
def _admin(clave):
    esperada = os.environ.get("RETO_ADMIN_KEY", "")
    if not esperada or not clave or not hmac.compare_digest(clave.encode(), esperada.encode()):
        time.sleep(1)  # frena intentos de adivinar la clave
        raise HTTPException(403, "Clave de moderador incorrecta.")


class Apertura(BaseModel):
    ponencia: str = Field(..., pattern=r"^0[1-6]$")
    abierta: bool


@router.post("/admin/ponencia")
def admin_ponencia(body: Apertura, x_reto_admin: str = Header(None)):
    _admin(x_reto_admin)
    _q("""update reto_ponencias set abierta = %(a)s,
              abierta_at = case when %(a)s then coalesce(abierta_at, now()) else abierta_at end
          where id = %(id)s""", {"a": body.abierta, "id": body.ponencia})
    _invalidar_cache()
    return admin_estado(x_reto_admin)


@router.get("/admin/estado")
def admin_estado(x_reto_admin: str = Header(None)):
    _admin(x_reto_admin)
    _invalidar_cache()
    data = _estado_global()
    resp = _q("select count(*) as n from reto_respuestas where respondida_at is not null", one=True)
    preg = _q("select ponencia_id, count(*) as n from reto_preguntas group by ponencia_id")
    cargadas = {f["ponencia_id"]: f["n"] for f in preg}
    return {
        "cerrado": data["cerrado"],
        "cierre_at": data["cierre_at"],
        "participantes": data["participantes"],
        "respuestas": resp["n"],
        "ponencias": [{**x, "preguntas_cargadas": cargadas.get(x["id"], 0),
                       "top": _top_ponencia(data, x["id"])} for x in data["ponencias"]],
    }


class Cierre(BaseModel):
    cerrar: bool = True


@router.post("/admin/cerrar")
def admin_cerrar(body: Cierre, x_reto_admin: str = Header(None)):
    """Cierra (o reabre, si hubo un error) el reto. Al cerrar, el servidor deja de
    aceptar respuestas de inmediato y el ranking queda congelado como final."""
    _admin(x_reto_admin)
    if body.cerrar:
        _q("update reto_config set abierto = false, cerrado_at = now() where id = 1")
    else:
        _q("update reto_config set abierto = true, cerrado_at = null where id = 1")
    _invalidar_cache()
    return admin_estado(x_reto_admin)


@router.get("/admin/csv")
def admin_csv(x_reto_admin: str = Header(None)):
    """Respaldo para exportar al cierre: ranking de cada ponencia con correos.
    Los 3 primeros de cada ponencia quedan marcados como ganadores."""
    _admin(x_reto_admin)
    filas = _q("""select r.ponencia_id, p.alias, p.correo,
                         sum(r.puntos) as puntos, coalesce(sum(r.tiempo_ms), 0) as tiempo_ms,
                         count(*) filter (where r.acierto) as aciertos
                  from reto_respuestas r join reto_participantes p on p.id = r.participante_id
                  group by r.ponencia_id, p.id, p.alias, p.correo, p.registrado_at
                  order by r.ponencia_id, puntos desc, tiempo_ms asc, p.registrado_at asc""")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ponencia", "puesto", "ganador", "alias", "correo", "puntos", "aciertos", "tiempo_ms"])
    puesto, actual = 0, None
    for f in filas:
        if f["ponencia_id"] != actual:
            actual, puesto = f["ponencia_id"], 0
        puesto += 1
        ganador = "sí" if puesto <= GANADORES_POR_PONENCIA and f["puntos"] > 0 else ""
        w.writerow([f["ponencia_id"], puesto, ganador, f["alias"], f["correo"], f["puntos"],
                    f["aciertos"], f["tiempo_ms"]])
    return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=reto_ponencias_ganadores.csv"})
