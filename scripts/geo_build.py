"""
geo_build.py — Optimización GEO del sitio GEIPER.

Genera, a partir del propio contenido del sitio:
  1. Un bloque JSON-LD (schema.org) en el <head> de cada página.
  2. Las tarjetas de eventos en HTML estático (index.html y pages/eventos.html),
     para que los rastreadores que no ejecutan JavaScript las lean.
  3. El archivo llms.txt en la raíz.

Es idempotente: reemplaza solo lo que está entre los marcadores
<!-- GEO:... START --> y <!-- GEO:... END -->. Se ejecuta en GitHub Actions
en cada push, así que basta con editar el HTML o data/eventos_feed.json.

Uso local:  python scripts/geo_build.py
Requiere:   beautifulsoup4
"""
import html
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://geiperud.github.io/geiper2026"
ORG_ID = f"{SITE}/#org"
WEBSITE_ID = f"{SITE}/#website"
MESES = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]

ORG_REF = {"@id": ORG_ID}


# ───────────────────────── utilidades ─────────────────────────
def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel, text):
    (ROOT / rel).write_text(text, encoding="utf-8")


def soup(rel):
    return BeautifulSoup(read(rel), "html.parser")


def clean(text):
    return " ".join((text or "").split())


def meta(s, name):
    tag = s.find("meta", attrs={"name": name})
    return clean(tag["content"]) if tag and tag.get("content") else ""


def title(s):
    return clean(s.title.get_text()) if s.title else ""


def h3_parts(h3):
    """Separa '<h3>Nombre<br><small>Lugar</small></h3>' en (nombre, lugar)."""
    small = h3.find("small")
    sub = clean(small.get_text()) if small else ""
    if small:
        small.extract()
    return clean(h3.get_text(" ")), sub


def replace_block(text, key, content, anchor):
    """Inserta o reemplaza el bloque marcado <!-- GEO:key START/END -->."""
    start, end = f"<!-- GEO:{key} START -->", f"<!-- GEO:{key} END -->"
    block = f"{start}\n{content}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    if anchor not in text:
        raise ValueError(f"No se encontró el ancla {anchor!r} para el bloque {key}")
    return text.replace(anchor, f"{block}\n{anchor}", 1)


def jsonld(graph):
    data = json.dumps({"@context": "https://schema.org", "@graph": graph},
                      ensure_ascii=False, indent=2)
    # Evita que un "</script>" dentro del texto cierre la etiqueta
    data = data.replace("</", "<\\/")
    return f'<script type="application/ld+json">\n{data}\n</script>'


def webpage(rel, s, page_type="WebPage", crumb=None):
    url = f"{SITE}/" if rel == "index.html" else f"{SITE}/{rel}"
    node = {
        "@type": page_type,
        "@id": f"{url}#webpage",
        "url": url,
        "name": title(s),
        "description": meta(s, "description"),
        "inLanguage": "es-CO",
        "isPartOf": {"@id": WEBSITE_ID},
        "publisher": ORG_REF,
    }
    graph = [node]
    if crumb:
        bc_id = f"{url}#breadcrumb"
        node["breadcrumb"] = {"@id": bc_id}
        graph.append({
            "@type": "BreadcrumbList",
            "@id": bc_id,
            "itemListElement": [
                {"@type": "ListItem", "position": 1, "name": "Inicio", "item": f"{SITE}/"},
                {"@type": "ListItem", "position": 2, "name": crumb, "item": url},
            ],
        })
    return node, graph


# ───────────────────────── eventos ─────────────────────────
def load_events():
    events = json.loads(read("data/eventos_feed.json"))
    return sorted(events, key=lambda e: e["date"], reverse=True)


def event_card(e):
    """Mismo marcado que construye js/eventos_rss.js."""
    y, m, d = (int(x) for x in e["date"].split("-"))
    esc = html.escape
    link = e["link"] if re.match(r"^https?://", e.get("link", "")) else "#"
    tags = "".join(f"<span>{esc(t)}</span>" for t in e.get("tags", []))
    return (
        '<div class="feed-row">'
        f'<div class="feed-date-block"><div class="feed-day">{d:02d}</div>'
        f'<div class="feed-month">{MESES[m - 1]} {y}</div></div>'
        f'<div><div class="feed-type">{esc(e["type"])}</div>'
        f'<h3 class="feed-title">{esc(e["title"])}</h3>'
        f'<p class="feed-desc">{esc(e["description"])}</p>'
        f'<div class="feed-tags">{tags}</div>'
        f'<a href="{esc(link)}" class="feed-link" target="_blank" rel="noopener noreferrer">Saber más →</a>'
        "</div></div>"
    )


def event_node(e):
    node = {
        "@type": "Event",
        "name": e["title"],
        "startDate": e["date"],
        "description": e["description"],
        "url": e["link"],
        "keywords": ", ".join(e.get("tags", [])),
        "eventStatus": "https://schema.org/EventScheduled",
    }
    # Campo opcional en eventos_feed.json: "location": "Buenos Aires, Argentina"
    if e.get("location"):
        node["location"] = {"@type": "Place", "name": e["location"],
                            "address": e["location"]}
    if e.get("endDate"):
        node["endDate"] = e["endDate"]
    return node


def render_static_events(text, limit=None):
    events = load_events()
    if limit:
        events = events[:limit]
    cards = "\n".join(event_card(e) for e in events) or \
        "<p>No hay eventos próximos en este momento.</p>"
    pattern = re.compile(r'(<div id="rssFeedContainer"[^>]*>)(.*?)(</div>\s*</div>\s*</section>)', re.S)
    body = f"\n<!-- GEO:eventos START -->\n{cards}\n<!-- GEO:eventos END -->\n    "
    new, n = pattern.subn(lambda m: m.group(1) + body + m.group(3), text, count=1)
    if n == 0:
        raise ValueError("No se encontró el contenedor rssFeedContainer")
    return new


# ───────────────────────── páginas ─────────────────────────
def page_index(s):
    page, graph = webpage("index.html", s)
    page["about"] = ORG_REF
    graph.append({
        "@type": "WebSite",
        "@id": WEBSITE_ID,
        "url": f"{SITE}/",
        "name": "Semillero GEIPER",
        "inLanguage": "es-CO",
        "publisher": ORG_REF,
    })
    return graph


def page_lineas(s):
    page, graph = webpage("pages/lineas.html", s, "CollectionPage", "Líneas de investigación")
    lineas = []
    for fila in s.select(".linea-fila"):
        h3, p = fila.find("h3"), fila.find("p")
        if h3:
            lineas.append({"@type": "DefinedTerm", "name": clean(h3.get_text()),
                           "description": clean(p.get_text()) if p else "",
                           "inDefinedTermSet": f"{SITE}/pages/lineas.html"})
    page["about"] = lineas
    return graph


def page_integrantes(s):
    page, graph = webpage("pages/integrantes.html", s, "ProfilePage", "Integrantes")
    people = []
    for card in s.select(".card"):
        h3 = card.find("h3")
        if not h3:
            continue
        badges = [clean(b.get_text()) for b in card.select(".badge")]
        p = card.find("p")
        a = card.find("a", href=True)
        person = {"@type": "Person", "name": clean(h3.get_text()), "memberOf": ORG_REF}
        if p:
            person["description"] = clean(p.get_text())
        if badges:
            person["jobTitle"] = " · ".join(badges)
        elif card.get("data-cat") is None:
            person["jobTitle"] = "Profesor vinculado"
        if a and re.match(r"^https?://", a["href"]):
            person["sameAs"] = a["href"]
        people.append(person)
    page["mainEntity"] = {
        "@type": "ItemList",
        "name": "Integrantes del Semillero GEIPER",
        "numberOfItems": len(people),
        "itemListElement": [{"@type": "ListItem", "position": i + 1, "item": per}
                            for i, per in enumerate(people)],
    }
    return graph, people


def page_ponencias(s):
    page, graph = webpage("pages/ponencias.html", s, "CollectionPage", "Ponencias")
    items = []
    for card in s.select("#ponencias .card"):
        h3, p = card.find("h3"), card.find("p")
        if not (h3 and p):
            continue
        evento, lugar = h3_parts(h3)
        work = {
            "@type": "CreativeWork",
            "genre": "Ponencia",
            "name": clean(p.get_text()).rstrip("."),
            "author": ORG_REF,
            "inLanguage": "es",
            "recordedAt": {"@type": "Event", "name": evento},
        }
        year = re.search(r"(19|20)\d{2}", evento)
        if year:
            work["dateCreated"] = year.group(0)
            work["recordedAt"]["startDate"] = year.group(0)
        if lugar:
            work["recordedAt"]["location"] = {"@type": "Place", "name": lugar, "address": lugar}
        items.append(work)
    page["hasPart"] = items
    return graph, items


def page_actividades(s):
    page, graph = webpage("pages/actividades.html", s, "CollectionPage", "Actividades")
    seen, parts = set(), []
    for h3 in s.find_all("h3"):
        name = clean(h3.get_text())
        p = h3.find_next_sibling("p")
        if name in seen:
            continue
        seen.add(name)
        parts.append({"@type": "Thing", "name": name,
                      "description": clean(p.get_text()) if p else ""})
    page["about"] = parts
    return graph


def page_repositorios(s):
    page, graph = webpage("pages/repositorios.html", s, "CollectionPage", "Repositorios")
    repos = []
    for card in s.select(".repo-card"):
        h3, a, p = card.find("h3"), card.find("a", href=True), card.find("p")
        if h3 and a:
            repos.append({"@type": "WebSite", "name": clean(h3.get_text(" ")),
                          "url": a["href"],
                          "description": clean(p.get_text()) if p else ""})
    page["mentions"] = repos
    return graph


def page_eventos(s):
    page, graph = webpage("pages/eventos.html", s, "CollectionPage", "Eventos")
    page["hasPart"] = [event_node(e) for e in load_events()]
    return graph


def page_app(rel, s, crumb, category, about):
    page, graph = webpage(rel, s, "WebPage", crumb)
    url = f"{SITE}/{rel}"
    app = {
        "@type": "WebApplication",
        "@id": f"{url}#app",
        "name": title(s).split(" – ")[0],
        "url": url,
        "description": meta(s, "description"),
        "applicationCategory": category,
        "operatingSystem": "Navegador web",
        "isAccessibleForFree": True,
        "inLanguage": "es-CO",
        "creator": ORG_REF,
        "about": about,
    }
    page["mainEntity"] = {"@id": app["@id"]}
    graph.append(app)
    return graph


# ───────────────────────── llms.txt ─────────────────────────
def build_llms(pages_meta, lineas_soup, people, ponencias):
    lines = [
        "# Semillero GEIPER",
        "",
        "> Semillero de investigación en percepción remota, sistemas de información "
        "geográfica (SIG), geomática e inteligencia artificial geoespacial (GeoAI) de la "
        "Universidad Distrital Francisco José de Caldas (Bogotá, Colombia), adscrito al "
        "grupo de investigación NIDE.",
        "",
        "GEIPER forma estudiantes y profesionales en procesamiento de imágenes satelitales "
        "y de radar, análisis espacial, catastro multipropósito y GeoAI. Publica sus "
        "ponencias, actividades y recursos abiertos en este sitio. Contacto: "
        "geiper@udistrital.edu.co.",
        "",
        "## Líneas de investigación",
        "",
    ]
    for fila in lineas_soup.select(".linea-fila"):
        h3, p = fila.find("h3"), fila.find("p")
        if h3:
            lines.append(f"- {clean(h3.get_text())}: {clean(p.get_text()) if p else ''}")
    lines += ["", "## Páginas principales", ""]
    for rel, name, desc in pages_meta:
        url = f"{SITE}/" if rel == "index.html" else f"{SITE}/{rel}"
        lines.append(f"- [{name}]({url}): {desc}")
    lines += ["", "## Ponencias recientes", ""]
    for w in ponencias:
        ev = w["recordedAt"]
        loc = ev.get("location", {}).get("name", "")
        lines.append(f"- {w['name']} — {ev['name']}" + (f" ({loc})" if loc else ""))
    events = load_events()
    if events:
        lines += ["", "## Agenda de eventos", ""]
        for e in events:
            lines.append(f"- {e['date']} · {e['title']} ({e['type']}): {e['link']}")
    lines += [
        "",
        "## Integrantes",
        "",
        f"El semillero tiene {len(people)} integrantes entre estudiantes, profesionales "
        f"y profesores vinculados. Lista completa: {SITE}/pages/integrantes.html",
        "",
        "## Opcional",
        "",
        f"- [Feed RSS de eventos]({SITE}/feed.xml)",
        f"- [Sitemap]({SITE}/sitemap.xml)",
        "",
    ]
    return "\n".join(lines)


# ───────────────────────── main ─────────────────────────
def main():
    S = {rel: soup(rel) for rel in [
        "index.html", "pages/lineas.html", "pages/integrantes.html",
        "pages/ponencias.html", "pages/actividades.html", "pages/repositorios.html",
        "pages/eventos.html", "pages/geovisor.html", "pages/chatbots.html",
    ]}
    ints_graph, people = page_integrantes(S["pages/integrantes.html"])
    pon_graph, ponencias = page_ponencias(S["pages/ponencias.html"])
    graphs = {
        "index.html": page_index(S["index.html"]),
        "pages/lineas.html": page_lineas(S["pages/lineas.html"]),
        "pages/integrantes.html": ints_graph,
        "pages/ponencias.html": pon_graph,
        "pages/actividades.html": page_actividades(S["pages/actividades.html"]),
        "pages/repositorios.html": page_repositorios(S["pages/repositorios.html"]),
        "pages/eventos.html": page_eventos(S["pages/eventos.html"]),
        "pages/geovisor.html": page_app(
            "pages/geovisor.html", S["pages/geovisor.html"], "Geovisor",
            "GISApplication", "Cartografía, límites municipales, POT de la Sabana de Bogotá y datos SAR"),
        "pages/chatbots.html": page_app(
            "pages/chatbots.html", S["pages/chatbots.html"], "Asistentes IA",
            "EducationalApplication", "Asistentes de IA sobre percepción remota, SIG e investigación"),
    }
    for rel, graph in graphs.items():
        text = read(rel)
        text = replace_block(text, "jsonld", jsonld(graph), "</head>")
        if rel == "index.html":
            text = render_static_events(text, limit=1)
        elif rel == "pages/eventos.html":
            text = render_static_events(text)
        write(rel, text)
        print(f"OK {rel}")

    pages_meta = [(rel, title(S[rel]).split(" – ")[0], meta(S[rel], "description"))
                  for rel in graphs]
    write("llms.txt", build_llms(pages_meta, S["pages/lineas.html"], people, ponencias))
    print("OK llms.txt")


if __name__ == "__main__":
    main()
