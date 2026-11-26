# Reto de las ponencias · Semana ICG 2026

Sección temporal del jueves 26 de noviembre de 2026 (Aduanilla de Paiba).

| Quién | Dirección |
|---|---|
| Participantes (QR) | `https://geiperud.github.io/geiper2026/pages/reto.html` |
| Pantalla del video beam | `…/pages/reto.html?modo=pantalla` |
| Moderador (con clave) | `…/pages/reto.html?modo=moderador` |

Archivos: `pages/reto.html`, `css/reto.css`, `js/reto.js` (página) y
`backend/reto/trivia.py` (API, montada en `main.py`), `backend/reto/schema.sql`
(base de datos en Supabase).

## Puesta en marcha (una sola vez)

1. **Supabase:** crear un proyecto gratuito con el correo del semillero.
   Elegir la región más cercana a la de Render (la misma, si es posible).
2. **Esquema:** SQL Editor → New query → pegar `schema.sql` → Run.
   Antes, revisar la hora de cierre (`cierre_at`, al inicio del archivo).
3. **Preguntas:** llenar `preguntas_plantilla.csv` (3 por ponencia; `correcta`
   es a, b, c o d) **en su computador**, guardarlo como `preguntas.csv` y subirlo en
   Table Editor → `reto_preguntas` → Insert → Import data from CSV.
   Ese archivo nunca se sube a GitHub (el repositorio es público; `.gitignore`
   ya lo bloquea).
4. **Render → Environment:**
   - `RETO_DATABASE_URL`: en Supabase, botón *Connect* → *Transaction pooler*
     (puerto 6543), con la contraseña de la base de datos.
   - `RETO_ADMIN_KEY`: una frase larga que solo conozca el moderador.
5. Subir los cambios a GitHub (Render redespliega solo).

## Antes del evento

- **Martes o miércoles:** prueba completa con 3 o 4 personas (registrarse,
  abrir una ponencia desde el moderador, responder, ver la pantalla). Luego, en
  Supabase: `truncate reto_respuestas, reto_participantes cascade;`
  y `update reto_ponencias set abierta = false;`
- **Jueves 8:40 a. m.:** abrir la página para despertar Render (tarda ~1 min si
  estaba dormido) y dejar la pantalla proyectada abierta.

## Durante y después

- El moderador abre cada ponencia en los últimos minutos de la charla.
- Al cerrar (botón o hora de cierre) el servidor rechaza toda respuesta nueva.
- Exportar el CSV desde el panel del moderador antes de cualquier limpieza.
- Consultas útiles (podio, sorteo, borrado de correos) al final de `schema.sql`.
- Después del evento: quitar el enlace de la burbuja del inicio; la página
  puede quedarse sin enlaces o borrarse.
