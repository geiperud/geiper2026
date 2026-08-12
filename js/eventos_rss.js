/**
 * GEIPER Events Feed (RSS Estilo)
 * Lee dinámicamente desde un JSON y pinta las tarjetas bajo el contenido actual de eventos.html
 */

document.addEventListener('DOMContentLoaded', () => {
  const rssContainer = document.getElementById('rssFeedContainer');
  if (!rssContainer) return;

  const url = rssContainer.dataset.feedUrl || '../data/eventos_feed.json';
  const limit = rssContainer.dataset.limit ? parseInt(rssContainer.dataset.limit, 10) : null;

  function sanitizeUrl(rawUrl) {
    try {
      const parsed = new URL(rawUrl);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
        return parsed.href;
      }
    } catch (_) {}
    return '#';
  }

  const MESES = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC'];

  function buildCard(evento) {
    const fecha = new Date(evento.date + 'T00:00:00');

    const row = document.createElement('div');
    row.className = 'feed-row';

    // Bloque de fecha (día grande + mes/año)
    const dateBlock = document.createElement('div');
    dateBlock.className = 'feed-date-block';
    const day = document.createElement('div');
    day.className = 'feed-day';
    day.textContent = String(fecha.getDate()).padStart(2, '0');
    const month = document.createElement('div');
    month.className = 'feed-month';
    month.textContent = `${MESES[fecha.getMonth()]} ${fecha.getFullYear()}`;
    dateBlock.appendChild(day);
    dateBlock.appendChild(month);
    row.appendChild(dateBlock);

    // Contenido
    const content = document.createElement('div');

    const typeDiv = document.createElement('div');
    typeDiv.className = 'feed-type';
    typeDiv.textContent = evento.type;
    content.appendChild(typeDiv);

    const h3 = document.createElement('h3');
    h3.className = 'feed-title';
    h3.textContent = evento.title;
    content.appendChild(h3);

    const p = document.createElement('p');
    p.className = 'feed-desc';
    p.textContent = evento.description;
    content.appendChild(p);

    const tagsDiv = document.createElement('div');
    tagsDiv.className = 'feed-tags';
    evento.tags.forEach(tag => {
      const span = document.createElement('span');
      span.textContent = tag;
      tagsDiv.appendChild(span);
    });
    content.appendChild(tagsDiv);

    const a = document.createElement('a');
    a.href = sanitizeUrl(evento.link);
    a.className = 'feed-link';
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = 'Saber más →';
    content.appendChild(a);

    row.appendChild(content);
    return row;
  }

  async function loadEvents() {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Error al cargar la fuente de eventos.');
      }
      const data = await response.json();

      if (data.length === 0) {
        const msg = document.createElement('p');
        msg.textContent = 'No hay eventos próximos en este momento.';
        rssContainer.appendChild(msg);
        return;
      }

      const sorted = [...data].sort((a, b) => new Date(b.date) - new Date(a.date));
      const eventos = limit ? sorted.slice(0, limit) : sorted;
      eventos.forEach(evento => {
        rssContainer.appendChild(buildCard(evento));
      });

    } catch (error) {
      console.error(error);
      const msg = document.createElement('p');
      msg.style.color = '#64748b';
      msg.textContent = 'No se pudieron cargar los eventos en este momento.';
      rssContainer.appendChild(msg);
    }
  }

  loadEvents();
});
