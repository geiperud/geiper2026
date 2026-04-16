/**
 * GEIPER Events Feed (RSS Estilo)
 * Lee dinámicamente desde un JSON y pinta las tarjetas bajo el contenido actual de eventos.html
 */

document.addEventListener('DOMContentLoaded', () => {
  const rssContainer = document.getElementById('rssFeedContainer');
  if (!rssContainer) return;

  const url = '../data/eventos_feed.json';

  function sanitizeUrl(rawUrl) {
    try {
      const parsed = new URL(rawUrl);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
        return parsed.href;
      }
    } catch (_) {}
    return '#';
  }

  function formatDate(dateStr) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateStr).toLocaleDateString('es-ES', options);
  }

  function buildCard(evento) {
    const card = document.createElement('div');
    card.className = 'event-feed-card';

    // Fecha
    const dateDiv = document.createElement('div');
    dateDiv.className = 'event-feed-date';
    const calIcon = document.createElement('i');
    calIcon.className = 'fa-regular fa-calendar';
    dateDiv.appendChild(calIcon);
    dateDiv.appendChild(document.createTextNode(' ' + formatDate(evento.date)));
    card.appendChild(dateDiv);

    // Tipo
    const typeDiv = document.createElement('div');
    const safeType = String(evento.type).toLowerCase().trim();
    typeDiv.className = `event-feed-type type-${safeType}`;
    typeDiv.textContent = evento.type;
    card.appendChild(typeDiv);

    // Título
    const h3 = document.createElement('h3');
    h3.className = 'event-feed-title';
    h3.textContent = evento.title;
    card.appendChild(h3);

    // Descripción
    const p = document.createElement('p');
    p.className = 'event-feed-desc';
    p.textContent = evento.description;
    card.appendChild(p);

    // Tags
    const tagsDiv = document.createElement('div');
    tagsDiv.className = 'event-feed-tags';
    evento.tags.forEach(tag => {
      const span = document.createElement('span');
      span.className = 'event-tag';
      span.textContent = tag;
      tagsDiv.appendChild(span);
    });
    card.appendChild(tagsDiv);

    // Enlace
    const a = document.createElement('a');
    a.href = sanitizeUrl(evento.link);
    a.className = 'btn event-feed-btn';
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = 'Saber más →';
    card.appendChild(a);

    return card;
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

      data.forEach(evento => {
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
