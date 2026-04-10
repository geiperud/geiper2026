/**
 * GEIPER Events Feed (RSS Estilo)
 * Lee dinámicamente desde un JSON y pinta las tarjetas bajo el contenido actual de eventos.html
 */

document.addEventListener('DOMContentLoaded', () => {
  const rssContainer = document.getElementById('rssFeedContainer');
  if(!rssContainer) return;

  const url = '../data/eventos_feed.json'; // Ruta al archivo JSON

  // Función asíncrona para buscar y pintar los eventos
  async function loadEvents() {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Error al cargar la fuente de eventos.');
      }
      const data = await response.json();
      
      if (data.length === 0) {
        rssContainer.innerHTML = '<p>No hay eventos próximos en este momento.</p>';
        return;
      }

      // Ordenar por fecha (más próximos primero)
      // Opcional: data.sort((a,b) => new Date(a.date) - new Date(b.date));

      let contentHTML = '';
      
      data.forEach(evento => {
        // Generar etiquetas (tags)
        let tagsHTML = '';
        evento.tags.forEach(tag => {
          tagsHTML += `<span class="event-tag">${tag}</span>`;
        });

        // Crear tarjeta (card)
        contentHTML += `
          <div class="event-feed-card">
            <div class="event-feed-date">
              <i class="fa-regular fa-calendar"></i> ${formatDate(evento.date)}
            </div>
            <div class="event-feed-type type-${evento.type.toLowerCase().trim()}">
              ${evento.type}
            </div>
            <h3 class="event-feed-title">${evento.title}</h3>
            <p class="event-feed-desc">${evento.description}</p>
            <div class="event-feed-tags">
              ${tagsHTML}
            </div>
            <a href="${evento.link}" class="btn event-feed-btn" target="_blank">Saber más →</a>
          </div>
        `;
      });
      
      rssContainer.innerHTML = contentHTML;

    } catch (error) {
      console.error(error);
      rssContainer.innerHTML = '<p style="color: #64748b;">No se pudieron cargar los eventos en este momento.</p>';
    }
  }

  // Función auxiliar para formatear la fecha
  function formatDate(dateStr) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    const d = new Date(dateStr);
    // Agregamos un ajuste de timezone rápido si es necesario, o directamente formato.
    return d.toLocaleDateString('es-ES', options);
  }

  loadEvents();
});
