/* ============================================
   GEIPER — Ventana "Acerca de este proyecto"
   Se abre desde el botón de la franja superior
   (cualquier elemento con el atributo data-acerca).
   ============================================ */
(function () {
  // Raíz del sitio a partir de la ruta de este script (js/acerca-proyecto.js)
  var script = document.currentScript;
  var raiz = script ? new URL('../', script.src).href : '';

  var dialog = null;

  function crearDialogo() {
    dialog = document.createElement('dialog');
    dialog.className = 'acerca-dialog';
    dialog.setAttribute('aria-labelledby', 'acercaTitulo');
    dialog.innerHTML =
      '<button type="button" class="acerca-cerrar" aria-label="Cerrar">&times;</button>' +
      '<div class="acerca-cuerpo">' +
        '<div class="acerca-cab">' +
          '<img src="' + raiz + 'assets/images/Logo_GEIPER.png" alt="">' +
          '<h2 id="acercaTitulo">Acerca de este proyecto</h2>' +
        '</div>' +
        '<p>Este sitio nace como el geoportal del Semillero de Investigación GEIPER, del grupo NIDE de la Universidad Distrital Francisco José de Caldas. Su propósito es reunir en un solo lugar las líneas de investigación, los proyectos, los datos geoespaciales y los repositorios abiertos del semillero, y acercar la percepción remota y los sistemas de información geográfica a estudiantes y a la comunidad.</p>' +
        '<p>Además de divulgar, el sitio integra un geovisor para consultar las capas producidas por los proyectos del semillero y asistentes de investigación basados en modelos de lenguaje de gran escala (LLM), que apoyan la revisión de literatura y la comprensión de la normativa académica de la universidad.</p>' +
        '<dl class="acerca-ficha">' +
          '<dt>Desarrollado por</dt><dd>Laura Dayana Díaz Beltrán</dd>' +
          '<dt>Programa</dt><dd>Ingeniería Catastral y Geodesia, Facultad de Ingeniería, UDFJC</dd>' +
          '<dt>Modalidad</dt><dd>Trabajo de grado, modalidad monografía</dd>' +
          '<dt>Título</dt><dd>Diseño e implementación de un geoportal con asistentes inteligentes basados en modelos de lenguaje de gran escala (LLM) para el Semillero de Investigación GEIPER de la UDFJC</dd>' +
          '<dt>Año</dt><dd>2026</dd>' +
        '</dl>' +
        '<div class="acerca-enlaces">' +
          '<a href="https://www.linkedin.com/in/laddiazb/" target="_blank" rel="noopener"><i class="fa-brands fa-linkedin" aria-hidden="true"></i>LinkedIn</a>' +
          '<a href="https://orcid.org/0009-0003-0125-5726" target="_blank" rel="noopener"><i class="fa-brands fa-orcid" aria-hidden="true"></i>ORCID</a>' +
          '<a href="https://github.com/geiperud/geiper2026" target="_blank" rel="noopener"><i class="fa-brands fa-github" aria-hidden="true"></i>Código fuente</a>' +
        '</div>' +
      '</div>';

    document.body.appendChild(dialog);

    dialog.querySelector('.acerca-cerrar').addEventListener('click', function () {
      dialog.close();
    });

    // Cerrar al hacer clic fuera del contenido
    dialog.addEventListener('click', function (e) {
      if (e.target === dialog) dialog.close();
    });
  }

  function abrir() {
    if (!dialog) crearDialogo();
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-acerca]');
    if (btn) { e.preventDefault(); abrir(); }
  });
})();
