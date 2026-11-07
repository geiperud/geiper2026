/* ============================================
   GEIPER - Trayectoria: ponencias y talleres
   - trayectoria.html: órbita de ponencias (#tlPonencias)
     + tarjetas de talleres con vista previa (#tlCards)
   - taller.html?t=<slug>: página completa (#tlTaller)
   El contenido vive en data/ponencias.json y
   data/talleres/<slug>.json, y se edita con
   pages/generador-talleres.html.
   ============================================ */
(function () {
  'use strict';

  var DATA = '../data/talleres/';
  var PIXEL = '../assets/talleres/geiper-pixel-mascota-colibri.webp';
  var PDFJS = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/';
  var SLUG_OK = /^[a-z0-9-]+$/;

  /* ── Utilidades ─────────────────────────────── */
  function esc(v) {
    return String(v == null ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function ruta(p) {
    if (!p) return null;
    if (/^https?:\/\//i.test(p)) return p;
    return '../' + String(p).replace(/^\/+/, '');
  }
  function urlSegura(u) {
    return /^https?:\/\//i.test(u || '') ? u : null;
  }
  function cargar(slug) {
    return fetch(DATA + slug + '.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error('No se encontró ' + slug);
      return r.json();
    });
  }
  function porOrden(lista) {
    return (lista || []).slice().sort(function (a, b) { return (a.orden || 0) - (b.orden || 0); });
  }
  function etiquetaEstado(t) {
    if (t.estado === 'proximamente') return 'Próximamente';
    var n = (t.ediciones || []).length;
    return n === 1 ? '1 edición' : n + ' ediciones';
  }
  /* Punto sobre una curva cuadrática de Bézier */
  function bezier(t, p0, p1, p2) {
    var u = 1 - t;
    return [
      u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
      u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
    ];
  }

  /* ── Mini órbita de las tarjetas ───────────────── */
  function miniOrbita(n) {
    var s = '';
    for (var i = 0; i < n; i++) {
      var p = bezier((i + 1) / (n + 1), [2, 32], [56, -14], [110, 32]);
      s += '<rect x="' + (p[0] - 4).toFixed(1) + '" y="' + (p[1] - 4).toFixed(1) +
           '" width="8" height="8" fill="#0B2E6B"/>';
    }
    return '<svg class="tl-card-orbit" viewBox="0 0 112 34" aria-hidden="true">' +
      '<path d="M2 32 Q56 -14 110 32" fill="none" stroke="#3F6FB5" stroke-width="1" stroke-dasharray="3 4"/>' +
      s + '</svg>';
  }

  /* ============================================
     TALLERES: TARJETAS + VISTA PREVIA (trayectoria.html)
     ============================================ */
  function iniciarListado(cont) {
    fetch(DATA + 'index.json', { cache: 'no-cache' })
      .then(function (r) { return r.json(); })
      .then(function (idx) {
        var slugs = (idx.talleres || []).filter(function (s) { return SLUG_OK.test(s); });
        return Promise.all(slugs.map(function (s) { return cargar(s).catch(function () { return null; }); }));
      })
      .then(function (talleres) {
        talleres = talleres.filter(Boolean);
        if (!talleres.length) {
          cont.innerHTML = '<p class="tl-estado-msg">Pronto publicaremos los talleres del semillero.</p>';
          return;
        }
        cont.innerHTML = talleres.map(function (t, i) {
          var num = String(i + 1).padStart(2, '0');
          return '<button type="button" class="tl-card" data-i="' + i + '" aria-haspopup="dialog">' +
            '<span class="tl-card-top"><span class="tl-card-num" aria-hidden="true">' + num + '</span>' +
            miniOrbita(t.estado === 'proximamente' ? 0 : (t.ediciones || []).length) + '</span>' +
            '<h3>' + esc(t.titulo) + '</h3>' +
            '<span class="tl-card-fac">' + esc(t.facilitador) + '</span>' +
            '<span class="tl-card-pie"><span class="tl-etiqueta">' + esc(etiquetaEstado(t)) + '</span>' +
            '<span class="tl-card-ver">Vista previa</span></span>' +
            '</button>';
        }).join('');
        cont.querySelectorAll('.tl-card').forEach(function (b) {
          b.addEventListener('click', function () { abrirPrevia(talleres[+b.dataset.i], b); });
        });
      })
      .catch(function () {
        cont.innerHTML = '<p class="tl-estado-msg">No pudimos cargar los talleres. Recarga la página para intentarlo de nuevo.</p>';
      });
  }

  function abrirPrevia(t, origen) {
    var d = document.getElementById('tlPrevia');
    if (!d) return;
    var foto = ruta(t.foto_destacada);
    var media = foto
      ? '<img class="tl-prev-foto" src="' + esc(foto) + '" alt="Foto del ' + esc(t.titulo) + '">'
      : '<div class="tl-prev-vacio"><img src="' + PIXEL + '" alt=""></div>';
    d.innerHTML = media +
      '<div class="tl-prev-cuerpo">' +
        '<div class="tl-prev-cab"><span class="tl-etiqueta">' + esc(etiquetaEstado(t)) + '</span>' +
        '<button type="button" class="tl-cerrar" aria-label="Cerrar vista previa"><i class="fa-solid fa-xmark"></i></button></div>' +
        '<h3 id="tlPreviaTitulo">' + esc(t.titulo) + '</h3>' +
        '<span class="tl-card-fac">' + esc(t.facilitador) + '</span>' +
        '<p>' + esc(t.resumen_corto || t.resumen) + '</p>' +
        '<a class="tl-btn" href="taller.html?t=' + encodeURIComponent(t.slug) + '">Ver taller completo</a>' +
      '</div>';
    d.setAttribute('aria-labelledby', 'tlPreviaTitulo');
    d.querySelector('.tl-cerrar').addEventListener('click', function () { d.close(); });
    d.onclose = function () { if (origen) origen.focus(); };
    d.onclick = function (e) { if (e.target === d) d.close(); };
    if (d.showModal) d.showModal(); else d.setAttribute('open', '');
  }

  /* ============================================
     PÁGINA DE TALLER (taller.html)
     ============================================ */
  function iniciarTaller(cont) {
    var slug = new URLSearchParams(location.search).get('t') || '';
    if (!SLUG_OK.test(slug)) { noEncontrado(cont); return; }
    cargar(slug).then(function (t) { pintarTaller(cont, t); }).catch(function () { noEncontrado(cont); });
  }

  function noEncontrado(cont) {
    cont.innerHTML = '<div class="tl-wrap"><div class="tl-migas"><a href="trayectoria.html#talleres">Trayectoria</a></div>' +
      '<div class="tl-espera" style="margin-top:48px"><img src="' + PIXEL + '" alt="Píxel, la mascota de GEIPER">' +
      '<p>No encontramos este taller. Revisa el enlace o vuelve a <a href="trayectoria.html#talleres" style="color:var(--azul-medio)">la lista de talleres</a>.</p></div>' +
      '<div class="tl-fin"></div></div>';
  }

  function pintarTaller(cont, t) {
    document.title = t.titulo + ' – Semillero GEIPER';
    var md = document.querySelector('meta[name="description"]');
    if (md && t.resumen_corto) md.setAttribute('content', t.resumen_corto);

    var ediciones = porOrden(t.ediciones);
    var temas = porOrden(t.temas);
    var foto = ruta(t.foto_destacada);
    var largo = (t.titulo || '').length > 45 ? ' tl-largo' : '';

    var h = '<div class="tl-wrap">' +
      '<div class="tl-migas" role="navigation" aria-label="Ruta"><a href="trayectoria.html#talleres">Trayectoria</a> / ' + esc(t.titulo) + '</div>' +
      '<div class="tl-hero' + largo + '">' +
        '<span class="tl-etiqueta">' + esc(etiquetaEstado(t)) + '</span>' +
        '<h1>' + esc(t.titulo) + '</h1>' +
        '<p class="tl-hero-res">' + esc(t.resumen) + '</p>' +
        '<dl class="tl-meta">' +
          '<div><dt>' + (t.estado === 'proximamente' ? 'Facilitadora' : 'Facilitador') + '</dt><dd>' + esc(t.facilitador) + '</dd></div>' +
          '<div><dt>Herramientas</dt><dd>' + esc((t.herramientas || []).join(', ')) + '</dd></div>' +
          '<div><dt>Ediciones</dt><dd>' + (t.estado === 'proximamente' ? 'Primera en preparación' : ediciones.length) + '</dd></div>' +
        '</dl>' +
      '</div>' +
      (foto ? '<img class="tl-foto-dest" src="' + esc(foto) + '" alt="Foto del ' + esc(t.titulo) + '">' : '');

    /* Ediciones */
    h += '<section class="tl-sec" aria-labelledby="tlSecEd">' +
      '<div class="tl-sec-cab"><h2 id="tlSecEd">Ediciones</h2><span class="tl-sec-sub">La experiencia de cada semestre</span></div>';
    if (ediciones.length) {
      h += '<div class="tl-orbita tl-ediciones" id="tlOrbEd" role="group" aria-label="Ediciones del taller"></div>' +
           '<div class="tl-panel" id="tlPanelEd" aria-live="polite"></div>';
    } else {
      h += '<div class="tl-espera"><img src="' + PIXEL + '" alt="Píxel, la mascota de GEIPER">' +
           '<p>La primera edición de este taller está en preparación. Cuando se realice, aquí contaremos cómo fue.</p></div>';
    }
    h += '</section>';

    /* Temas */
    if (temas.length) {
      h += '<section class="tl-sec" aria-labelledby="tlSecTe">' +
        '<div class="tl-sec-cab"><h2 id="tlSecTe">Temas</h2><span class="tl-sec-sub">El contenido del taller, con un consejo de Píxel en cada uno</span></div>' +
        '<div class="tl-orbita tl-temas" id="tlOrbTe" role="group" aria-label="Temas del taller"></div>' +
        '<div class="tl-panel" id="tlPanelTe" aria-live="polite"></div>' +
        '</section>';
    }

    /* Recursos */
    var recursos = (t.recursos || []).filter(function (r) { return urlSegura(ruta(r.url)) || (r.url && !/^[a-z]+:/i.test(r.url)); });
    if (recursos.length) {
      h += '<section class="tl-sec" aria-labelledby="tlSecRe"><div class="tl-sec-cab"><h2 id="tlSecRe">Recursos generales</h2></div><ul class="tl-recursos">' +
        recursos.map(function (r) {
          var desc = r.tipo === 'descarga';
          var href = ruta(r.url);
          return '<li><a href="' + esc(href) + '"' + (desc ? ' download' : ' target="_blank" rel="noopener"') + '>' +
            '<i class="fa-solid ' + (desc ? 'fa-file-arrow-down' : 'fa-link') + '" aria-hidden="true"></i>' +
            '<span>' + esc(r.titulo) + '</span>' +
            '<span class="tl-ext">' + (desc ? 'Descargar' : 'Abrir sitio') + '</span></a></li>';
        }).join('') + '</ul></section>';
    }

    h += '<div class="tl-fin"></div></div>';
    cont.innerHTML = h;

    var ctrlEd = ediciones.length ? orbita(document.getElementById('tlOrbEd'), ediciones, 'edicion',
      function (e, i) { panelEdicion(document.getElementById('tlPanelEd'), e, i, ediciones.length); }) : null;
    var ctrlTe = temas.length ? orbita(document.getElementById('tlOrbTe'), temas, 'tema',
      function (e, i) { panelTema(document.getElementById('tlPanelTe'), e, i, temas.length); }) : null;

    /* Enlaces directos: #edicion-2, #tema-3 */
    var m = /^#(edicion|tema)-(\d+)$/.exec(location.hash);
    var inicialEd = ediciones.length - 1, inicialTe = 0;
    if (m) {
      var n = +m[2] - 1;
      if (m[1] === 'edicion' && ediciones[n]) inicialEd = n;
      if (m[1] === 'tema' && temas[n]) inicialTe = n;
    }
    if (ctrlEd) ctrlEd.elegir(inicialEd, true);
    if (ctrlTe) ctrlTe.elegir(inicialTe, true);
    if (m) {
      var destino = document.getElementById(m[1] === 'edicion' ? 'tlOrbEd' : 'tlOrbTe');
      if (destino) setTimeout(function () { destino.scrollIntoView({ block: 'start' }); }, 50);
    }
  }

  /* ── Órbita: arco con nodos seleccionables ────── */
  function orbita(cont, items, tipo, alElegir) {
    var p0 = [4, 82], p1 = [50, -32], p2 = [96, 82];
    var svg = '<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">' +
      '<path d="M' + p0 + ' Q' + p1 + ' ' + p2 + '"/></svg>';
    var nodos = items.map(function (it, i) {
      var p = bezier((i + 1) / (items.length + 1), p0, p1, p2);
      var corto, largo;
      if (tipo === 'edicion') {
        corto = it.titulo || (i + 1) + '.ª edición';
        largo = corto + (it.semestre ? ' · ' + it.semestre : '');
      } else if (tipo === 'ponencia') {
        corto = it.ciudad || '';
        largo = it.anio + ' · ' + (it.ciudad || '');
      } else {
        corto = String(it.orden || i + 1);
        largo = (it.orden || i + 1) + '. ' + (it.titulo || '');
      }
      return '<button type="button" class="tl-nodo" data-i="' + i + '" aria-pressed="false" ' +
        'style="left:' + p[0].toFixed(2) + '%;top:' + p[1].toFixed(2) + '%" ' +
        'aria-label="' + esc(tipo === 'ponencia' ? largo + ': ' + it.titulo : largo) + '">' +
        '<span class="tl-marca" aria-hidden="true"></span>' +
        '<span class="tl-nodo-txt" aria-hidden="true"><span class="tl-nt-corto">' + esc(tipo === 'edicion' ? largo : corto) + '</span>' +
        '<span class="tl-nt-largo">' + esc(largo) + '</span></span></button>';
    }).join('');
    /* Ponencias: el año se escribe debajo de cada grupo de nodos */
    var anios = '';
    if (tipo === 'ponencia') {
      var grupos = {};
      items.forEach(function (it, i) {
        var x = bezier((i + 1) / (items.length + 1), p0, p1, p2)[0];
        (grupos[it.anio] = grupos[it.anio] || []).push(x);
      });
      Object.keys(grupos).forEach(function (a) {
        var xs = grupos[a];
        var medio = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2;
        anios += '<span class="tl-anio" aria-hidden="true" style="left:' + medio.toFixed(2) + '%">' + esc(a) + '</span>';
      });
    }
    cont.innerHTML = svg + nodos + anios;
    var botones = cont.querySelectorAll('.tl-nodo');

    function elegir(i, silencioso) {
      botones.forEach(function (b, j) { b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
      alElegir(items[i], i);
      if (!silencioso && history.replaceState) history.replaceState(null, '', '#' + tipo + '-' + (i + 1));
      var b = botones[i];
      if (b && cont.scrollWidth > cont.clientWidth) b.scrollIntoView({ block: 'nearest', inline: 'center' });
    }
    botones.forEach(function (b) {
      b.addEventListener('click', function () { elegir(+b.dataset.i); });
      b.addEventListener('keydown', function (e) {
        var i = +b.dataset.i, n = botones.length, sig = null;
        if (e.key === 'ArrowRight') sig = (i + 1) % n;
        if (e.key === 'ArrowLeft') sig = (i - 1 + n) % n;
        if (sig !== null) { e.preventDefault(); botones[sig].focus(); elegir(sig); }
      });
    });
    return { elegir: elegir };
  }

  /* ── Paneles ───────────────────────────────────── */
  function panelEdicion(panel, e, i, total) {
    panel.innerHTML =
      '<div class="tl-panel-meta">Edición ' + (i + 1) + ' de ' + total + (e.semestre ? ' · ' + esc(e.semestre) : '') + '</div>' +
      '<h3>' + esc(e.titulo || (i + 1) + '.ª edición') + '</h3>' +
      (e.resumen ? '<p class="tl-panel-res">' + esc(e.resumen) + '</p>'
                 : '<p class="tl-vacio">Pronto contaremos la experiencia de esta edición.</p>') +
      galeria(e.fotos, e.titulo) +
      (e.pdf ? '<div class="tl-visor" data-pdf="' + esc(ruta(e.pdf)) + '"></div>' : '');
    activar(panel);
  }

  function panelTema(panel, e, i, total) {
    panel.innerHTML =
      '<div class="tl-panel-meta">Tema ' + (i + 1) + ' de ' + total + '</div>' +
      '<h3>' + esc(e.titulo) + '</h3>' +
      (e.resumen ? '<p class="tl-panel-res">' + esc(e.resumen) + '</p>' : '') +
      (e.pdf ? '<div class="tl-visor" data-pdf="' + esc(ruta(e.pdf)) + '"></div>'
             : '<p class="tl-vacio" style="margin-top:14px">La presentación de este tema estará disponible pronto.</p>') +
      galeria(e.fotos, e.titulo) +
      (e.tip_pixel ? '<div class="tl-tip"><img src="' + PIXEL + '" alt="Píxel, la mascota de GEIPER">' +
        '<div><div class="tl-tip-nombre">Consejo de Píxel</div><p>' + esc(e.tip_pixel) + '</p></div></div>' : '');
    activar(panel);
  }

  function galeria(fotos, titulo) {
    fotos = (fotos || []).filter(Boolean);
    if (!fotos.length) return '';
    return '<div class="tl-galeria">' + fotos.map(function (f, k) {
      var src = esc(ruta(f));
      return '<button type="button" data-src="' + src + '" aria-label="Ampliar foto ' + (k + 1) + '">' +
        '<img src="' + src + '" alt="Foto ' + (k + 1) + ' de ' + esc(titulo) + '" loading="lazy"></button>';
    }).join('') + '</div>';
  }

  function activar(panel) {
    panel.querySelectorAll('.tl-galeria button').forEach(function (b) {
      b.addEventListener('click', function () { ampliar(b.dataset.src, b.querySelector('img').alt); });
    });
    var v = panel.querySelector('.tl-visor');
    if (v) visor(v, v.dataset.pdf);
  }

  /* ── Lightbox ──────────────────────────────────── */
  function ampliar(src, alt) {
    var d = document.getElementById('tlLightbox');
    if (!d) return;
    d.innerHTML = '<button type="button" class="tl-cerrar" aria-label="Cerrar foto"><i class="fa-solid fa-xmark"></i></button>' +
      '<img src="' + esc(src) + '" alt="' + esc(alt) + '">';
    d.querySelector('.tl-cerrar').addEventListener('click', function () { d.close(); });
    d.onclick = function (e) { if (e.target === d) d.close(); };
    d.showModal();
  }

  /* ── Visor PDF (PDF.js, carga diferida) ────────── */
  var pdfjsPromesa = null;
  function cargarPdfjs() {
    if (window.pdfjsLib) return Promise.resolve(window.pdfjsLib);
    if (pdfjsPromesa) return pdfjsPromesa;
    pdfjsPromesa = new Promise(function (ok, mal) {
      var s = document.createElement('script');
      s.src = PDFJS + 'pdf.min.js';
      s.onload = function () {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS + 'pdf.worker.min.js';
        ok(window.pdfjsLib);
      };
      s.onerror = function () { pdfjsPromesa = null; mal(new Error('pdfjs')); };
      document.head.appendChild(s);
    });
    return pdfjsPromesa;
  }

  function visor(cont, url) {
    cont.innerHTML =
      '<div class="tl-visor-barra"><span><i class="fa-regular fa-file-pdf" aria-hidden="true"></i> Presentación</span>' +
      '<span class="tl-visor-ctrl">' +
        '<button type="button" data-a="prev" aria-label="Diapositiva anterior"><i class="fa-solid fa-chevron-left"></i></button>' +
        '<span class="tl-visor-pag" aria-live="polite">…</span>' +
        '<button type="button" data-a="next" aria-label="Diapositiva siguiente"><i class="fa-solid fa-chevron-right"></i></button>' +
        '<a href="' + esc(url) + '" download aria-label="Descargar presentación"><i class="fa-solid fa-download"></i></a>' +
      '</span></div>' +
      '<div class="tl-visor-lienzo"><p class="tl-visor-msg">Cargando presentación…</p></div>';

    var lienzo = cont.querySelector('.tl-visor-lienzo');
    var pag = cont.querySelector('.tl-visor-pag');
    var bPrev = cont.querySelector('[data-a="prev"]');
    var bNext = cont.querySelector('[data-a="next"]');
    var doc = null, actual = 1, tarea = null;

    function pintar(n) {
      if (!doc) return;
      actual = n;
      pag.textContent = n + ' / ' + doc.numPages;
      bPrev.disabled = n <= 1;
      bNext.disabled = n >= doc.numPages;
      doc.getPage(n).then(function (p) {
        var ancho = Math.max(lienzo.clientWidth, 280);
        var base = p.getViewport({ scale: 1 });
        var dpr = window.devicePixelRatio || 1;
        var vp = p.getViewport({ scale: (ancho / base.width) * dpr });
        var c = document.createElement('canvas');
        c.width = vp.width; c.height = vp.height;
        c.style.width = ancho + 'px';
        c.setAttribute('role', 'img');
        c.setAttribute('aria-label', 'Diapositiva ' + n + ' de ' + doc.numPages);
        if (tarea) tarea.cancel();
        tarea = p.render({ canvasContext: c.getContext('2d'), viewport: vp });
        tarea.promise.then(function () { lienzo.innerHTML = ''; lienzo.appendChild(c); }).catch(function () {});
      });
    }

    bPrev.addEventListener('click', function () { if (actual > 1) pintar(actual - 1); });
    bNext.addEventListener('click', function () { if (doc && actual < doc.numPages) pintar(actual + 1); });

    cargarPdfjs()
      .then(function (lib) { return lib.getDocument(url).promise; })
      .then(function (d) { doc = d; pintar(1); })
      .catch(function () {
        pag.textContent = '';
        bPrev.disabled = bNext.disabled = true;
        lienzo.innerHTML = '<p class="tl-visor-msg">No pudimos mostrar la presentación aquí. Usa el botón de descarga para verla.</p>';
      });
  }

  /* ============================================
     PONENCIAS (trayectoria.html)
     ============================================ */
  function iniciarPonencias(cont) {
    fetch('../data/ponencias.json', { cache: 'no-cache' })
      .then(function (r) { if (!r.ok) throw new Error('ponencias'); return r.json(); })
      .then(function (d) {
        var lista = (d.ponencias || []).map(function (p, i) { return { p: p, i: i }; })
          .sort(function (a, b) { return (a.p.anio - b.p.anio) || (a.i - b.i); })
          .map(function (x) { return x.p; });
        if (!lista.length) {
          cont.innerHTML = '<p class="tl-estado-msg">Pronto publicaremos las ponencias del semillero.</p>';
          return;
        }
        cont.innerHTML = '<div class="tl-orbita tl-ponencias" id="tlOrbPo" role="group" aria-label="Ponencias por año"></div>' +
          '<div class="tl-panel" id="tlPanelPo" aria-live="polite"></div>';
        var ctrl = orbita(document.getElementById('tlOrbPo'), lista, 'ponencia', function (p, i) {
          panelPonencia(document.getElementById('tlPanelPo'), p, i, lista.length);
        });
        var m = /^#ponencia-(\d+)$/.exec(location.hash);
        var inicial = lista.length - 1;
        if (m && lista[+m[1] - 1]) inicial = +m[1] - 1;
        ctrl.elegir(inicial, true);
        if (m) setTimeout(function () { document.getElementById('ponencias').scrollIntoView({ block: 'start' }); }, 50);
      })
      .catch(function () {
        cont.innerHTML = '<p class="tl-estado-msg">No pudimos cargar las ponencias. Recarga la página para intentarlo de nuevo.</p>';
      });
  }

  function panelPonencia(panel, p, i, total) {
    var lugar = [p.ciudad, p.pais].filter(Boolean).join(', ');
    var ponentes = (p.ponentes || []).filter(Boolean);
    panel.innerHTML =
      '<div class="tl-panel-cab"><span class="tl-etiqueta">' + esc(p.evento) + '</span>' +
      '<span class="tl-panel-meta">' + esc(lugar) + ' · ' + esc(p.anio) + '</span></div>' +
      '<h3>' + esc(p.titulo) + '</h3>' +
      (ponentes.length ? '<p class="tl-ponentes">' + (ponentes.length > 1 ? 'Ponentes: ' : 'Ponente: ') + esc(ponentes.join(', ')) + '</p>' : '') +
      (p.resumen ? '<p class="tl-panel-res">' + esc(p.resumen) + '</p>'
                 : '<p class="tl-vacio">Pronto agregaremos el resumen y la presentación de esta ponencia.</p>') +
      (p.pdf ? '<div class="tl-visor" data-pdf="' + esc(ruta(p.pdf)) + '"></div>' : '') +
      galeria(p.fotos, p.titulo);
    activar(panel);
  }

  /* ── Arranque ───────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', function () {
    var pon = document.getElementById('tlPonencias');
    if (pon) iniciarPonencias(pon);
    var cards = document.getElementById('tlCards');
    if (cards) iniciarListado(cards);
    var taller = document.getElementById('tlTaller');
    if (taller) iniciarTaller(taller);
  });
})();
