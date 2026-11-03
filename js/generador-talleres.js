/* ============================================
   GEIPER - Generador de contenido de la trayectoria
   Carga data/ponencias.json o data/talleres/<slug>.json,
   aplica lo que se llena en el formulario y entrega el
   archivo completo. No guarda nada: el cambio se hace
   en GitHub.
   ============================================ */
(function () {
  'use strict';

  var DATA = '../data/talleres/';
  var $ = function (id) { return document.getElementById(id); };
  var PONENCIAS = '__ponencias';
  var estado = { slug: null, original: null, modo: 'edicion', sel: 'nuevo', ponencias: null };
  function slugify(t) {
    return String(t || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  }
  function ordenPonencias(lista) {
    return lista.map(function (p, i) { return { p: p, i: i }; })
      .sort(function (a, b) { return (a.p.anio - b.p.anio) || (a.i - b.i); })
      .map(function (x) { return x.p; });
  }

  function esc(v) {
    return String(v == null ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function copia(o) { return JSON.parse(JSON.stringify(o)); }
  function base(p) { return p ? String(p).split('/').pop() : ''; }
  function lista(txt) { return String(txt || '').split(',').map(function (x) { return x.trim(); }).filter(Boolean); }
  function limpiarNombre(n) {
    return String(n || '').trim().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .replace(/\s+/g, '-').replace(/[^A-Za-z0-9._\/-]/g, '');
  }
  function enCarpeta(carpeta, nombre) {
    nombre = limpiarNombre(nombre);
    if (!nombre) return null;
    return nombre.indexOf('/') >= 0 ? nombre : carpeta + nombre;
  }
  function ordinal(n) { return n + '.ª edición'; }

  /* ── Carga inicial ─────────────────────────────── */
  Promise.all([
    fetch('../data/ponencias.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }),
    fetch(DATA + 'index.json', { cache: 'no-cache' }).then(function (r) { return r.json(); })
      .then(function (idx) {
        return Promise.all((idx.talleres || []).map(function (s) {
          return fetch(DATA + s + '.json', { cache: 'no-cache' }).then(function (r) { return r.json(); });
        }));
      })
  ])
    .then(function (res) {
      estado.ponencias = res[0];
      var talleres = res[1];
      $('gTaller').innerHTML = '<option value="' + PONENCIAS + '">Ponencias</option>' +
        '<optgroup label="Talleres">' + talleres.map(function (t) {
          return '<option value="' + esc(t.slug) + '">' + esc(t.titulo) + '</option>';
        }).join('') + '</optgroup>';
      var mapa = {};
      talleres.forEach(function (t) { mapa[t.slug] = t; });
      function elegirTaller() {
        estado.slug = $('gTaller').value;
        estado.original = estado.slug === PONENCIAS ? estado.ponencias : mapa[estado.slug];
        estado.sel = 'nuevo';
        $('gModoWrap').style.display = estado.slug === PONENCIAS ? 'none' : '';
        campos();
      }
      $('gTaller').addEventListener('change', elegirTaller);
      elegirTaller();
    })
    .catch(function () {
      $('eTaller').textContent = 'No se pudo cargar el contenido. Abre esta página desde el sitio publicado o con un servidor local.';
    });

  document.querySelectorAll('.tl-seg button').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('.tl-seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      estado.modo = b.dataset.modo;
      estado.sel = 'nuevo';
      campos();
    });
  });

  /* ── Campos según el modo ──────────────────────── */
  function campo(id, etiqueta, html, ayuda) {
    return '<div class="tl-campo"><label for="' + id + '">' + etiqueta + '</label>' + html +
      (ayuda ? '<span class="tl-ayuda">' + ayuda + '</span>' : '') +
      '<span class="tl-error" id="e_' + id + '"></span></div>';
  }
  function txt(id, val, ph) { return '<input id="' + id + '" value="' + esc(val) + '" placeholder="' + esc(ph || '') + '">'; }
  function area(id, val, filas, ph) { return '<textarea id="' + id + '" rows="' + filas + '" placeholder="' + esc(ph || '') + '">' + esc(val) + '</textarea>'; }

  function selector(items, nombre, etiquetaItem) {
    var opts = '<option value="nuevo">' + nombre + '</option>' + items.map(function (it, i) {
      return '<option value="' + i + '"' + (String(estado.sel) === String(i) ? ' selected' : '') + '>' + esc(etiquetaItem(it, i)) + '</option>';
    }).join('');
    return campo('gSel', 'Agregar o editar', '<select id="gSel">' + opts + '</select>');
  }

  function campos() {
    var t = estado.original;
    if (!t) return;
    var h = '', it = null;
    var slug = t.slug;

    if (estado.slug === PONENCIAS) {
      var lista = t.ponencias || [];
      h += selector(lista, 'Nueva ponencia', function (p) { return p.anio + ' · ' + p.ciudad + ' · ' + p.titulo; });
      it = estado.sel === 'nuevo' ? {} : lista[+estado.sel];
      h += campo('fAnio', 'Año', '<input id="fAnio" type="number" min="1990" max="2100" step="1" value="' + esc(it.anio || new Date().getFullYear()) + '">');
      h += campo('fEvento', 'Evento', txt('fEvento', it.evento, 'Conferencia de Geoinformación SELPER 2025'));
      h += campo('fCiudad', 'Ciudad', txt('fCiudad', it.ciudad, 'Mérida'));
      h += campo('fPais', 'País', txt('fPais', it.pais, 'México'));
      h += campo('fTitulo', 'Título de la ponencia', txt('fTitulo', it.titulo));
      h += campo('fPon', 'Ponentes', txt('fPon', (it.ponentes || []).join(', '), 'Nombre Apellido, Nombre Apellido'), 'Separados por coma. Opcional.');
      h += campo('fRes', 'Resumen', area('fRes', it.resumen, 4, 'Problema, enfoque y principal aporte.'));
      h += campo('fPdf', 'Presentación (PDF)', txt('fPdf', base(it.pdf), 'presentacion.pdf'));
      h += campo('fFotos', 'Fotos', txt('fFotos', (it.fotos || []).map(base).join(', '), 'foto1.webp, foto2.webp'), 'Separadas por coma. Máximo 10 recomendadas.');
      return pintarCampos(h);
    }

    if (estado.modo === 'taller') {
      h += campo('fTitulo', 'Título del taller', txt('fTitulo', t.titulo));
      h += campo('fFac', 'Facilitador o facilitadora', txt('fFac', t.facilitador, 'Ing. Nombre Apellido'));
      h += campo('fHer', 'Herramientas', txt('fHer', (t.herramientas || []).join(', '), 'R, Zotero, Mendeley'), 'Separadas por coma.');
      h += campo('fEst', 'Estado', '<select id="fEst"><option value="activo"' + (t.estado !== 'proximamente' ? ' selected' : '') + '>Activo (muestra el número de ediciones)</option><option value="proximamente"' + (t.estado === 'proximamente' ? ' selected' : '') + '>Próximamente</option></select>');
      h += campo('fCorto', 'Resumen corto (tarjeta y vista previa)', area('fCorto', t.resumen_corto, 2), 'Una o dos líneas.');
      h += campo('fRes', 'Resumen completo (página del taller)', area('fRes', t.resumen, 4));
      h += campo('fFoto', 'Foto destacada', txt('fFoto', base(t.foto_destacada), 'portada.webp'), 'Nombre del archivo. Se guarda en assets/talleres/' + esc(slug) + '/');
    }

    if (estado.modo === 'edicion') {
      var eds = t.ediciones || [];
      h += selector(eds, 'Nueva edición', function (e, i) { return (e.titulo || ordinal(i + 1)) + (e.semestre ? ' · ' + e.semestre : ''); });
      it = estado.sel === 'nuevo' ? {} : eds[+estado.sel];
      var nE = estado.sel === 'nuevo' ? eds.length + 1 : (it.orden || +estado.sel + 1);
      h += campo('fOrden', 'Número de edición', '<input id="fOrden" type="number" min="1" step="1" value="' + nE + '">');
      h += campo('fSem', 'Semestre', txt('fSem', it.semestre, '2025-II'));
      h += campo('fRes', 'Cómo fue esta edición', area('fRes', it.resumen, 4, 'Qué se trabajó, quiénes participaron y qué se logró.'));
      h += campo('fPdf', 'Presentación de cierre (PDF)', txt('fPdf', base(it.pdf), 'cierre.pdf'), 'Opcional.');
      h += campo('fFotos', 'Fotos', txt('fFotos', (it.fotos || []).map(base).join(', '), 'foto1.webp, foto2.webp'), 'Separadas por coma. Máximo 10 recomendadas.');
    }

    if (estado.modo === 'tema') {
      var tms = t.temas || [];
      h += selector(tms, 'Nuevo tema', function (e, i) { return (e.orden || i + 1) + '. ' + e.titulo; });
      it = estado.sel === 'nuevo' ? {} : tms[+estado.sel];
      var nT = estado.sel === 'nuevo' ? tms.length + 1 : (it.orden || +estado.sel + 1);
      h += campo('fOrden', 'Orden del tema', '<input id="fOrden" type="number" min="1" step="1" value="' + nT + '">');
      h += campo('fTitulo', 'Título del tema', txt('fTitulo', it.titulo, 'Ecuaciones de búsqueda'));
      h += campo('fRes', 'Resumen', area('fRes', it.resumen, 3, 'De qué trata el tema en una o dos frases.'));
      h += campo('fPdf', 'Presentación (PDF)', txt('fPdf', base(it.pdf), 'ecuaciones-busqueda.pdf'));
      h += campo('fFotos', 'Fotos', txt('fFotos', (it.fotos || []).map(base).join(', '), 'foto1.webp'), 'Opcional. Separadas por coma.');
      h += campo('fTip', 'Consejo de Píxel', area('fTip', it.tip_pixel, 2, 'Un consejo corto y práctico.'));
    }

    if (estado.modo === 'recurso') {
      var rec = t.recursos || [];
      h += selector(rec, 'Nuevo recurso', function (e) { return e.titulo; });
      it = estado.sel === 'nuevo' ? {} : rec[+estado.sel];
      h += campo('fTitulo', 'Nombre del recurso', txt('fTitulo', it.titulo, 'Descarga de SNAP (ESA)'));
      h += campo('fTipo', 'Tipo', '<select id="fTipo"><option value="enlace"' + (it.tipo !== 'descarga' ? ' selected' : '') + '>Enlace a un sitio externo</option><option value="descarga"' + (it.tipo === 'descarga' ? ' selected' : '') + '>Archivo para descargar</option></select>');
      h += campo('fUrl', 'Enlace o nombre del archivo', txt('fUrl', it.tipo === 'descarga' ? base(it.url) : it.url, 'https://…  o  guia-instalacion.pdf'));
    }

    pintarCampos(h);
  }

  function pintarCampos(h) {
    $('gCampos').innerHTML = h;
    var sel = $('gSel');
    if (sel) sel.addEventListener('change', function () { estado.sel = sel.value; campos(); });
    $('gCampos').querySelectorAll('input, textarea, select').forEach(function (el) {
      if (el.id === 'gSel') return;
      el.addEventListener('input', function () { var e = $('e_' + el.id); if (e) e.textContent = ''; $('eSalida').textContent = ''; generar(); });
      el.addEventListener('change', generar);
    });
    generar();
  }

  /* ── Construcción del archivo ──────────────────── */
  function val(id) { var el = $(id); return el ? el.value.trim() : ''; }

  function construirPonencias() {
    var d = copia(estado.original), errores = [];
    var nueva = estado.sel === 'nuevo';
    var p = nueva ? {} : copia(d.ponencias[+estado.sel]);
    var anio = Math.round(+val('fAnio'));
    if (!(anio >= 1990 && anio <= 2100)) errores.push(['fAnio', 'Escribe un año válido, por ejemplo 2026']);
    if (!val('fEvento')) errores.push(['fEvento', 'Escribe el nombre del evento']);
    if (!val('fCiudad')) errores.push(['fCiudad', 'Escribe la ciudad']);
    if (!val('fTitulo')) errores.push(['fTitulo', 'Escribe el título de la ponencia']);
    if (!p.id) p.id = (anio || 'sin-anio') + '-' + slugify(val('fCiudad')) + '-' + slugify(val('fTitulo')).split('-').slice(0, 3).join('-');
    var carpeta = 'assets/ponencias/' + p.id + '/';
    p.anio = anio;
    p.evento = val('fEvento');
    p.ciudad = val('fCiudad');
    p.pais = val('fPais');
    p.titulo = val('fTitulo');
    p.ponentes = lista(val('fPon'));
    p.resumen = val('fRes');
    p.pdf = enCarpeta(carpeta, val('fPdf'));
    p.fotos = lista(val('fFotos')).map(function (f) { return enCarpeta(carpeta, f); }).filter(Boolean);
    var archivos = [];
    if (p.pdf) archivos.push(base(p.pdf));
    p.fotos.forEach(function (f) { archivos.push(base(f)); });
    d.ponencias = d.ponencias || [];
    if (nueva) d.ponencias.push(p); else d.ponencias[+estado.sel] = p;
    var pos = ordenPonencias(d.ponencias).indexOf(p) + 1;
    return { t: d, archivo: 'data/ponencias.json', archivos: archivos, carpeta: carpeta,
             ancla: 'trayectoria.html#ponencia-' + pos, errores: errores };
  }

  function construir() {
    if (estado.slug === PONENCIAS) return construirPonencias();
    var t = copia(estado.original), slug = t.slug, raiz = 'assets/talleres/' + slug + '/';
    var archivos = [], carpeta = raiz, ancla = '', errores = [];

    if (estado.modo === 'taller') {
      if (!val('fTitulo')) errores.push(['fTitulo', 'Escribe el título del taller']);
      t.titulo = val('fTitulo');
      t.facilitador = val('fFac');
      t.herramientas = lista(val('fHer'));
      t.estado = val('fEst');
      t.resumen_corto = val('fCorto');
      t.resumen = val('fRes');
      t.foto_destacada = enCarpeta(raiz, val('fFoto'));
      if (t.foto_destacada) archivos.push(base(t.foto_destacada));
    }

    if (estado.modo === 'edicion' || estado.modo === 'tema') {
      var clave = estado.modo === 'edicion' ? 'ediciones' : 'temas';
      var n = Math.max(1, Math.round(+val('fOrden') || 1));
      carpeta = raiz + clave + '/' + n + '/';
      var o = estado.sel === 'nuevo' ? {} : copia(t[clave][+estado.sel]);
      o.tipo = estado.modo;
      o.orden = n;
      if (estado.modo === 'edicion') {
        o.titulo = ordinal(n);
        o.semestre = val('fSem');
      } else {
        if (!val('fTitulo')) errores.push(['fTitulo', 'Escribe el título del tema']);
        o.titulo = val('fTitulo');
      }
      o.resumen = val('fRes');
      o.pdf = enCarpeta(carpeta, val('fPdf'));
      o.fotos = lista(val('fFotos')).map(function (f) { return enCarpeta(carpeta, f); }).filter(Boolean);
      if (estado.modo === 'tema') o.tip_pixel = val('fTip');
      if (o.pdf) archivos.push(base(o.pdf));
      o.fotos.forEach(function (f) { archivos.push(base(f)); });

      t[clave] = t[clave] || [];
      var dup = t[clave].findIndex(function (x, i) { return x.orden === n && String(i) !== String(estado.sel); });
      if (dup >= 0) errores.push(['fOrden', 'Ya existe ' + (estado.modo === 'edicion' ? 'la edición ' : 'el tema ') + n + '. Elígelo arriba para editarlo o usa otro número.']);
      if (estado.sel === 'nuevo') t[clave].push(o); else t[clave][+estado.sel] = o;
      t[clave].sort(function (a, b) { return a.orden - b.orden; });
      ancla = '#' + estado.modo + '-' + (t[clave].indexOf(o) + 1);
    }

    if (estado.modo === 'recurso') {
      var r = { titulo: val('fTitulo'), tipo: val('fTipo'), url: val('fUrl') };
      if (!r.titulo) errores.push(['fTitulo', 'Escribe el nombre del recurso']);
      if (r.tipo === 'enlace') {
        if (!/^https?:\/\/\S+$/i.test(r.url)) errores.push(['fUrl', 'Escribe un enlace completo que empiece por https://']);
      } else {
        carpeta = raiz + 'recursos/';
        r.url = enCarpeta(carpeta, r.url);
        if (!r.url) errores.push(['fUrl', 'Escribe el nombre del archivo']);
        else archivos.push(base(r.url));
      }
      t.recursos = t.recursos || [];
      if (estado.sel === 'nuevo') t.recursos.push(r); else t.recursos[+estado.sel] = r;
    }

    return { t: t, archivo: 'data/talleres/' + slug + '.json', archivos: archivos, carpeta: carpeta,
             ancla: 'taller.html?t=' + slug + ancla, errores: errores };
  }

  var ultimo = null;
  function generar() {
    ultimo = construir();
    $('gNombre').textContent = ultimo.archivo;
    $('gSalida').textContent = JSON.stringify(ultimo.t, null, 2) + '\n';
    var p = '';
    if (ultimo.archivos.length) {
      p += '<li><span>En GitHub, entra a <code>' + esc(ultimo.carpeta) + '</code> (créala con <em>Add file → Upload files</em> si no existe) y sube: ' +
        ultimo.archivos.map(function (a) { return '<code>' + esc(a) + '</code>'; }).join(', ') +
        '. Cada archivo debe pesar menos de 25 MB; las fotos, idealmente en WebP.</span></li>';
    }
    p += '<li><span>Abre <code>' + esc(ultimo.archivo) + '</code>, pulsa el lápiz para editar, selecciona todo y pega el archivo copiado. También puedes subir el archivo descargado para reemplazarlo.</span></li>';
    p += '<li><span>Guarda con un mensaje como <code>Actualiza ' + esc(estado.slug === PONENCIAS ? 'ponencias' : estado.modo + ' de ' + estado.slug) + '</code>.</span></li>';
    p += '<li><span>En uno o dos minutos revisa <code>pages/' + esc(ultimo.ancla) + '</code>.</span></li>';
    $('gPasos').innerHTML = p;
  }

  function validar() {
    if (!ultimo) return false;
    ultimo = construir();
    ultimo.errores.forEach(function (e) { var el = $('e_' + e[0]); if (el) el.textContent = e[1]; });
    if (ultimo.errores.length) {
      $('eSalida').textContent = 'Revisa los campos marcados antes de continuar.';
      return false;
    }
    return true;
  }

  $('gCopiar').addEventListener('click', function () {
    if (!validar()) return;
    var texto = $('gSalida').textContent, b = this;
    function hecho() { b.textContent = 'Archivo copiado'; setTimeout(function () { b.textContent = 'Copiar archivo'; }, 2000); }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(texto).then(hecho, function () { $('eSalida').textContent = 'No se pudo copiar. Selecciona el texto y cópialo con Ctrl+C.'; });
    } else {
      $('eSalida').textContent = 'Tu navegador no permite copiar aquí. Selecciona el texto y cópialo con Ctrl+C.';
    }
  });

  $('gBajar').addEventListener('click', function () {
    if (!validar()) return;
    var blob = new Blob([$('gSalida').textContent], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = base(ultimo.archivo);
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  });
})();
