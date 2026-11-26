/* ============================================
   GEIPER · Aviso de las Preguntas de las ponencias (temporal)
   Ventana emergente del inicio que lleva a pages/reto.html.
   Solo aparece el día del evento (hora de Colombia) y una
   vez por sesión. Para probarla otro día: index.html?reto=1
   Para desactivarla: AVISO_ACTIVO = false, o quitar las
   2 líneas (css/js) que la cargan en index.html.
   ============================================ */
(function () {
  'use strict';
  var AVISO_ACTIVO = true;
  var DIA = '2026-11-26';          // fecha del evento
  var HASTA_HORA = 12;             // deja de aparecer a las 12:00 m.
  var URL_RETO = 'pages/reto.html';
  var API = 'https://geiper2026.onrender.com/reto/publico';
  var CLAVE = 'geiperRetoAvisoVisto';
  var PROGRAMA = [
    ['01', '9:00', 'Huella de carbono'], ['02', '9:30', 'Bibliometría'],
    ['03', '10:00', 'Sabana Occidental'], ['04', '10:00', 'Ortomosaico SAR'],
    ['05', '10:20', 'RAG en SIG'], ['06', '10:30', 'Territorios y catastro']
  ];

  function hoyBogota() {
    try {
      var p = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Bogota', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', hour12: false })
        .formatToParts(new Date());
      var g = {}; p.forEach(function (x) { g[x.type] = x.value; });
      return { fecha: g.year + '-' + g.month + '-' + g.day, hora: +g.hour };
    } catch (e) { return { fecha: '', hora: 0 }; }
  }
  var forzar = /[?&]reto=1\b/.test(location.search);
  var ahora = hoyBogota();
  if (!forzar) {
    if (!AVISO_ACTIVO || ahora.fecha !== DIA || ahora.hora >= HASTA_HORA) return;
    try { if (sessionStorage.getItem(CLAVE)) return; } catch (e) {}
  }

  var brujula =
    '<svg class="ra-brujula" viewBox="0 0 200 200" aria-hidden="true">' +
      '<circle cx="100" cy="100" r="94" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="1.5"/>' +
      '<circle cx="100" cy="100" r="80" fill="rgba(255,255,255,.05)" stroke="rgba(255,255,255,.18)"/>' +
      marcas() +
      '<text x="100" y="40" text-anchor="middle" font-family="DM Serif Text, Georgia, serif" font-size="17" fill="#fff">N</text>' +
      '<text x="166" y="106" text-anchor="middle" font-family="DM Serif Text, Georgia, serif" font-size="17" fill="#fff">E</text>' +
      '<text x="100" y="172" text-anchor="middle" font-family="DM Serif Text, Georgia, serif" font-size="17" fill="#fff">S</text>' +
      '<text x="34" y="106" text-anchor="middle" font-family="DM Serif Text, Georgia, serif" font-size="17" fill="#fff">O</text>' +
      nodos() +
      '<g class="ra-aguja"><polygon points="100,48 106,100 100,100" fill="#D9A80E"/><polygon points="100,48 94,100 100,100" fill="#F7C21B"/>' +
      '<polygon points="100,132 106,100 100,100" fill="#B9C8E2"/><polygon points="100,132 94,100 100,100" fill="#fff"/></g>' +
      '<circle cx="100" cy="100" r="6" fill="#0B2E6B" stroke="#fff" stroke-width="2"/>' +
    '</svg>';
  function marcas() {
    var s = '';
    for (var d = 0; d < 360; d += 10) {
      var a = d * Math.PI / 180, l = d % 90 === 0 ? 9 : 4;
      s += '<line x1="' + (100 + 94 * Math.sin(a)).toFixed(1) + '" y1="' + (100 - 94 * Math.cos(a)).toFixed(1) +
        '" x2="' + (100 + (94 - l) * Math.sin(a)).toFixed(1) + '" y2="' + (100 - (94 - l) * Math.cos(a)).toFixed(1) +
        '" stroke="rgba(255,255,255,' + (d % 90 === 0 ? '.9' : '.45') + ')" stroke-width="1.2"/>';
    }
    return s;
  }
  function nodos() {
    var s = '';
    for (var i = 0; i < 6; i++) {
      var a = (60 * i + 30) * Math.PI / 180;
      s += '<circle cx="' + (100 + 62 * Math.sin(a)).toFixed(1) + '" cy="' + (100 - 62 * Math.cos(a)).toFixed(1) +
        '" r="4.5" fill="' + (i < 2 ? '#F7C21B' : 'rgba(255,255,255,.25)') + '"/>';
    }
    return s;
  }

  var lista = PROGRAMA.map(function (p) {
    return '<li><span class="h">' + p[1] + '</span><span class="t">' + p[2] + '</span><span class="e" data-id="' + p[0] + '"></span></li>';
  }).join('');

  var ov = document.createElement('div');
  ov.id = 'raOverlay';
  ov.innerHTML =
    '<div class="ra-modal" role="dialog" aria-modal="true" aria-labelledby="raTitulo" aria-describedby="raTexto">' +
      '<div class="ra-lado">' +
        '<span class="ra-vivo"><i></i>En vivo hoy</span>' +
        brujula +
        '<div class="ra-fecha"><b>Jueves 26 de noviembre</b><span>9:00 a 11:30 a. m. · Aduanilla de Paiba</span></div>' +
      '</div>' +
      '<div class="ra-cuerpo">' +
        '<span class="ra-eyebrow">Semana de Ingeniería Catastral y Geodesia 2026</span>' +
        '<h2 class="ra-titulo" id="raTitulo">Preguntas de las ponencias</h2>' +
        '<div class="ra-subraya"></div>' +
        '<p class="ra-texto" id="raTexto">Seis charlas del semillero, tres preguntas por cada una. Cuando termina una ponencia se abren sus preguntas: ' +
          'respóndalas desde el celular en <b>25 segundos</b> y compita por el <b>podio de esa charla</b>. Los 3 mejores puntajes de cada ponencia se llevan <b>premios personalizados del semillero</b>.</p>' +
        '<ul class="ra-programa" aria-label="Programa de ponencias">' + lista + '</ul>' +
        '<p class="ra-estado" id="raEstado" aria-live="polite"></p>' +
        '<div class="ra-datos"><span>Correo @udistrital.edu.co</span><span>Un registro por persona</span><span>Gana el mejor puntaje, desempata la rapidez</span></div>' +
        '<div class="ra-acciones">' +
          '<a class="ra-cta" href="' + URL_RETO + '" id="raIr">Responder preguntas <i class="fa-solid fa-arrow-right" aria-hidden="true"></i></a>' +
          '<button type="button" class="ra-luego" id="raLuego">Ahora no</button>' +
        '</div>' +
        '<p class="ra-url">Desde otro dispositivo: <b>geiperud.github.io/geiper2026/reto</b></p>' +
      '</div>' +
      '<button type="button" class="ra-cerrar" id="raCerrar" aria-label="Cerrar aviso"><i class="fa-solid fa-xmark" aria-hidden="true"></i></button>' +
    '</div>';

  var previo = null;
  function cerrar() {
    try { sessionStorage.setItem(CLAVE, '1'); } catch (e) {}
    ov.classList.remove('ra-visible');
    document.removeEventListener('keydown', teclas);
    setTimeout(function () { ov.remove(); if (previo && previo.focus) previo.focus(); }, 350);
  }
  function teclas(e) {
    if (e.key === 'Escape') { cerrar(); return; }
    if (e.key !== 'Tab') return;
    var f = ov.querySelectorAll('a[href], button');
    var primero = f[0], ultimo = f[f.length - 1];
    if (e.shiftKey && document.activeElement === primero) { e.preventDefault(); ultimo.focus(); }
    else if (!e.shiftKey && document.activeElement === ultimo) { e.preventDefault(); primero.focus(); }
  }

  function abrir() {
    document.body.appendChild(ov);
    previo = document.activeElement;
    requestAnimationFrame(function () { ov.classList.add('ra-visible'); });
    document.getElementById('raIr').focus({ preventScroll: true });
    document.getElementById('raLuego').addEventListener('click', cerrar);
    document.getElementById('raCerrar').addEventListener('click', cerrar);
    document.getElementById('raIr').addEventListener('click', function () { try { sessionStorage.setItem(CLAVE, '1'); } catch (e) {} });
    ov.addEventListener('click', function (e) { if (e.target === ov) cerrar(); });
    document.addEventListener('keydown', teclas);
    estadoEnVivo();
  }

  // Estado real de las ponencias (si el servidor responde a tiempo; si no, el aviso igual funciona)
  function estadoEnVivo() {
    if (!window.fetch) return;
    var ctrl = window.AbortController ? new AbortController() : null;
    var t = setTimeout(function () { if (ctrl) ctrl.abort(); }, 8000);
    fetch(API, { cache: 'no-store', signal: ctrl ? ctrl.signal : undefined }).then(function (r) { return r.json(); }).then(function (d) {
      clearTimeout(t);
      var abiertas = [];
      (d.ponencias || []).forEach(function (p) {
        var el = ov.querySelector('.e[data-id="' + p.id + '"]');
        if (!el) return;
        var fase = d.cerrado && p.fase === 'abierta' ? 'cerrada' : p.fase;
        el.className = 'e ' + fase;
        el.title = fase === 'abierta' ? 'Preguntas abiertas' : fase === 'cerrada' ? 'Ya tiene ganadores' : 'Próxima';
        if (fase === 'abierta') abiertas.push(p.id);
      });
      var est = document.getElementById('raEstado');
      if (d.cerrado) est.textContent = 'Las preguntas ya cerraron: entre para ver los ganadores de cada ponencia.';
      else if (abiertas.length) est.innerHTML = '<b>Preguntas abiertas ahora</b> en ' + (abiertas.length === 1 ? 'la ponencia ' : 'las ponencias ') + abiertas.join(' y ') + '.';
      else est.textContent = 'Las preguntas de cada charla se abren en sus últimos minutos.';
    }).catch(function () { clearTimeout(t); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { setTimeout(abrir, 900); });
  else setTimeout(abrir, 900);
})();
