/* ============================================
   GEIPER · Reto de las ponencias (pages/reto.html)
   - Jugador (celular y computador): registro, brújula,
     preguntas con 25 s, resultado y ganadores por ponencia.
   - ?modo=pantalla   → pantalla proyectada (ranking + QR)
   - ?modo=moderador  → abrir preguntas y cerrar el reto
   Todo lo decide el backend (backend/reto/trivia.py):
   esta página solo muestra y envía.
   ============================================ */
(function () {
  'use strict';

  var ES_LOCAL = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);
  var params = new URLSearchParams(location.search);
  // Solo en pruebas locales se permite apuntar a otro backend (?api=...)
  var API = (ES_LOCAL && params.get('api')) || 'https://geiper2026.onrender.com';
  var MODO = params.get('modo') || 'jugador';
  var POLL_JUGADOR = 15000, POLL_PANTALLA = 10000, POLL_MOD = 10000;
  var CLAVE_TOKEN = 'geiper_reto_token';
  // Títulos cortos para la banda y los podios (los completos se ven en el detalle)
  var CORTO = { '01': 'Huella de carbono', '02': 'Bibliometría', '03': 'Sabana Occidental',
                '04': 'Ortomosaico SAR', '05': 'RAG en SIG', '06': 'Territorios y catastro' };
  function corto(p) { return CORTO[p.id] || p.titulo; }

  function $(id) { return document.getElementById(id); }
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function miles(n) { return Number(n || 0).toLocaleString('es-CO'); }
  function hora(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleTimeString('es-CO', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Bogota' }); }
    catch (e) { return '—'; }
  }
  function guardar(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {} }
  function leer(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

  function api(ruta, opts) {
    opts = opts || {};
    var headers = { 'Content-Type': 'application/json' };
    if (opts.token) headers['X-Reto-Token'] = opts.token;
    if (opts.admin) headers['X-Reto-Admin'] = opts.admin;
    return fetch(API + '/reto' + ruta, {
      method: opts.method || 'GET',
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
      cache: 'no-store'
    }).then(function (r) {
      if (opts.raw && r.ok) return r;
      return r.json().catch(function () { return {}; }).then(function (d) {
        if (!r.ok) { var e = new Error(d.detail || 'Error ' + r.status); e.status = r.status; throw e; }
        return d;
      });
    });
  }

  function cadaCuanto(fn, ms) {
    // Solo consulta mientras la pestaña está visible: celulares bloqueados no cargan el servidor
    var t = null;
    function tick() { if (document.visibilityState === 'visible') fn(); }
    t = setInterval(tick, ms);
    document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'visible') fn(); });
    return t;
  }

  /* ════════════════════════════════════════════
     BRÚJULA (SVG estático + aguja + 6 nodos)
     ════════════════════════════════════════════ */
  function brujulaSVG() {
    var C = 170, R = 160, s = '', i, a, r2;
    var rad = function (d) { return d * Math.PI / 180; };
    var px = function (r, d) { return (C + r * Math.sin(rad(d))).toFixed(1); };
    var py = function (r, d) { return (C - r * Math.cos(rad(d))).toFixed(1); };
    s += '<circle cx="170" cy="170" r="' + R + '" fill="#fff" stroke="#0B2E6B" stroke-width="1.5"/>';
    s += '<circle cx="170" cy="170" r="' + (R - 14) + '" fill="#F5F7FB" stroke="#D5DCE6"/>';
    s += '<circle cx="170" cy="170" r="98" fill="none" stroke="#B7C1CF" stroke-dasharray="2 4"/>';
    for (i = 0; i < 360; i += 5) {
      var larga = i % 30 === 0; r2 = R - (larga ? 11 : 5);
      s += '<line x1="' + px(R, i) + '" y1="' + py(R, i) + '" x2="' + px(r2, i) + '" y2="' + py(r2, i) +
        '" stroke="' + (larga ? '#0B2E6B' : '#9AA6B8') + '" stroke-width="' + (larga ? 1.4 : 0.8) + '"/>';
    }
    [[0, 'N'], [90, 'E'], [180, 'S'], [270, 'O']].forEach(function (c) {
      s += '<text x="' + px(132, c[0]) + '" y="' + (Number(py(132, c[0])) + 8).toFixed(1) +
        '" text-anchor="middle" font-family="DM Serif Text, Georgia, serif" font-size="24" fill="#0B2E6B">' + c[1] + '</text>';
    });
    for (a = 0; a < 360; a += 45) {
      var grande = a % 90 === 0, L = grande ? 80 : 52, w = grande ? 12 : 9;
      var c1 = grande ? '#C9D5E6' : '#DCE3EE', c2 = grande ? '#E8EDF5' : '#F1F4F9';
      s += '<polygon points="' + px(L, a) + ',' + py(L, a) + ' ' + px(w, a - 45) + ',' + py(w, a - 45) + ' 170,170" fill="' + c1 + '"/>';
      s += '<polygon points="' + px(L, a) + ',' + py(L, a) + ' ' + px(w, a + 45) + ',' + py(w, a + 45) + ' 170,170" fill="' + c2 + '"/>';
    }
    var base = '<svg viewBox="0 0 340 340" width="340" height="340" aria-hidden="true">' + s + '</svg>';
    var aguja = '<svg viewBox="0 0 340 340" width="340" height="340" aria-hidden="true"><g class="rt-aguja" id="aguja">' +
      '<polygon points="170,86 179,170 170,170" fill="#D9A80E"/><polygon points="170,86 161,170 170,170" fill="#F7C21B"/>' +
      '<polygon points="170,220 179,170 170,170" fill="#0B2E6B"/><polygon points="170,220 161,170 170,170" fill="#1F4A8A"/>' +
      '<circle cx="170" cy="170" r="8" fill="#fff" stroke="#0B2E6B" stroke-width="2"/><circle cx="170" cy="170" r="2.5" fill="#0B2E6B"/></g></svg>';
    return base + aguja;
  }

  /* ════════════════════════════════════════════
     JUGADOR
     ════════════════════════════════════════════ */
  var J = {
    token: leer(CLAVE_TOKEN), datos: null, sel: null, fase: 'detalle',
    preg: null, reloj: null, finReloj: 0, respondida: false
  };
  var ESTADO_TXT = {
    hecha: ['Respondida', '#1B7F4B'],
    abierta: ['Preguntas abiertas', '#8A6A0A'],
    bloqueada: ['Aún no abre', '#5F6673']
  };

  function iniciarJugador() {
    $('vistaJugador').hidden = false;
    $('brujula').innerHTML = brujulaSVG();
    for (var i = 0; i < 6; i++) {
      var b = document.createElement('button');
      var ang = 60 * i * Math.PI / 180;
      b.type = 'button'; b.className = 'rt-nodo'; b.dataset.i = i;
      b.style.left = (170 + 98 * Math.sin(ang) - 22).toFixed(0) + 'px';
      b.style.top = (170 - 98 * Math.cos(ang) - 22).toFixed(0) + 'px';
      b.innerHTML = '<span></span>';
      b.addEventListener('click', function () { if (J.fase !== 'pregunta') elegir(+this.dataset.i); });
      $('brujula').appendChild(b);
    }
    $('formRegistro').addEventListener('submit', registrar);
    if (J.token) cargarEstado(true); else mostrar('secRegistro');
    cadaCuanto(function () { if (J.token) cargarEstado(false); }, POLL_JUGADOR);
  }

  function mostrar(id) {
    ['secRegistro', 'secJuego', 'secCerrado'].forEach(function (s) { $(s).hidden = s !== id; });
  }

  function registrar(ev) {
    ev.preventDefault();
    var alias = $('alias').value.trim(), correo = $('correo').value.trim().toLowerCase();
    var err = $('errRegistro');
    err.textContent = '';
    if (alias.length < 3) { err.textContent = 'Escriba un alias de al menos 3 caracteres.'; $('alias').focus(); return; }
    if (!/^[a-z0-9._%+-]+@udistrital\.edu\.co$/.test(correo)) { err.textContent = 'Use su correo institucional @udistrital.edu.co.'; $('correo').focus(); return; }
    if (!$('acepto').checked) { err.textContent = 'Debe autorizar el uso del correo para participar.'; return; }
    var btn = $('btnRegistro'); btn.disabled = true; btn.textContent = 'Conectando…';
    api('/registro', { method: 'POST', body: { alias: alias, correo: correo, acepto: true } })
      .then(function (d) { J.token = d.token; guardar(CLAVE_TOKEN, d.token); cargarEstado(true); })
      .catch(function (e) {
        err.textContent = e.status ? e.message : 'No hay conexión con el servidor. Espere unos segundos e intente de nuevo.';
      })
      .then(function () { btn.disabled = false; btn.textContent = 'Entrar al programa'; });
  }

  function cargarEstado(primera) {
    return api('/estado', { token: J.token }).then(function (d) {
      $('avisoConexion').hidden = true;
      J.datos = d;
      $('cabYo').innerHTML = 'Participando como <b>' + esc(d.alias) + '</b>';
      if (d.cerrado) { pintarCierre(d); return; }
      if (J.sel === null || primera) J.sel = elegirInicial(d.ponencias);
      mostrar('secJuego');
      pintarJuego();
    }).catch(function (e) {
      if (e.status === 401) { J.token = null; guardar(CLAVE_TOKEN, null); mostrar('secRegistro'); return; }
      var av = $('avisoConexion');
      av.hidden = false;
      av.textContent = 'Reconectando con el servidor… Si es la primera visita de la mañana puede tardar hasta un minuto.';
    });
  }

  function elegirInicial(p) {
    for (var i = 0; i < p.length; i++) if (p[i].estado === 'abierta') return i;
    for (i = 0; i < p.length; i++) if (p[i].estado === 'bloqueada') return i;
    return p.length - 1;
  }

  function elegir(i) { J.sel = i; J.fase = 'detalle'; pintarJuego(); }

  function pintarJuego() {
    var d = J.datos, p = d.ponencias, sel = p[J.sel];
    var hechas = p.filter(function (x) { return x.estado === 'hecha'; }).length;
    $('cifPonencias').textContent = hechas + ' / 6';
    $('cifPuntaje').textContent = miles(d.puntaje);
    $('cifPuesto').textContent = d.podios || 0;
    $('premiosCierre').textContent = 'Premios · cierre ' + hora(d.cierre_at);
    document.querySelectorAll('.rt-nodo').forEach(function (b, i) {
      var x = p[i];
      b.dataset.estado = x.estado;
      b.setAttribute('aria-pressed', i === J.sel ? 'true' : 'false');
      b.setAttribute('aria-label', 'Ponencia ' + x.id + ', ' + x.hora + ', ' + ESTADO_TXT[x.estado][0].toLowerCase());
    });
    $('aguja').style.transform = 'rotate(' + (60 * J.sel) + 'deg)';
    $('secJuego').classList.toggle('preguntando', J.fase !== 'detalle');
    // Detalle
    $('detMeta').textContent = 'Ponencia ' + sel.id + ' · ' + sel.hora + ' a. m. · ' + sel.sala;
    $('detTitulo').textContent = sel.titulo;
    $('detPonentes').textContent = sel.ponentes;
    var et = ESTADO_TXT[sel.estado];
    var txt = et[0], color = et[1];
    if (sel.estado === 'hecha') {
      txt = 'Respondida · ' + miles(sel.puntos) + ' puntos';
      if (sel.gano) { txt = 'Ganador · puesto ' + sel.mi_puesto + ' · ' + miles(sel.puntos) + ' puntos'; color = '#8A6A0A'; }
      else if (sel.mi_puesto) txt += sel.final ? ' · puesto final ' + sel.mi_puesto : ' · va en el puesto ' + sel.mi_puesto;
    } else if (sel.estado === 'bloqueada') {
      txt = sel.fase === 'cerrada' ? 'Preguntas cerradas' : 'Se abre al final de la charla de las ' + sel.hora;
    }
    $('detEstado').textContent = txt;
    $('detEstado').style.color = color;
    if (J.fase === 'detalle') pintarAccion(sel);
    pintarTop(d);
  }

  function pintarAccion(sel) {
    var z = $('zonaAccion');
    if (sel.estado === 'abierta') {
      var faltan = 3 - sel.respondidas;
      z.innerHTML = '<button class="rt-btn rt-bloque" type="button" id="btnEmpezar">' +
        (sel.respondidas ? 'Continuar · faltan ' + faltan : 'Responder las 3 preguntas') + '</button>';
      $('btnEmpezar').addEventListener('click', function () { pedirPregunta(sel.id); });
    } else if (sel.estado === 'hecha') {
      z.innerHTML = '<button class="rt-btn rt-btn-sec rt-bloque" type="button" id="btnResumen">Ver mis respuestas de esta ponencia</button>';
      $('btnResumen').addEventListener('click', function () { verResumen(sel.id); });
    } else {
      z.innerHTML = sel.fase === 'cerrada'
        ? '<p class="rt-aviso">Las preguntas de esta ponencia ya se cerraron. Abajo están sus ganadores.</p>'
        : '<p class="rt-aviso">Las preguntas se abren en los últimos minutos de la charla. Esta pantalla se actualiza sola.</p>';
    }
  }

  function pintarTop(d) {
    var sel = d.ponencias[J.sel], top = sel.top || [], h = '';
    $('topTitulo').textContent = 'Top 3 · ponencia ' + sel.id + ' · ' + (sel.final ? 'ganadores' : 'en vivo, se actualiza cada 15 s');
    top.forEach(function (r) {
      h += '<div><span class="rt-num">' + r.puesto + '.º</span><span class="alias">' + esc(r.alias) +
        '</span><span class="rt-num" style="font-size:18px">' + miles(r.puntos) + '</span></div>';
    });
    for (var k = top.length; k < 3; k++) h += '<div><span class="rt-num">' + (k + 1) + '.º</span><span class="alias" style="color:#9AA6B8">—</span><span></span></div>';
    h += '<div class="yo"><span class="rt-num">' + (sel.mi_puesto ? sel.mi_puesto + '.º' : '—') + '</span><span class="alias">' + esc(d.alias) +
      ' (usted)</span><span class="rt-num" style="font-size:18px">' + miles(sel.puntos) + '</span></div>';
    $('topRanking').innerHTML = h;
  }

  /* ── Preguntas ───────────────────────────── */
  function pedirPregunta(pid, reintento) {
    J.fase = 'pregunta';
    $('secJuego').classList.add('preguntando');
    api('/pregunta?ponencia=' + encodeURIComponent(pid), { token: J.token }).then(function (q) {
      if (q.terminada) { verResumen(pid); return; }
      // La anterior venció pero el servidor aún la tiene en su margen de gracia: esperar un momento
      if (J.preg && J.preg.n === q.n && J.preg.ponencia === q.ponencia && q.segundos_restantes <= 0 && (reintento || 0) < 4) {
        $('zonaAccion').innerHTML = '<p class="rt-aviso">Cargando la siguiente pregunta…</p>';
        setTimeout(function () { pedirPregunta(pid, (reintento || 0) + 1); }, 1500);
        return;
      }
      J.preg = q; J.respondida = false;
      pintarPregunta(q);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }).catch(function (e) {
      J.fase = 'detalle';
      $('zonaAccion').innerHTML = '<p class="rt-error" role="alert">' + esc(e.message) + '</p>';
      cargarEstado(false);
    });
  }

  function pintarPregunta(q) {
    var z = $('zonaAccion');
    var avance = '';
    for (var k = 1; k <= 3; k++) avance += '<i class="' + (k <= q.n ? 'on' : '') + '"></i>';
    var ops = q.opciones.map(function (o) {
      return '<button type="button" class="rt-op" data-op="' + o.id + '"><b>' + o.id.toUpperCase() + '</b><span>' + esc(o.texto) + '</span><small></small></button>';
    }).join('');
    z.innerHTML =
      '<div class="rt-caja">' +
        '<div class="rt-caja-cab"><span class="rt-eyebrow">Pregunta ' + q.n + ' de 3 · 100 puntos + bono por velocidad</span>' +
        '<span class="rt-num rt-reloj" id="reloj" aria-live="off">0:15</span></div>' +
        '<div class="rt-barra" role="progressbar" aria-label="Tiempo restante" aria-valuemin="0" aria-valuemax="' + q.limite_s + '" id="barraWrap"><i id="barra"></i></div>' +
        '<div class="rt-avance" aria-hidden="true">' + avance + '</div>' +
        '<h3 class="rt-enunciado">' + esc(q.enunciado) + '</h3>' +
        '<div class="rt-opciones" role="group" aria-label="Opciones de respuesta">' + ops + '</div>' +
        '<div class="rt-pie-preg"><span class="rt-feedback" id="feedback" role="status"></span><span id="sigWrap"></span></div>' +
      '</div>';
    z.querySelectorAll('.rt-op').forEach(function (b) {
      b.addEventListener('click', function () { responder(b.dataset.op); });
    });
    arrancarReloj(q.segundos_restantes, q.limite_s);
  }

  function arrancarReloj(restante, limite) {
    clearInterval(J.reloj);
    J.finReloj = performance.now() + restante * 1000;
    function tick() {
      var r = Math.max(0, (J.finReloj - performance.now()) / 1000);
      var el = $('reloj'); if (!el) { clearInterval(J.reloj); return; }
      el.textContent = '0:' + String(Math.ceil(r)).padStart(2, '0');
      el.classList.toggle('urgente', r <= 5);
      $('barra').style.width = (100 * r / limite).toFixed(1) + '%';
      $('barraWrap').setAttribute('aria-valuenow', Math.ceil(r));
      if (r <= 0) { clearInterval(J.reloj); if (!J.respondida) tiempoAgotado(); }
    }
    tick();
    J.reloj = setInterval(tick, 200);
  }

  function bloquearOpciones() {
    document.querySelectorAll('.rt-op').forEach(function (b) { b.disabled = true; });
  }

  function tiempoAgotado() {
    J.respondida = true;
    bloquearOpciones();
    document.querySelectorAll('.rt-op').forEach(function (b) { b.classList.add('apagada'); });
    $('feedback').textContent = 'Se acabó el tiempo: 0 puntos en esta pregunta.';
    $('feedback').style.color = '#B42318';
    botonSiguiente(J.preg.n >= 3);
  }

  function responder(op) {
    if (J.respondida) return;
    J.respondida = true;
    clearInterval(J.reloj);
    bloquearOpciones();
    var q = J.preg;
    api('/responder', { method: 'POST', token: J.token, body: { ponencia: q.ponencia, n: q.n, opcion: op } })
      .then(function (r) {
        document.querySelectorAll('.rt-op').forEach(function (b) {
          var id = b.dataset.op;
          if (id === r.correcta) { b.classList.add('ok'); b.querySelector('small').textContent = 'Correcta'; }
          else if (id === op) { b.classList.add('mal'); b.querySelector('small').textContent = 'Su respuesta'; }
          else b.classList.add('apagada');
        });
        var f = $('feedback');
        f.textContent = r.acierto ? 'Correcta: +' + r.puntos + ' puntos (' + (r.puntos - 100) + ' de bono por velocidad).' : 'Incorrecta: 0 puntos.';
        f.style.color = r.acierto ? '#1B7F4B' : '#B42318';
        botonSiguiente(r.ponencia_respondidas >= 3);
      })
      .catch(function (e) {
        var f = $('feedback');
        f.textContent = e.message; f.style.color = '#B42318';
        if (e.status === 423) { J.fase = 'detalle'; setTimeout(function () { cargarEstado(false); }, 1500); return; }
        botonSiguiente(q.n >= 3);
      });
  }

  function botonSiguiente(ultima) {
    var pid = J.preg.ponencia;
    $('sigWrap').innerHTML = '<button class="rt-btn" type="button" id="btnSig">' + (ultima ? 'Ver resultado de la ponencia' : 'Siguiente pregunta') + '</button>';
    $('btnSig').addEventListener('click', function () { ultima ? verResumen(pid) : pedirPregunta(pid); });
    $('btnSig').focus();
  }

  function verResumen(pid) {
    J.fase = 'resumen';
    clearInterval(J.reloj);
    $('secJuego').classList.add('preguntando');
    api('/resumen?ponencia=' + encodeURIComponent(pid), { token: J.token }).then(function (r) {
      var filas = r.preguntas.map(function (x) {
        var c = x.acierto ? '#1B2433' : '#B42318';
        return '<div><span>0' + x.n + '</span><span>' + esc(x.enunciado || 'Pregunta ' + x.n) + '</span><span style="color:' + c + '">' + x.puntos + '</span></div>';
      }).join('');
      $('zonaAccion').innerHTML =
        '<div class="rt-caja">' +
          '<div class="rt-caja-cab"><span class="rt-eyebrow">Ponencia ' + esc(pid) + ' completada</span>' +
          '<span class="rt-num" style="font-size:34px">' + miles(r.total) + '</span></div>' +
          '<div class="rt-tabla">' + filas + '</div>' +
          '<p style="margin:0;font-size:13px;line-height:1.5;color:#5F6673">Los 3 mejores puntajes de esta ponencia ganan; desempata el menor tiempo. Se anuncian al cerrar sus preguntas.</p>' +
          '<button class="rt-btn" type="button" id="btnVolver">Volver a la brújula</button>' +
        '</div>';
      $('btnVolver').addEventListener('click', function () { J.fase = 'detalle'; J.preg = null; cargarEstado(false); });
      cargarEstado(false);
    });
  }

  function pintarCierre(d) {
    clearInterval(J.reloj);
    mostrar('secCerrado');
    var h = '';
    d.ponencias.forEach(function (p) {
      var g = (p.top || []).map(function (r) { return r.puesto + '. ' + esc(r.alias); }).join(' · ') || 'Sin ganadores';
      h += '<div><span class="rt-num" style="font-size:28px">' + esc(p.id) + '</span><span class="alias" style="font-size:16px">' + esc(corto(p)) +
        '</span><span style="font-size:13px;color:#5F6673">' + g + '</span></div>';
    });
    $('podioJugador').innerHTML = h;
    var mios = d.ponencias.filter(function (p) { return p.gano; });
    $('cerradoYo').textContent = mios.length
      ? '¡Felicitaciones! Quedó entre los 3 primeros en: ' + mios.map(function (p) { return corto(p) + ' (puesto ' + p.mi_puesto + ')'; }).join(', ') + '.'
      : 'Sumó ' + miles(d.puntaje) + ' puntos en total. Gracias por participar.';
  }

  /* ════════════════════════════════════════════
     PANTALLA PROYECTADA
     ════════════════════════════════════════════ */
  function iniciarPantalla() {
    $('vistaPantalla').hidden = false;
    var url = location.origin + location.pathname;
    $('qrUrl').textContent = url.replace(/^https?:\/\//, '');
    var s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js';
    s.onload = function () {
      try { new window.QRCode($('qr'), { text: url, width: 240, height: 240, colorDark: '#0B2E6B', colorLight: '#ffffff' }); } catch (e) {}
    };
    s.onerror = function () { $('qr').hidden = true; $('qrUrl').style.fontSize = '18px'; $('qrUrl').style.color = '#0B2E6B'; };
    document.head.appendChild(s);
    function cargar() {
      api('/publico').then(function (d) {
        var FASE = { abierta: 'Preguntas abiertas', cerrada: 'Ganadores', proxima: 'Próxima' };
        var b = '', pod = '';
        d.ponencias.forEach(function (p) {
          var f = d.cerrado && p.fase === 'abierta' ? 'cerrada' : p.fase;
          b += '<div class="' + f + '"><div class="m"><i></i><i></i><i></i></div><span>' + esc(p.id) + ' · ' + esc(p.hora) +
            '</span><b>' + esc(corto(p)) + '</b><em>' + FASE[f] + '</em></div>';
          var filas = (p.top || []).map(function (r) {
            return '<li><span class="n">' + r.puesto + '</span><span class="a">' + esc(r.alias) + '</span><span class="s">' + miles(r.puntos) + '</span></li>';
          }).join('');
          pod += '<section class="rt-podio-pon ' + f + '"><header><b>' + esc(p.id) + ' · ' + esc(corto(p)) + '</b><em>' +
            (f === 'cerrada' ? 'Ganadores' : f === 'abierta' ? 'En vivo' : esc(p.hora)) + '</em></header>' +
            (filas ? '<ol>' + filas + '</ol>' : '<p class="vacio">' + (f === 'proxima' ? 'Aún no abre' : 'Esperando respuestas…') + '</p>') + '</section>';
        });
        $('proyBanda').innerHTML = b;
        $('proyPodios').innerHTML = pod;
        var abiertas = d.ponencias.filter(function (p) { return p.fase === 'abierta'; }).length;
        $('proyBandaTit').textContent = '6 ponencias · ' + (d.cerrado ? 'preguntas cerradas' : abiertas ? abiertas + ' con preguntas abiertas' : 'jueves 26 de noviembre');
        $('proyCierre').textContent = d.cerrado ? 'Cerrado' : hora(d.cierre_at);
        $('proyEstado').textContent = d.cerrado ? 'Resultados finales' : 'Ranking en vivo';
        $('proyParticipantes').textContent = d.participantes + (d.participantes === 1 ? ' participante registrado' : ' participantes registrados');
      }).catch(function () { $('proyEstado').textContent = 'Reconectando…'; });
    }
    cargar();
    cadaCuanto(cargar, POLL_PANTALLA);
  }

  /* ════════════════════════════════════════════
     MODERADOR
     ════════════════════════════════════════════ */
  var M = { clave: null, datos: null };
  function iniciarModerador() {
    $('vistaModerador').hidden = false;
    try { M.clave = sessionStorage.getItem('geiper_reto_admin'); } catch (e) {}
    $('formClave').addEventListener('submit', function (ev) {
      ev.preventDefault();
      M.clave = $('clave').value.trim();
      $('errClave').textContent = 'Conectando…';
      cargarMod(true);
    });
    $('btnCerrar').addEventListener('click', function () {
      var cerrar = !(M.datos && M.datos.cerrado);
      var msg = cerrar ? '¿Cerrar todas las preguntas? Desde este momento el servidor rechaza todas las respuestas y los ganadores quedan como finales.'
                       : '¿Reabrir las preguntas? Solo hágalo si se cerraron por error.';
      if (!window.confirm(msg)) return;
      accionMod('/admin/cerrar', { cerrar: cerrar });
    });
    $('btnCsv').addEventListener('click', function () {
      api('/admin/csv', { admin: M.clave, raw: true }).then(function (r) { return r.blob(); }).then(function (blob) {
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob); a.download = 'reto_ponencias_resultados.csv';
        document.body.appendChild(a); a.click(); a.remove();
      }).catch(function (e) { $('errMod').textContent = e.message; });
    });
    if (M.clave) cargarMod(false);
    cadaCuanto(function () { if (M.clave && !$('modPanel').hidden) cargarMod(false); }, POLL_MOD);
  }

  function cargarMod(manual) {
    api('/admin/estado', { admin: M.clave }).then(function (d) {
      try { sessionStorage.setItem('geiper_reto_admin', M.clave); } catch (e) {}
      $('formClave').hidden = true; $('modPanel').hidden = false;
      pintarMod(d);
    }).catch(function (e) {
      if (e.status === 403) {
        try { sessionStorage.removeItem('geiper_reto_admin'); } catch (x) {}
        $('formClave').hidden = false; $('modPanel').hidden = true;
        if (manual) $('errClave').textContent = e.message;
      } else {
        // Si aún no se ha entrado, el mensaje va debajo del botón (el panel está oculto)
        var msg = e.status ? 'Error del servidor (' + e.status + '): ' + e.message
                           : 'Sin conexión con el servidor. Si Render estaba dormido, espere un minuto y vuelva a intentar.';
        if ($('modPanel').hidden) $('errClave').textContent = msg; else $('errMod').textContent = msg;
      }
    });
  }

  function accionMod(ruta, body) {
    $('errMod').textContent = '';
    api(ruta, { method: 'POST', admin: M.clave, body: body }).then(pintarMod)
      .catch(function (e) { $('errMod').textContent = e.message; });
  }

  function pintarMod(d) {
    M.datos = d;
    $('errMod').textContent = '';
    $('modCifras').innerHTML = '<b>' + d.participantes + '</b> participantes · <b>' + d.respuestas + '</b> respuestas';
    $('modEstadoReto').textContent = d.cerrado
      ? 'Las preguntas están CERRADAS: no se reciben respuestas y los ganadores son finales.'
      : 'Las preguntas están habilitadas. Cierre automático: ' + hora(d.cierre_at);
    $('btnCerrar').textContent = d.cerrado ? 'Reabrir (solo si fue un error)' : 'Cerrar todo y mostrar los ganadores';
    var h = '<div><span class="rt-eyebrow">N.º</span><span class="rt-eyebrow">Hora</span><span class="rt-eyebrow">Ponencia</span>' +
      '<span class="rt-eyebrow">Auditorio</span><span class="rt-eyebrow">Preguntas</span><span class="rt-eyebrow">Estado</span><span></span></div>';
    d.ponencias.forEach(function (p) {
      var listas = p.preguntas_cargadas >= 3;
      var top = (p.top || []).map(function (r) { return '<b>' + r.puesto + '.</b> ' + esc(r.alias) + ' (' + r.puntos + ')'; }).join(' · ');
      var est = p.abierta ? ['Abierta', '#8A6A0A'] : p.fase === 'cerrada' ? ['Cerrada', '#1B7F4B'] : ['Sin abrir', '#5F6673'];
      var boton = p.abierta ? 'Cerrar y anunciar' : p.fase === 'cerrada' ? 'Reabrir' : 'Abrir preguntas';
      h += '<div><span class="num">' + esc(p.id) + '</span><span>' + esc(p.hora) + '</span>' +
        '<span class="t"><b>' + esc(p.titulo) + '</b><span>' + esc(p.ponentes) + '</span>' +
        (top ? '<div class="top">' + (p.fase === 'cerrada' ? 'Ganadores: ' : 'Va ganando: ') + top + '</div>' : '') + '</span>' +
        '<span style="font-size:14px;color:#5F6673">' + esc(p.sala) + '</span>' +
        '<span class="' + (listas ? '' : 'falta') + '">' + p.preguntas_cargadas + ' / 3</span>' +
        '<span class="rt-estado" style="color:' + est[1] + '">' + est[0] + '</span>' +
        '<button type="button" class="rt-btn ' + (p.abierta ? 'rt-btn-sec' : '') + '" style="min-height:42px" data-id="' + esc(p.id) + '" data-abrir="' + (!p.abierta) + '"' +
        (listas || p.abierta ? '' : ' disabled title="Faltan preguntas por cargar en Supabase"') + '>' + boton + '</button></div>';
    });
    $('modTabla').innerHTML = h;
    $('modTabla').querySelectorAll('button[data-id]').forEach(function (b) {
      b.addEventListener('click', function () { accionMod('/admin/ponencia', { ponencia: b.dataset.id, abierta: b.dataset.abrir === 'true' }); });
    });
  }

  /* ── Arranque ───────────────────────────── */
  if (MODO === 'pantalla') iniciarPantalla();
  else if (MODO === 'moderador') iniciarModerador();
  else iniciarJugador();
})();
