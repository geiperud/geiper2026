/* ============================================================
   Observatorio bibliométrico — IA y RAG en geociencias
   Dibuja los gráficos en SVG (sin dependencias externas) y
   gestiona la tabla filtrable de los estudios de la revisión.
   Datos: ../data/observatorio/observatorio.json
   ============================================================ */
(function () {
  'use strict';
  var AZUL = '#0B2E6B', NARANJA = '#F08A1C', GRIS = '#6B7280', LINEA = '#D1D5DB', TINTA = '#111827';
  var SERIE = [AZUL, NARANJA, '#3F6FB5', GRIS, '#B45309', TINTA];
  var NS = 'http://www.w3.org/2000/svg';
  var D = null;

  function el(n, a, t) {
    var e = document.createElementNS(NS, n);
    for (var k in a) if (a.hasOwnProperty(k)) e.setAttribute(k, a[k]);
    if (t !== undefined) e.textContent = t;
    return e;
  }
  /* ───────── Red de coocurrencia interactiva ─────────
     Parte de la disposición calculada en el análisis (la misma de la figura del
     documento) y añade tres cosas que una imagen no puede dar: arrastrar un nodo
     para despejar su vecindad, filtrar por clúster y resaltar las conexiones de
     un término al pasar el cursor. La simulación de fuerzas es propia: con 38
     nodos no hace falta ninguna biblioteca externa. */
  function redInteractiva(cont, R, opts) {
    opts = opts || {};
    if (!cont || !R) return;
    var W = opts.ancho || 900, H = opts.alto || 620;
    var M = Math.max(34, Math.min(60, W * 0.07)); // margen para las etiquetas del borde
    var COLOR = { 1: AZUL, 2: NARANJA, 3: '#3F6FB5', 4: '#8C6D1F', 5: GRIS };

    // En pantallas estrechas se dibujan menos términos: con 38 la red no se lee.
    var listaN = R.nodos;
    if (opts.tope && opts.tope < listaN.length) {
      listaN = listaN.slice().sort(function (a, b) { return b.n - a.n; }).slice(0, opts.tope);
    }
    var maxN = Math.max.apply(null, listaN.map(function (d) { return d.n; }));
    var maxW = Math.max.apply(null, R.aristas.map(function (d) { return d.w; }));
    var indice = {};
    var nodos = listaN.map(function (d, i) {
      indice[d.p] = i;
      return { p: d.p, n: d.n, c: d.c, r: 5 + 13 * Math.sqrt(d.n / maxN),
               x: M + d.x * (W - 2 * M), y: M + (1 - d.y) * (H - 2 * M), vx: 0, vy: 0, fijo: false };
    });
    var aristas = R.aristas.map(function (e) {
      return { o: indice[e.a], d: indice[e.b], w: e.w };
    }).filter(function (e) { return e.o !== undefined && e.d !== undefined; });

    var vecinos = nodos.map(function () { return {}; });
    aristas.forEach(function (e) { vecinos[e.o][e.d] = 1; vecinos[e.d][e.o] = 1; });

    var s = svgBase(W, H);
    s.style.touchAction = 'none';
    s.style.cursor = 'grab';
    var gE = el('g', {}), gN = el('g', {}), gT = el('g', {});
    s.appendChild(gE); s.appendChild(gN); s.appendChild(gT);

    var lineas = aristas.map(function (e) {
      var l = el('line', { stroke: '#C7CCD4', 'stroke-width': 0.4 + 1.9 * e.w / maxW, 'stroke-opacity': 0.75 });
      gE.appendChild(l); return l;
    });
    var circulos = [], etiquetas = [];
    nodos.forEach(function (d, i) {
      var c = el('circle', { r: d.r, fill: COLOR[d.c] || GRIS, stroke: '#FFFFFF', 'stroke-width': 1.4 });
      c.style.cursor = 'grab';
      c.appendChild(el('title', {}, d.p + ': ' + fmt(d.n) + ' documentos'));
      gN.appendChild(c); circulos.push(c);
      var t = el('text', { 'text-anchor': 'middle', 'font-size': 10 + 3 * (d.n / maxN),
                           'pointer-events': 'none', style: 'fill:#111827' }, d.p);
      gT.appendChild(t); etiquetas.push(t);
    });

    /* Separa en vertical las etiquetas que se solapan. Se ejecuta solo cuando la
       red queda quieta: durante el arrastre estorbaría más de lo que ayuda. */
    function acomodaEtiquetas() {
      var cajas = etiquetas.map(function (t, i) {
        var anchoT = t.textContent.length * parseFloat(t.getAttribute('font-size')) * 0.52;
        return { i: i, x: nodos[i].x - anchoT / 2, y: nodos[i].y - nodos[i].r - 16, w: anchoT, h: 13, dy: 0 };
      }).sort(function (a, b) { return a.y - b.y; });
      for (var k = 0; k < 24; k++) {
        var movido = false;
        for (var i = 0; i < cajas.length; i++) {
          for (var j = i + 1; j < cajas.length; j++) {
            var a = cajas[i], b = cajas[j];
            var ya = a.y + a.dy, yb = b.y + b.dy;
            if (a.x < b.x + b.w && a.x + a.w > b.x && ya < yb + b.h && ya + a.h > yb) {
              a.dy -= 1.6; b.dy += 1.6; movido = true;
            }
          }
        }
        if (!movido) break;
      }
      cajas.forEach(function (c) {
        var anchoT = c.w;
        etiquetas[c.i].setAttribute('x', Math.min(W - anchoT / 2 - 3, Math.max(anchoT / 2 + 3, nodos[c.i].x)));
        etiquetas[c.i].setAttribute('y', nodos[c.i].y - nodos[c.i].r - 5 + c.dy);
      });
    }

    function pinta() {
      aristas.forEach(function (e, i) {
        var a = nodos[e.o], b = nodos[e.d];
        lineas[i].setAttribute('x1', a.x); lineas[i].setAttribute('y1', a.y);
        lineas[i].setAttribute('x2', b.x); lineas[i].setAttribute('y2', b.y);
      });
      nodos.forEach(function (d, i) {
        circulos[i].setAttribute('cx', d.x); circulos[i].setAttribute('cy', d.y);
        var anchoT = d.p.length * parseFloat(etiquetas[i].getAttribute('font-size')) * 0.52;
        etiquetas[i].setAttribute('x', Math.min(W - anchoT / 2 - 3, Math.max(anchoT / 2 + 3, d.x)));
        etiquetas[i].setAttribute('y', d.y - d.r - 5);
      });
    }

    /* Un paso de simulación: repulsión entre todos los nodos, resorte en cada
       enlace y una atracción suave al centro para que nada se escape del lienzo. */
    function paso(amortigua) {
      var i, j, a, b, dx, dy, dist, f;
      for (i = 0; i < nodos.length; i++) {
        a = nodos[i];
        for (j = i + 1; j < nodos.length; j++) {
          b = nodos[j];
          dx = b.x - a.x; dy = b.y - a.y;
          dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
          var min = a.r + b.r + 26;
          f = (6200 + (dist < min ? 14000 : 0)) / (dist * dist);
          dx /= dist; dy /= dist;
          a.vx -= f * dx; a.vy -= f * dy;
          b.vx += f * dx; b.vy += f * dy;
        }
      }
      aristas.forEach(function (e) {
        var a = nodos[e.o], b = nodos[e.d];
        var dx = b.x - a.x, dy = b.y - a.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        var reposo = 125 - 45 * (e.w / maxW);
        var f = 0.012 * (dist - reposo);
        dx /= dist; dy /= dist;
        a.vx += f * dx; a.vy += f * dy;
        b.vx -= f * dx; b.vy -= f * dy;
      });
      nodos.forEach(function (d) {
        d.vx += (W / 2 - d.x) * 0.0008;
        d.vy += (H / 2 - d.y) * 0.0013;
        if (d.fijo) { d.vx = 0; d.vy = 0; return; }
        d.vx *= amortigua; d.vy *= amortigua;
        d.x = Math.min(W - M / 2, Math.max(M / 2, d.x + d.vx));
        d.y = Math.min(H - M / 2, Math.max(M / 2, d.y + d.vy));
      });
    }

    var animando = 0, temp = null;
    function anima(ciclos) {
      cancelAnimationFrame(temp);
      animando = ciclos;
      (function cuadro() {
        if (animando-- <= 0) { acomodaEtiquetas(); return; }
        paso(0.82); pinta();
        temp = requestAnimationFrame(cuadro);
      })();
    }

    // ── Arrastrar ────────────────────────────────────────────────────────────
    var cogido = null;
    function aLienzo(ev) {
      var caja = s.getBoundingClientRect();
      return { x: (ev.clientX - caja.left) * W / caja.width, y: (ev.clientY - caja.top) * H / caja.height };
    }
    circulos.forEach(function (c, i) {
      c.addEventListener('pointerdown', function (ev) {
        ev.preventDefault();
        cogido = i; nodos[i].fijo = true;
        c.setPointerCapture(ev.pointerId);
        c.style.cursor = 'grabbing';
        anima(100000);
      });
      c.addEventListener('pointermove', function (ev) {
        if (cogido !== i) return;
        var p = aLienzo(ev);
        nodos[i].x = p.x; nodos[i].y = p.y;
        pinta();
      });
      function suelta(ev) {
        if (cogido !== i) return;
        cogido = null; nodos[i].fijo = false;
        c.style.cursor = 'grab';
        try { c.releasePointerCapture(ev.pointerId); } catch (e) {}
        anima(140);
      }
      c.addEventListener('pointerup', suelta);
      c.addEventListener('pointercancel', suelta);

      // Resaltar vecindad al pasar el cursor.
      c.addEventListener('pointerenter', function () { if (cogido === null) resalta(i); });
      c.addEventListener('pointerleave', function () { if (cogido === null) resalta(null); });
    });

    function resalta(i) {
      nodos.forEach(function (d, j) {
        var dentro = i === null || j === i || vecinos[i][j];
        circulos[j].setAttribute('opacity', dentro ? 1 : 0.18);
        etiquetas[j].setAttribute('opacity', dentro ? 1 : 0.15);
      });
      aristas.forEach(function (e, k) {
        var dentro = i === null || e.o === i || e.d === i;
        lineas[k].setAttribute('stroke-opacity', dentro ? (i === null ? 0.75 : 0.95) : 0.08);
        lineas[k].setAttribute('stroke', dentro && i !== null ? '#8A9099' : '#C7CCD4');
      });
    }

    // ── Filtro por clúster ───────────────────────────────────────────────────
    var filtro = null;
    function aplicaFiltro() {
      nodos.forEach(function (d, j) {
        var dentro = filtro === null || d.c === filtro;
        circulos[j].setAttribute('opacity', dentro ? 1 : 0.12);
        etiquetas[j].setAttribute('opacity', dentro ? 1 : 0.1);
      });
      aristas.forEach(function (e, k) {
        var dentro = filtro === null || (nodos[e.o].c === filtro && nodos[e.d].c === filtro);
        lineas[k].setAttribute('stroke-opacity', dentro ? 0.75 : 0.05);
      });
    }

    cont.appendChild(s);

    // Leyenda interactiva: cada entrada enciende y apaga su clúster.
    var barra = document.createElement('div');
    barra.className = 'obs-red-leyenda';
    Object.keys(R.clusters).sort().forEach(function (k) {
      var c = parseInt(k, 10);
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'obs-red-chip';
      b.setAttribute('aria-pressed', 'false');
      b.innerHTML = '<span class="punto" style="background:' + (COLOR[c] || GRIS) + '"></span>' + k + '. ' + R.clusters[k];
      b.addEventListener('click', function () {
        filtro = (filtro === c) ? null : c;
        Array.prototype.forEach.call(barra.children, function (otro) {
          otro.setAttribute('aria-pressed', String(otro === b && filtro !== null));
        });
        aplicaFiltro();
      });
      barra.appendChild(b);
    });
    var reinicia = document.createElement('button');
    reinicia.type = 'button';
    reinicia.className = 'obs-red-chip obs-red-reinicia';
    reinicia.textContent = 'Reiniciar disposición';
    reinicia.addEventListener('click', function () {
      listaN.forEach(function (d, i) {
        nodos[i].x = M + d.x * (W - 2 * M);
        nodos[i].y = M + (1 - d.y) * (H - 2 * M);
        nodos[i].vx = 0; nodos[i].vy = 0;
      });
      filtro = null;
      Array.prototype.forEach.call(barra.children, function (o) { o.setAttribute('aria-pressed', 'false'); });
      aplicaFiltro(); pinta(); anima(60);
    });
    barra.appendChild(reinicia);
    cont.appendChild(barra);

    pinta();
    anima(170); // acomoda la disposición al lienzo sin desarmar la del análisis
  }

  /* ───────── Mapa de áreas (treemap) ─────────
     Rectángulos proporcionales al número de documentos de cada país, colocados
     con el algoritmo «squarified» de Bruls, Huizing y Van Wijk (2000), que busca
     proporciones cercanas al cuadrado para que las áreas sean comparables. */
  function squarify(datos, x, y, w, h, salida) {
    if (!datos.length) return salida;
    var total = datos.reduce(function (a, d) { return a + d.n; }, 0);
    var corto = Math.min(w, h);
    var escala = (w * h) / total;
    var fila = [], resto = datos.slice(), mejor = Infinity;

    function proporcion(fila, lado) {
      var suma = fila.reduce(function (a, d) { return a + d.n; }, 0) * escala;
      var max = Math.max.apply(null, fila.map(function (d) { return d.n * escala; }));
      var min = Math.min.apply(null, fila.map(function (d) { return d.n * escala; }));
      return Math.max(lado * lado * max / (suma * suma), (suma * suma) / (lado * lado * min));
    }

    while (resto.length) {
      var prueba = fila.concat([resto[0]]);
      var p = proporcion(prueba, corto);
      if (fila.length && p > mejor) break;
      fila = prueba; mejor = p; resto.shift();
    }

    var areaFila = fila.reduce(function (a, d) { return a + d.n; }, 0) * escala;
    var grueso = areaFila / corto;
    var pos = 0;
    fila.forEach(function (d) {
      var largo = (d.n * escala) / grueso;
      if (w >= h) salida.push({ d: d, x: x, y: y + pos, w: grueso, h: largo });
      else salida.push({ d: d, x: x + pos, y: y, w: largo, h: grueso });
      pos += largo;
    });

    if (w >= h) return squarify(resto, x + grueso, y, w - grueso, h, salida);
    return squarify(resto, x, y + grueso, w, h - grueso, salida);
  }

  function mapaAreas(cont, datos, opts) {
    opts = opts || {};
    if (!cont) return;
    var W = opts.ancho || 560, H = opts.alto || 360;
    var s = svgBase(W, H);
    var orden = datos.slice().sort(function (a, b) { return b.n - a.n; });
    var cajas = squarify(orden, 0, 0, W, H, []);
    var total = orden.reduce(function (a, d) { return a + d.n; }, 0);

    cajas.forEach(function (c) {
      var fondo = c.d.agregado ? '#E5E7EB' : (c.d.latam ? NARANJA : AZUL);
      var tinta = c.d.agregado ? '#374151' : '#FFFFFF';
      var r = el('rect', {
        x: c.x, y: c.y, width: Math.max(0, c.w - 1.5), height: Math.max(0, c.h - 1.5),
        fill: fondo, stroke: '#FFFFFF', 'stroke-width': 1
      });
      r.appendChild(el('title', {}, c.d.pais + ': ' + fmt(c.d.n) + ' documentos ('
        + String((100 * c.d.n / total).toFixed(1)).replace('.', ',') + ' %)'));
      s.appendChild(r);

      // La etiqueta solo se escribe si cabe dentro del rectángulo; si no, queda
      // el nombre en la información emergente.
      var fs = Math.max(9, Math.min(17, Math.sqrt(c.w * c.h) / 5.2));
      if (c.w < 42 || c.h < fs + 10) return;
      var nombre = c.d.pais, tope = Math.floor((c.w - 12) / (fs * 0.54));
      if (nombre.length > tope) nombre = nombre.slice(0, Math.max(2, tope - 1)) + '…';
      var t1 = el('text', { x: c.x + 7, y: c.y + fs + 5, 'font-size': fs, 'font-weight': 600,
                            style: 'fill:' + tinta }, nombre);
      s.appendChild(t1);
      if (c.h > fs * 2 + 12) {
        s.appendChild(el('text', { x: c.x + 7, y: c.y + fs * 2 + 8, 'font-size': Math.max(9, fs - 3),
                                   style: 'fill:' + tinta + ';fill-opacity:.82' }, fmt(c.d.n)));
      }
    });
    cont.appendChild(s);

    // Los países latinoamericanos ocupan rectángulos demasiado pequeños para
    // llevar etiqueta, así que se enumeran debajo.
    var lat = orden.filter(function (d) { return d.latam; });
    if (lat.length) {
      var pie = document.createElement('p');
      pie.className = 'obs-nota';
      pie.style.marginTop = '8px';
      pie.innerHTML = '<b style="color:' + NARANJA + '">Latinoamérica:</b> '
        + lat.map(function (d) { return d.pais + ' (' + fmt(d.n) + ')'; }).join(' · ');
      cont.appendChild(pie);
    }
  }

  /* ───────── Nube de palabras clave ─────────
     Tamaño proporcional a la raíz cuadrada de la frecuencia (el área, y no la
     altura, representa el valor) y color según el clúster temático. El texto va
     siempre horizontal: una nube con palabras giradas se ve pero no se lee. */
  function nubePalabras(cont, datos, opts) {
    opts = opts || {};
    if (!cont) return;
    var W = opts.ancho || 560, H = opts.alto || 340;
    var s = svgBase(W, H);
    var orden = datos.slice().sort(function (a, b) { return b.n - a.n; }).slice(0, opts.tope || 45);
    var maxN = orden[0].n, minN = orden[orden.length - 1].n;
    var fsMax = opts.fsMax || 34, fsMin = opts.fsMin || 10;
    var COLOR = [AZUL, NARANJA, '#3F6FB5', GRIS, '#7A9CD0'];
    var puestos = [], g = el('g', {});

    function choca(a) {
      for (var i = 0; i < puestos.length; i++) {
        var b = puestos[i];
        if (a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y) return true;
      }
      return false;
    }

    orden.forEach(function (d) {
      var t = maxN === minN ? 1 : (Math.sqrt(d.n) - Math.sqrt(minN)) / (Math.sqrt(maxN) - Math.sqrt(minN));
      var fs = fsMin + t * (fsMax - fsMin);
      var w = d.p.length * fs * 0.5 + 10, h = fs * 1.22;
      // Espiral de Arquímedes desde el centro hasta encontrar hueco.
      var cx = W / 2, cy = H / 2, caja = null;
      for (var paso = 0; paso < 1800; paso++) {
        var ang = paso * 0.32, r = 2.2 * ang;
        var x = cx + r * Math.cos(ang) - w / 2, y = cy + r * Math.sin(ang) * 0.58 - h / 2;
        if (x < 2 || y < 2 || x + w > W - 2 || y + h > H - 2) continue;
        var c = { x: x, y: y, w: w, h: h };
        if (!choca(c)) { caja = c; break; }
      }
      if (!caja) return;
      puestos.push(caja);
      var te = el('text', {
        x: caja.x + w / 2, y: caja.y + h * 0.78, 'text-anchor': 'middle',
        'font-size': fs, 'font-weight': fs > 18 ? 600 : 500,
        style: 'fill:' + COLOR[(d.c || 0) % COLOR.length]
      }, d.p);
      te.appendChild(el('title', {}, d.p + ': ' + fmt(d.n) + ' documentos'));
      g.appendChild(te);
    });

    // Las palabras se colocan desde el centro hacia fuera, así que el conjunto
    // suele quedar descentrado: se desplaza para que ocupe el lienzo con holgura pareja.
    if (puestos.length) {
      var x0 = Math.min.apply(null, puestos.map(function (c) { return c.x; }));
      var x1 = Math.max.apply(null, puestos.map(function (c) { return c.x + c.w; }));
      var y0 = Math.min.apply(null, puestos.map(function (c) { return c.y; }));
      var y1 = Math.max.apply(null, puestos.map(function (c) { return c.y + c.h; }));
      g.setAttribute('transform', 'translate(' + ((W - (x1 - x0)) / 2 - x0).toFixed(1)
        + ',' + ((H - (y1 - y0)) / 2 - y0).toFixed(1) + ')');
    }
    s.appendChild(g);
    cont.appendChild(s);
  }

  /* Ancho al que conviene dibujar: el que ocupa el contenedor en pantalla.
     Así el SVG no se estira y las etiquetas mantienen su tamaño real. */
  function anchoDe(cont, minimo) {
    var w = cont && cont.clientWidth ? cont.clientWidth : 560;
    return Math.max(minimo || 320, Math.round(w));
  }
  function svgBase(w, h) {
    var s = el('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img' });
    return s;
  }
  /* Devuelve un paso de rejilla redondo y el máximo del eje, de modo que las
     marcas caigan siempre en valores legibles (0, 10, 20…) y no en decimales. */
  function escala(max) {
    if (max <= 0) return { max: 1, paso: 1 };
    var bruto = max / 4;
    var exp = Math.pow(10, Math.floor(Math.log10(bruto))), f = bruto / exp;
    var m = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
    var paso = m * exp;
    return { max: Math.ceil(max / paso) * paso, paso: paso };
  }
  function fmt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }
  /* Porcentaje con coma decimal, como corresponde en español. */
  function pct(n) { return String(n).replace('.', ',') + ' %'; }

  /* ── Gráfico de líneas ── */
  function lineas(cont, anios, series, opts) {
    opts = opts || {};
    var W = opts.ancho || 560, H = opts.alto || 300, mL = 48, mR = 14, mT = 14, mB = 38;
    var s = svgBase(W, H);
    var max = 0;
    series.forEach(function (se) { se.v.forEach(function (v) { if (v > max) max = v; }); });
    var esc = escala(max); max = esc.max;
    var x = function (i) { return mL + i * (W - mL - mR) / Math.max(1, anios.length - 1); };
    var y = function (v) { return H - mB - (v / max) * (H - mT - mB); };
    for (var v = 0; v <= max + 1e-9; v += esc.paso) {
      s.appendChild(el('line', { x1: mL, x2: W - mR, y1: y(v), y2: y(v), stroke: LINEA, 'stroke-width': 0.8 }));
      s.appendChild(el('text', { x: mL - 8, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 11 }, opts.pct ? v + '%' : fmt(Math.round(v))));
    }
    anios.forEach(function (a, i) {
      s.appendChild(el('text', { x: x(i), y: H - mB + 18, 'text-anchor': 'middle', 'font-size': 11 }, a));
    });
    s.appendChild(el('line', { x1: mL, x2: W - mR, y1: H - mB, y2: H - mB, stroke: GRIS, 'stroke-width': 1 }));
    series.forEach(function (se, k) {
      var d = se.v.map(function (v, i) { return (i ? 'L' : 'M') + x(i) + ' ' + y(v); }).join(' ');
      s.appendChild(el('path', { d: d, fill: 'none', stroke: se.color || SERIE[k % SERIE.length], 'stroke-width': 2, 'stroke-linejoin': 'round' }));
      se.v.forEach(function (v, i) {
        var c = el('circle', { cx: x(i), cy: y(v), r: 3.5, fill: se.color || SERIE[k % SERIE.length] });
        c.appendChild(el('title', {}, se.n + ' · ' + anios[i] + ': ' + (opts.pct ? v + ' %' : fmt(v))));
        s.appendChild(c);
      });
    });
    cont.appendChild(s);
  }

  /* ── Barras horizontales ── */
  function barras(cont, cats, vals, opts) {
    opts = opts || {};
    var n = cats.length, fila = opts.fila || 22, mL = opts.mL || 180, mR = 54, mT = 6, mB = 26;
    var W = opts.ancho || 560, H = mT + mB + n * fila;
    var fs = W < 520 ? 10 : 11.5;
    var recorta = [];
    var s = svgBase(W, H);
    var esc = escala(Math.max.apply(null, vals)), max = esc.max;
    var x = function (v) { return mL + (v / max) * (W - mL - mR); };
    for (var v = 0; v <= max + 1e-9; v += esc.paso) {
      s.appendChild(el('line', { x1: x(v), x2: x(v), y1: mT, y2: H - mB, stroke: LINEA, 'stroke-width': 0.8 }));
      s.appendChild(el('text', { x: x(v), y: H - mB + 16, 'text-anchor': 'middle', 'font-size': 11 }, fmt(Math.round(v))));
    }
    cats.forEach(function (c, i) {
      var yy = mT + i * fila, col = (opts.colores && opts.colores[i]) ? opts.colores[i] : AZUL;
      var r = el('rect', { x: mL, y: yy + 3, width: Math.max(1, x(vals[i]) - mL), height: fila - 8, fill: col });
      r.appendChild(el('title', {}, c + ': ' + fmt(vals[i])));
      s.appendChild(r);
      var te = el('text', { x: mL - 8, y: yy + fila / 2 + 3, 'text-anchor': 'end', 'font-size': fs }, c);
      te.appendChild(el('title', {}, c));
      s.appendChild(te);
      recorta.push(te);
      s.appendChild(el('text', { x: x(vals[i]) + 6, y: yy + fila / 2 + 3, 'font-size': 11, fill: TINTA }, fmt(vals[i])));
    });
    s.appendChild(el('line', { x1: mL, x2: mL, y1: mT, y2: H - mB, stroke: GRIS, 'stroke-width': 1 }));
    cont.appendChild(s);
    // Una vez en el documento se puede medir el texto: las etiquetas que no caben
    // en el margen izquierdo se recortan carácter a carácter hasta que caben.
    var ancho = mL - 10;
    recorta.forEach(function (t) {
      if (t.getComputedTextLength() <= ancho) return;
      var txt = t.textContent;
      while (txt.length > 2 && t.getComputedTextLength() > ancho) {
        txt = txt.slice(0, -1);
        t.textContent = txt + '…';
      }
    });
  }

  function leyenda(cont, items) {
    var d = document.createElement('div');
    d.className = 'obs-leyenda';
    items.forEach(function (it) {
      var s = document.createElement('span');
      s.innerHTML = '<i style="background:' + it.c + '"></i>' + it.t;
      d.appendChild(s);
    });
    cont.appendChild(d);
  }

  /* ── Tabla de estudios ── */
  function tabla() {
    var datos = D.estudios.slice(), orden = { col: 'n', dir: 1 };
    var tb = document.getElementById('obs-tbody');
    var cont = document.getElementById('obs-conteo');
    var fA = document.getElementById('f-anio'), fP = document.getElementById('f-pais');
    var fC = document.getElementById('f-cita'), fQ = document.getElementById('f-busca');

    [['anio', fA, 'Todos los años'], ['pais', fP, 'Todos los países']].forEach(function (par) {
      var vals = [], k = par[0];
      datos.forEach(function (r) { if (vals.indexOf(r[k]) < 0 && r[k] !== '') vals.push(r[k]); });
      vals.sort();
      vals.forEach(function (v) {
        var o = document.createElement('option'); o.value = v; o.textContent = v; par[1].appendChild(o);
      });
    });

    function filtrados() {
      var q = (fQ.value || '').toLowerCase().trim();
      return datos.filter(function (r) {
        if (fA.value && String(r.anio) !== fA.value) return false;
        if (fP.value && r.pais !== fP.value) return false;
        if (fC.value === 'si' && String(r.cita_fuentes).indexOf('Sí') !== 0) return false;
        if (fC.value === 'no' && String(r.cita_fuentes).indexOf('Sí') === 0) return false;
        if (!q) return true;
        return (r.cita + ' ' + r.titulo + ' ' + r.llm + ' ' + r.embeddings + ' ' + r.base_vectorial + ' ' +
                r.corpus + ' ' + r.metricas + ' ' + r.pais).toLowerCase().indexOf(q) >= 0;
      });
    }
    function pinta() {
      var f = filtrados();
      f.sort(function (a, b) {
        var x = a[orden.col], y = b[orden.col];
        if (typeof x === 'number' && typeof y === 'number') return (x - y) * orden.dir;
        return String(x).localeCompare(String(y), 'es') * orden.dir;
      });
      tb.innerHTML = '';
      f.forEach(function (r) {
        var tr = document.createElement('tr');
        tr.innerHTML = '<td>' + r.n + '</td>' +
          '<td><button class="obs-ver" type="button">' + r.cita + '</button></td>' +
          '<td class="num">' + r.anio + '</td>' +
          '<td>' + r.pais + '</td>' +
          '<td>' + r.llm + '</td>' +
          '<td>' + r.base_vectorial + '</td>' +
          '<td>' + (String(r.cita_fuentes).indexOf('Sí') === 0 ? 'Sí' : 'No reportado') + '</td>' +
          '<td class="num">' + r.calidad_total + ' / 8</td>';
        tb.appendChild(tr);
        var det = document.createElement('tr');
        det.className = 'obs-detalle'; det.style.display = 'none';
        var doi = r.doi ? '<dt>DOI</dt><dd><a href="https://doi.org/' + r.doi + '" target="_blank" rel="noopener">' + r.doi + '</a></dd>' : '';
        det.innerHTML = '<td colspan="8"><dl>' +
          '<dt>Título</dt><dd>' + r.titulo + '</dd>' + doi +
          '<dt>Corpus</dt><dd>' + r.corpus + '</dd>' +
          '<dt>Modelo de lenguaje</dt><dd>' + r.llm + '</dd>' +
          '<dt>Embeddings</dt><dd>' + r.embeddings + '</dd>' +
          '<dt>Almacén</dt><dd>' + r.base_vectorial + '</dd>' +
          '<dt>Fragmentación</dt><dd>' + r.chunking + '</dd>' +
          '<dt>Métricas</dt><dd>' + r.metricas + '</dd>' +
          '<dt>Resultados</dt><dd>' + r.resultados + '</dd>' +
          '<dt>Manejo de alucinaciones</dt><dd>' + r.manejo_alucinacion + '</dd>' +
          '<dt>Limitaciones</dt><dd>' + r.limitaciones + '</dd>' +
          '<dt>Base de datos</dt><dd>' + r.fuente_bd + '</dd>' +
          '</dl></td>';
        tb.appendChild(det);
        tr.querySelector('.obs-ver').addEventListener('click', function () {
          det.style.display = det.style.display === 'none' ? '' : 'none';
        });
      });
      cont.textContent = f.length + ' de ' + datos.length + ' estudios';
    }
    [fA, fP, fC].forEach(function (s) { s.addEventListener('change', pinta); });
    fQ.addEventListener('input', pinta);
    Array.prototype.forEach.call(document.querySelectorAll('#obs-tabla thead th[data-col]'), function (th) {
      th.addEventListener('click', function () {
        var c = th.getAttribute('data-col');
        orden.dir = (orden.col === c) ? -orden.dir : 1;
        orden.col = c;
        Array.prototype.forEach.call(document.querySelectorAll('#obs-tabla thead th'), function (o) { o.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', orden.dir === 1 ? 'ascending' : 'descending');
        pinta();
      });
    });
    pinta();
  }

  function texto(id, v) { var e = document.getElementById(id); if (e) e.textContent = v; }

  function pinta() {
    var I = D.ind, R = D.rsl, P = D.prisma, C = D.cob;
    texto('c-nucleo', fmt(I.n_AB));
    texto('c-estudios', R.n);
    texto('c-latam', I.n_latam_nucleo_docs);
    texto('c-nicho', I.n_ABC);
    texto('t-registros', fmt(P.AB_total_registros));
    texto('t-candidatos', fmt(P.candidatos_cribado_titulo_resumen));
    texto('t-elegibles', P.elegibles_cumplen_criterios);
    texto('t-incluidos', P.incluidos);
    texto('t-colombia', I.n_colombia_nucleo);
    var ec = document.getElementById('t-colombia-s'); if (ec) ec.textContent = I.n_colombia_nucleo === 1 ? 'documento' : 'documentos';
    texto('t-ac', I.n_AC);
    texto('t-catastro', I.n_catastro_AC);
    texto('t-cobertura', pct(C.pct_cobertura_openalex));
    texto('t-citas', R.cita_fuentes_si);
    texto('t-fidelidad', R.met_fidelidad);
    texto('t-aluc', R.aluc_mecanismo);
    texto('t-abierto', R.llm_abierto);
    texto('t-sinpais', pct(I.pct_sin_pais_nucleo));
    texto('t-sinpais2', pct(I.pct_sin_pais_nucleo));
    texto('t-sinfuente', pct(I.pct_sin_fuente_nucleo));

    dibujaGraficos();
    tabla();
  }


  /* Vacía un contenedor y lo devuelve, para poder redibujar sin duplicar. */
  function limpia(id) {
    var c = document.getElementById(id);
    if (c) while (c.firstChild) c.removeChild(c.firstChild);
    return c;
  }

  /* Dibuja los seis gráficos al ancho que ocupan en pantalla. Se vuelve a llamar
     al cambiar el tamaño de la ventana para que el texto no se estire. */
  function dibujaGraficos() {
    var a = D.anual, cont;
    cont = limpia('g-anual');
    leyenda(cont, [{ c: AZUL, t: 'Núcleo A∩B (RAG/LLM + corpus académico)' }, { c: GRIS, t: 'A∩C (RAG/LLM + geociencias)' }, { c: NARANJA, t: 'Nicho A∩B∩C' }]);
    lineas(cont, a.anios, [{ n: 'A∩B', v: a.AB, color: AZUL }, { n: 'A∩C', v: a.AC, color: GRIS }, { n: 'A∩B∩C', v: a.ABC, color: NARANJA }], { ancho: anchoDe(cont) });

    cont = limpia('g-paises');
    barras(cont, D.paises.cat, D.paises.n, { ancho: anchoDe(cont), colores: D.paises.latam.map(function (b) { return b ? NARANJA : AZUL; }) });

    cont = limpia('g-latam');
    barras(cont, D.latam.cat, D.latam.n, { ancho: anchoDe(cont), colores: D.latam.cat.map(function () { return NARANJA; }), mL: 120 });

    cont = limpia('g-fuentes');
    barras(cont, D.fuentes.cat, D.fuentes.n, { ancho: anchoDe(cont), mL: 230 });

    cont = limpia('g-tend');
    leyenda(cont, D.tend.series.map(function (s, i) { return { c: SERIE[i % SERIE.length], t: s.nombre }; }));
    lineas(cont, D.tend.anios, D.tend.series.map(function (s) { return { n: s.nombre, v: s.pct }; }), { pct: true, ancho: anchoDe(cont), alto: 340 });

    cont = limpia('g-mapa');
    if (cont) mapaAreas(cont, D.paises_mapa, { ancho: anchoDe(cont), alto: 400 });

    cont = limpia('g-red');
    if (cont) {
      var anchoRed = anchoDe(cont);
      redInteractiva(cont, D.red, {
        ancho: anchoRed,
        alto: anchoRed < 620 ? 540 : 620,
        tope: anchoRed < 520 ? 16 : (anchoRed < 820 ? 26 : 38)
      });
    }

    cont = limpia('g-nube');
    if (cont) nubePalabras(cont, D.nube, { ancho: anchoDe(cont), alto: 360, fsMax: anchoDe(cont) < 620 ? 24 : 34 });

    cont = limpia('g-comp');
    barras(cont, D.componentes.map(function (c) { return c.k; }), D.componentes.map(function (c) { return c.v; }), { ancho: anchoDe(cont), mL: Math.min(260, anchoDe(cont) * 0.42), fila: 26 });

  }

  /* Redibuja al cambiar el ancho de la ventana, con una espera para no
     recalcular en cada píxel del arrastre. */
  var temporizador = null;
  window.addEventListener('resize', function () {
    if (!D) return;
    clearTimeout(temporizador);
    temporizador = setTimeout(dibujaGraficos, 200);
  });

  /* Deja constancia visible del fallo: en el aviso de la página y dentro de cada
     recuadro de figura, para que el problema no se confunda con una página vacía. */
  function avisa(mensaje) {
    console.error('[observatorio] ' + mensaje);
    var av = document.getElementById('obs-aviso');
    if (av) { av.style.display = 'block'; av.textContent = mensaje; }
    var cajas = document.querySelectorAll('.obs-fig');
    for (var i = 0; i < cajas.length; i++) {
      if (cajas[i].querySelector('svg')) continue;
      var p = document.createElement('p');
      p.className = 'obs-nota';
      p.textContent = 'Gráfico no disponible: no se cargaron los datos.';
      cajas[i].appendChild(p);
    }
  }

  function arranca() {
    var ruta = '../data/observatorio/observatorio.json';
    fetch(ruta)
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (j) {
        D = j;
        try { pinta(); }
        catch (e) { avisa('Error al dibujar el observatorio: ' + e.message); }
      })
      .catch(function (e) {
        avisa('No fue posible cargar ' + ruta + ' (' + e.message + '). Comprueba que la carpeta data/observatorio/ esté en el repositorio y que la página se abra desde un servidor, no con doble clic. Los archivos CSV siguen disponibles en la sección de descargas.');
      });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', arranca); else arranca();
})();
