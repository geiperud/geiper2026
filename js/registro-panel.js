/* ============================================
   GEIPER - Popup de registro (temporal)
   Config: ajusta fechas de campaña, datos de la
   reunión y el link del formulario aquí abajo.
   Para desactivar el popup: pon CAMPAIGN_ENABLED
   en false, o quita las 2 líneas (css/js) que lo
   cargan en cada página.
   ============================================ */

(function () {
  var CAMPAIGN_ENABLED = true;

  // Rango de fechas de la campaña (formato YYYY-MM-DD).
  // Si hoy está fuera de este rango, el popup no aparece.
  var CAMPAIGN_START = '2026-08-22';
  var CAMPAIGN_END   = '2026-09-04';

  // Link directo al formulario (se abre en pestaña nueva; evita el
  // bloqueo de cookies de terceros que rompe el embebido por iframe).
  var FORM_URL = 'https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=74gT1bBqY0OflNVmRKRZcHVPAxNIpkBCmaAAujrqeu1UQU1RN1BCN0pXSTVXRTFCRDFJQjc3WjRETi4u';

  var SIDE_EYEBROW = 'Semillero de investigación';
  var SIDE_TITLE   = 'GEIPER';

  var MAIN_EYEBROW = 'Registro abierto';
  var MAIN_TITLE   = 'Reunión informativa 2026-III';

  var MEETING_DATE = 'Viernes 4 de septiembre · 10:00 a.m.';
  var MEETING_SUB  = 'Virtual · el link llega a tu correo';

  var DESC_TEXT = 'Presentamos las líneas de investigación, los proyectos en curso y el proceso de vinculación al semillero. Cupos limitados.';

  var CTA_TEXT     = 'Abrir formulario';
  var DISMISS_TEXT = 'Ahora no';

  var SHOW_DELAY_MS = 1500;
  var SESSION_KEY = 'geiperRegistroPopupDismissed';

  function withinCampaignDates() {
    var today = new Date().toISOString().slice(0, 10);
    return today >= CAMPAIGN_START && today <= CAMPAIGN_END;
  }

  function init() {
    if (!CAMPAIGN_ENABLED || !withinCampaignDates()) return;

    var basePath = location.pathname.indexOf('/pages/') !== -1 ? '../' : '';
    var logoSrc = basePath + 'assets/images/Logo_GEIPER.png';

    var overlay = document.createElement('div');
    overlay.id = 'registroOverlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Registro a reunión informativa');
    overlay.innerHTML =
      '<div id="registroModal">' +
        '<div class="rp-card">' +
          '<div class="rp-side">' +
            '<div class="rp-logo-circle"><img src="' + logoSrc + '" alt="GEIPER"></div>' +
            '<div>' +
              '<p class="rp-side-eyebrow">' + SIDE_EYEBROW + '</p>' +
              '<p class="rp-side-title">' + SIDE_TITLE + '</p>' +
            '</div>' +
          '</div>' +
          '<div class="rp-main">' +
            '<button class="rp-close" id="registroPopupClose" aria-label="Cerrar">&times;</button>' +
            '<p class="rp-eyebrow">' + MAIN_EYEBROW + '</p>' +
            '<p class="rp-title">' + MAIN_TITLE + '</p>' +
            '<div class="rp-highlight">' +
              '<p class="rp-highlight-date">' + MEETING_DATE + '</p>' +
              '<p class="rp-highlight-sub">' + MEETING_SUB + '</p>' +
            '</div>' +
            '<p class="rp-desc">' + DESC_TEXT + '</p>' +
            '<div class="rp-actions">' +
              '<a class="rp-cta" href="' + FORM_URL + '" target="_blank" rel="noopener">' + CTA_TEXT + '</a>' +
              '<button type="button" class="rp-dismiss" id="registroDismissLink">' + DISMISS_TEXT + '</button>' +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>';

    var pill = document.createElement('button');
    pill.id = 'registroPill';
    pill.type = 'button';
    pill.innerHTML = '<i class="fa-solid fa-user-plus"></i> Registrarme';

    document.body.appendChild(overlay);
    document.body.appendChild(pill);

    var closeBtn = document.getElementById('registroPopupClose');
    var dismissLink = document.getElementById('registroDismissLink');

    function showPopup() {
      overlay.classList.add('rp-visible');
      pill.classList.remove('rp-pill-visible');
    }

    function hidePopup(remember) {
      overlay.classList.remove('rp-visible');
      pill.classList.add('rp-pill-visible');
      if (remember) {
        try { sessionStorage.setItem(SESSION_KEY, '1'); } catch (_) {}
      }
    }

    closeBtn.addEventListener('click', function () { hidePopup(true); });
    dismissLink.addEventListener('click', function () { hidePopup(true); });
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) hidePopup(true);
    });
    pill.addEventListener('click', showPopup);

    var alreadyDismissed = false;
    try { alreadyDismissed = sessionStorage.getItem(SESSION_KEY) === '1'; } catch (_) {}

    if (alreadyDismissed) {
      pill.classList.add('rp-pill-visible');
    } else {
      setTimeout(showPopup, SHOW_DELAY_MS);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
