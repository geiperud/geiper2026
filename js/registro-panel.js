/* ============================================
   GEIPER - Popup de registro (temporal)
   Config: ajusta fechas de campaña, textos y el
   link del formulario aquí abajo.
   Para desactivar el popup: pon CAMPAIGN_ENABLED
   en false, o quita las 2 líneas (css/js) que lo
   cargan en cada página.
   ============================================ */

(function () {
  var CAMPAIGN_ENABLED = true;

  // Rango de fechas de la campaña (formato YYYY-MM-DD).
  var CAMPAIGN_START = '2026-08-22';
  var CAMPAIGN_END   = '2026-09-30';

  // Link directo al formulario (se abre en pestaña nueva; evita el
  // bloqueo de cookies de terceros que rompe el embebido por iframe).
  var FORM_URL = 'https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=74gT1bBqY0OflNVmRKRZcHVPAxNIpkBCmaAAujrqeu1UQU1RN1BCN0pXSTVXRTFCRDFJQjc3WjRETi4u';

  var EYEBROW_TEXT = 'Registro a reunión informativa 2026-III';
  var TITLE_TEXT   = 'GEIPER';
  var DESC_TEXT    = 'Cupos limitados. El link de conexión llega directo a tu correo.';
  var CTA_TEXT     = 'Abrir formulario de registro';

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
        '<div class="rp-seal"><img src="' + logoSrc + '" alt="GEIPER"></div>' +
        '<div class="rp-card">' +
          '<button class="rp-close" id="registroPopupClose" aria-label="Cerrar">&times;</button>' +
          '<p class="rp-eyebrow">' + EYEBROW_TEXT + '</p>' +
          '<p class="rp-title">' + TITLE_TEXT + '</p>' +
          '<div class="rp-divider"></div>' +
          '<p class="rp-desc">' + DESC_TEXT + '</p>' +
          '<a class="rp-cta" href="' + FORM_URL + '" target="_blank" rel="noopener">' + CTA_TEXT + '</a>' +
        '</div>' +
      '</div>';

    var pill = document.createElement('button');
    pill.id = 'registroPill';
    pill.type = 'button';
    pill.innerHTML = '<i class="fa-solid fa-user-plus"></i> Registrarme';

    document.body.appendChild(overlay);
    document.body.appendChild(pill);

    var closeBtn = document.getElementById('registroPopupClose');

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
