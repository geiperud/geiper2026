/* ============================================
   GEIPER - Panel de registro (temporal)
   Config: ajusta las fechas de campaña y el
   link del formulario/iframe aquí abajo.
   Para desactivar el panel: pon CAMPAIGN_ENABLED
   en false, o simplemente quita las 2 líneas
   (css/js) que lo cargan en cada página.
   ============================================ */

(function () {
  var CAMPAIGN_ENABLED = true;

  // Rango de fechas de la campaña (formato YYYY-MM-DD).
  // Si hoy está fuera de este rango, el panel no aparece.
  var CAMPAIGN_START = '2026-08-22';
  var CAMPAIGN_END   = '2026-09-30';

  var FORM_IFRAME_SRC = 'https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=74gT1bBqY0OflNVmRKRZcHVPAxNIpkBCmaAAujrqeu1UQU1RN1BCN0pXSTVXRTFCRDFJQjc3WjRETi4u&embed=true';

  var PANEL_TITLE = 'Regístrate a la sesión';
  var PANEL_DESC  = 'Te enviaremos el link de conexión por correo.';
  var BADGE_TEXT  = 'Inscripciones abiertas';

  var SHOW_DELAY_MS = 1500;
  var SESSION_KEY = 'geiperRegistroPanelDismissed';

  function withinCampaignDates() {
    var today = new Date().toISOString().slice(0, 10);
    return today >= CAMPAIGN_START && today <= CAMPAIGN_END;
  }

  function init() {
    if (!CAMPAIGN_ENABLED || !withinCampaignDates()) return;

    var panel = document.createElement('div');
    panel.id = 'registroPanel';
    panel.setAttribute('role', 'complementary');
    panel.setAttribute('aria-label', 'Registro a la sesión');
    panel.innerHTML =
      '<div class="rp-header">' +
        '<span>' + BADGE_TEXT + '</span>' +
        '<button class="rp-close" id="registroPanelClose" aria-label="Cerrar">&times;</button>' +
      '</div>' +
      '<div class="rp-body">' +
        '<p class="rp-title">' + PANEL_TITLE + '</p>' +
        '<p class="rp-desc">' + PANEL_DESC + '</p>' +
        '<div class="rp-frame-wrap">' +
          '<iframe src="' + FORM_IFRAME_SRC + '" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen webkitallowfullscreen mozallowfullscreen msallowfullscreen title="Formulario de registro GEIPER"></iframe>' +
        '</div>' +
      '</div>';

    var reopenBtn = document.createElement('button');
    reopenBtn.id = 'registroPanelReopen';
    reopenBtn.type = 'button';
    reopenBtn.innerHTML = '<i class="fa-solid fa-user-plus"></i> Registrarme';

    document.body.appendChild(panel);
    document.body.appendChild(reopenBtn);

    var closeBtn = document.getElementById('registroPanelClose');

    function showPanel() {
      panel.classList.add('rp-visible');
      reopenBtn.classList.remove('rp-reopen-visible');
    }

    function hidePanel(remember) {
      panel.classList.remove('rp-visible');
      reopenBtn.classList.add('rp-reopen-visible');
      if (remember) {
        try { sessionStorage.setItem(SESSION_KEY, '1'); } catch (_) {}
      }
    }

    closeBtn.addEventListener('click', function () { hidePanel(true); });
    reopenBtn.addEventListener('click', showPanel);

    var alreadyDismissed = false;
    try { alreadyDismissed = sessionStorage.getItem(SESSION_KEY) === '1'; } catch (_) {}

    if (alreadyDismissed) {
      reopenBtn.classList.add('rp-reopen-visible');
    } else {
      setTimeout(showPanel, SHOW_DELAY_MS);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
