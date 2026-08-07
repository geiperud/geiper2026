/**
 * GEIPER Chatbots Lógica de Interfaz y Conexión
 * Diseñado bajo principios ISO/IEC 25010 (Gestión de errores, usabilidad)
 */

document.addEventListener('DOMContentLoaded', () => {

  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const botOptions = document.querySelectorAll('.bot-option');
  const currentBotTitle = document.getElementById('currentBotTitle');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const serverStatus = document.getElementById('serverStatus');

  let currentMode = 'tematico'; // modos válidos: 'tematico', 'investigacion'
  // Memoria de conversacion: ventana deslizante de los ultimos intercambios.
  // Se reinicia al cambiar de asistente o al limpiar el chat.
  let historial = [];
  const MAX_TURNOS_HISTORIAL = 6; // 3 intercambios (usuario + bot)
  // Controla si ya se le ofrecio al usuario ver el listado de documentos
  // (se ofrece una sola vez, despues del primer intercambio de cada conversacion)
  let listadoDocumentosOfrecido = false;
  // URL del backend: usa variable global, o detecta producción vs local
  const API_BASE = window.GEIPER_API_URL ||
                   (window.location.hostname === 'geiperud.github.io'
                     ? 'https://geiper2026.onrender.com'
                     : 'http://localhost:8000');
  const API_ENDPOINT = API_BASE + '/chat';

  // Configuración de los bots (Saludo inicial y títulos)
  const botsConfig = {
    tematico: {
      title: '<i class="fa-solid fa-book-open-reader"></i> Asistente Temático',
      greeting: '¡Hola! Soy el <strong>Asistente Temático</strong> del semillero GEIPER. Puedo resolver dudas sobre nuestros documentos o buscar información en la web. ¿En qué te ayudo?'
    },
    investigacion: {
      title: '<i class="fa-solid fa-magnifying-glass-chart"></i> Asistente de Investigación',
      greeting: 'Saludos, soy el <strong>Asistente de Investigación</strong>. Conozco a fondo la estructura de la universidad y sus enfoques investigativos. ¿Qué deseas saber?'
    }
  };

  // ── Inicialización ──
  resetChat();
  checkServerConnection();

  // ── Eventos ──
  botOptions.forEach(opt => {
    opt.addEventListener('click', () => {
      botOptions.forEach(b => b.classList.remove('active'));
      opt.classList.add('active');

      const newMode = opt.dataset.bot;
      if (newMode !== currentMode) {
        currentMode = newMode;
        resetChat();
      }
    });
  });

  clearChatBtn.addEventListener('click', resetChat);

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = chatInput.value.trim();
    if (!prompt) return;

    addMessage(prompt, 'user');
    chatInput.value = '';

    setLoadingState(true);
    const typingId = addTypingIndicator();

    try {
      const responseText = await fetchFromBackend(prompt, currentMode);
      removeElement(typingId);
      addMessage(responseText, 'bot');

      // Registrar el intercambio en el historial (ventana deslizante).
      // Se limpia el HTML del pie de citas y se trunca, para no ensuciar
      // el contexto que se le manda al modelo ni exceder el limite del backend.
      const respuestaLimpia = responseText
        .replace(/<[^>]+>/g, '')
        .trim()
        .slice(0, 2000);
      historial.push({ role: 'user', content: prompt.slice(0, 2000) });
      historial.push({ role: 'assistant', content: respuestaLimpia });
      if (historial.length > MAX_TURNOS_HISTORIAL) {
        historial = historial.slice(-MAX_TURNOS_HISTORIAL);
      }

      // Despues del PRIMER intercambio de la conversacion, se ofrece (no se
      // impone) ver el listado de documentos -- solo se muestra si el usuario
      // lo pide con el boton.
      if (!listadoDocumentosOfrecido && historial.length === 2) {
        listadoDocumentosOfrecido = true;
        addOfferMessage();
      }
    } catch (error) {
      removeElement(typingId);
      addErrorMessage();
      console.error(error);
    } finally {
      setLoadingState(false);
    }
  });

  // ── Funciones de UI ──

  function resetChat() {
    chatMessages.innerHTML = '';
    historial = [];
    listadoDocumentosOfrecido = false;
    currentBotTitle.innerHTML = botsConfig[currentMode].title;
    addMessage(botsConfig[currentMode].greeting, 'bot', true);
  }

  function addOfferMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    const img = document.createElement('img');
    img.src = '../assets/images/Logo_GEIPER.png';
    img.alt = 'Bot';
    avatar.appendChild(img);
    msgDiv.appendChild(avatar);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const p = document.createElement('p');
    p.textContent = 'Por cierto, si te sirve, puedo mostrarte la lista de documentos a los que tengo acceso en esta conversación.';
    contentDiv.appendChild(p);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'offer-docs-btn';
    btn.textContent = 'Sí, muéstrame';
    btn.addEventListener('click', () => {
      btn.disabled = true;
      btn.textContent = 'Mostrando...';
      mostrarContextoDocumentos().then(() => {
        btn.remove();
      });
    });
    contentDiv.appendChild(btn);

    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function mostrarContextoDocumentos() {
    try {
      const res = await fetch(`${API_BASE}/documentos?modo=${currentMode}`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.documentos && data.documentos.length > 0) {
        const lista = data.documentos.join('\n');
        addMessage(
          `Ahora mismo tengo acceso a estos documentos en esta conversación:\n\n${lista}\n\n¿Sobre cuál te gustaría empezar?`,
          'bot'
        );
      }
    } catch (error) {
      // Si falla (servidor dormido, sin conexion), simplemente no se muestra
      // el contexto -- no se interrumpe el uso del chat por esto.
      console.warn('No se pudo cargar el contexto de documentos:', error);
    }
  }

  function addMessage(text, sender, isHtml = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;

    if (sender === 'bot') {
      const avatar = document.createElement('div');
      avatar.className = 'message-avatar';
      const img = document.createElement('img');
      img.src = '../assets/images/Logo_GEIPER.png';
      img.alt = 'Bot';
      avatar.appendChild(img);
      msgDiv.appendChild(avatar);
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isHtml) {
      // Solo se usa para saludos fijos definidos en botsConfig (texto propio, no del backend)
      const p = document.createElement('p');
      p.innerHTML = text;
      contentDiv.appendChild(p);
    } else if (sender === 'bot' && typeof marked !== 'undefined') {
      // La respuesta viene del backend (LLM + resultados web): siempre se sanitiza
      // antes de insertarla, para evitar HTML/JS malicioso inyectado vía prompt injection.
      const rawHtml = marked.parse(text);
      contentDiv.innerHTML = typeof DOMPurify !== 'undefined'
        ? DOMPurify.sanitize(rawHtml)
        : rawHtml;
    } else {
      const p = document.createElement('p');
      text.split('\n').forEach((line, i) => {
        if (i > 0) p.appendChild(document.createElement('br'));
        p.appendChild(document.createTextNode(line));
      });
      contentDiv.appendChild(p);
    }

    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);
    scrollBottom();
  }

  function addErrorMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    msgDiv.innerHTML = `
      <div class="message-avatar" style="background:#fee2e2;color:#ef4444;"><i class="fa-solid fa-triangle-exclamation"></i></div>
      <div class="message-content" style="background:#fee2e2; border:1px solid #fca5a5;">
        <p style="color:#991b1b;">Lo siento, ha ocurrido un error al conectar con el servidor. Por favor, asegúrate de que el Backend de IA esté encendido.</p>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
    scrollBottom();
  }

  function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-message';
    msgDiv.id = id;
    msgDiv.innerHTML = `
      <div class="message-avatar"><img src="../assets/images/Logo_GEIPER.png" alt="Bot"></div>
      <div class="message-content" style="padding: 10px 20px;">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
    scrollBottom();
    return id;
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function scrollBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function setLoadingState(isLoading) {
    chatInput.disabled = isLoading;
    sendBtn.disabled = isLoading;
  }

  // ── Llamadas al Backend ──

  async function checkServerConnection() {
    try {
      const res = await fetch(API_BASE + '/status');
      if (res.ok) {
        serverStatus.textContent = "Servidor en línea";
        serverStatus.style.color = "#10b981";
      } else {
        throw new Error("bad-status");
      }
    } catch (e) {
      serverStatus.textContent = "Servicio no disponible";
      serverStatus.style.color = "#ef4444";
    }
  }

  async function fetchFromBackend(query, mode) {
    const res = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, historial })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    return data.response;
  }

});
