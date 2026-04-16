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
  const API_BASE = window.GEIPER_API_URL || 'http://localhost:8000';
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
    currentBotTitle.innerHTML = botsConfig[currentMode].title;
    addMessage(botsConfig[currentMode].greeting, 'bot', true);
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
    const p = document.createElement('p');

    if (isHtml) {
      p.innerHTML = text;
    } else {
      text.split('\n').forEach((line, i) => {
        if (i > 0) p.appendChild(document.createElement('br'));
        p.appendChild(document.createTextNode(line));
      });
    }

    contentDiv.appendChild(p);
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
      body: JSON.stringify({ query, mode })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    return data.response;
  }

});
