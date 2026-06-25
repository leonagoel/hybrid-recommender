// =============================================================================
// recommendations.js — Hybrid Recommendations & WebSocket
// =============================================================================
import { state, setState, getAnonymousUserId } from './state.js';
import { renderProductCards, showToast, setLoadingState, showLoadingBar, hideLoadingBar } from './ui.js';

const PAGE_SIZE = 20;
let currentOffset = 0;
let currentTitle = '';
let hasMoreRecs = false;

function getRealtimeUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/recommendations`;
}

let recommendationSocket = null;
let realtimeReady = false;
let realtimeFallbackTimer = null;
let pendingRecommendationTitle = null;
let reconnectTimer = null;
let heartbeatTimer = null;
let reconnectAttempts = 0;

const MAX_RECONNECT_ATTEMPTS = 5;
const HEARTBEAT_INTERVAL_MS = 25000;

function setRealtimeStatus(status) {
  const el = document.getElementById('realtime-status');
  if (!el) return;

  const labels = {
    live: 'Live',
    reconnecting: 'Reconnecting',
    offline: 'Offline',
  };

  el.textContent = labels[status] || 'Offline';
  el.dataset.status = status;
}

function startRealtimeHeartbeat() {
  clearInterval(heartbeatTimer);

  heartbeatTimer = setInterval(() => {
    if (!recommendationSocket || !realtimeReady) return;

    recommendationSocket.send(JSON.stringify({
      type: 'ping',
      payload: {},
    }));
  }, HEARTBEAT_INTERVAL_MS);
}


function stopRealtimeHeartbeat() {
  clearInterval(heartbeatTimer);
  heartbeatTimer = null;
}


function scheduleRealtimeReconnect() {
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    setRealtimeStatus('offline');
    return;
  }

  setRealtimeStatus('reconnecting');

  const delay = Math.min(1000 * 2 ** reconnectAttempts, 10000);
  reconnectAttempts += 1;

  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    recommendationSocket = null;
    initRecommendationSocket();
  }, delay);
}

export function initRecommendationSocket() {
  if (!('WebSocket' in window) || recommendationSocket) return;

  const socket = new WebSocket(getRealtimeUrl());
  recommendationSocket = socket;
  setRealtimeStatus('reconnecting');

  socket.addEventListener('open', () => {
    realtimeReady = true;
    reconnectAttempts = 0;
    setRealtimeStatus('live');
    startRealtimeHeartbeat();
  });

  socket.addEventListener('message', (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.type === 'pong' || data.type === 'connection_status') {
        setRealtimeStatus('live');
        return;
      }

      if (data.type === 'recommendations') {
        clearTimeout(realtimeFallbackTimer);
        renderRecommendations(data.payload || data);
        return;
      }

      if (data.type === 'error') {
        throw new Error(data.error?.message || data.message || 'Realtime request failed');
      }
    } catch (err) {
      console.warn('Realtime recommendation update failed:', err);
      if (pendingRecommendationTitle) {
        fallbackRecommendationRequest(pendingRecommendationTitle);
      }
    }
  });

  socket.addEventListener('close', () => {
    realtimeReady = false;
    recommendationSocket = null;
    stopRealtimeHeartbeat();
    scheduleRealtimeReconnect();
  });

  socket.addEventListener('error', () => {
    realtimeReady = false;
    setRealtimeStatus('offline');
  });
}

function requestRealtimeRecommendations(title) {
  if (!realtimeReady || !recommendationSocket) {
    return false;
  }

  pendingRecommendationTitle = title;

  recommendationSocket.send(JSON.stringify({
    type: 'recommendation_request',
    payload: {
      item_title: title,
      top_n: 12,
      explain: false,
      user_id: getAnonymousUserId(),
    },
  }));

  return true;
}

async function fallbackRecommendationRequest(title) {
  if (!title) return;

  try {
    const data = await API.post('/api/realtime/behavior', {
      item_title: title,
      top_n: 12,
    });
    renderRecommendations(data);
  } catch {
    await loadRecommendationsOverHttp(title);
  }
}

/**
 * Update the recommendations section heading.
 * Shows "Your personalized recommendations" when the backend confirms
 * the user has interaction history, otherwise shows the default heading.
 * @param {boolean} hasHistory
 */
function _updateRecsHeading(hasHistory) {
  const titleEl = document.querySelector('#recs-section .section-title');
  if (!titleEl) return;
  const icon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>`;
  titleEl.innerHTML = hasHistory
    ? `${icon} Your personalized recommendations`
    : `${icon} Recommended for you`;
}

function renderRecommendations(data, append = false) {
  const recs = data.recommendations || [];

  _updateRecsHeading(!!data.has_history);

  const recsStrip = document.getElementById('recs-strip');
  const recsLoader = document.getElementById('recs-loader');
  if (!recsStrip) return;

  if (recsLoader) recsLoader.hidden = true;
  recsStrip.hidden = false;

  if (!recs.length) {
    if (!append) {
      recsStrip.innerHTML = `
        <div class="empty-recommendations">
          <span class="empty-icon" aria-hidden="true">🔍</span>
          <p>No recommendations found. Try a different product!</p>
        </div>
      `;
    }
    return;
  }

  const cardsHtml = recs.map((r) => `
    <div class="rec-card" data-title="${escapeHtml(r.title)}">
      <div class="rec-card__title">${escapeHtml(r.title)}</div>
      <div class="rec-card__rating">
        <div class="star-rating">${renderStars(r.rating || 0)}</div>
        <span class="rating-value">${(r.rating || 0).toFixed(1)}</span>
        <span class="review-count">(${r.review_count || 0} reviews)</span>
      </div>
      <div class="rec-card__score">
        Score: ${(r.hybrid_score || 0).toFixed(3)}
        · Content: ${(r.content_score || 0).toFixed(2)}
        · Collab: ${(r.collab_score || 0).toFixed(2)}
      </div>
    </div>
  `).join('');

  if (append) {
    recsStrip.insertAdjacentHTML('beforeend', cardsHtml);
  } else {
    recsStrip.innerHTML = cardsHtml;
  }

  recsStrip.querySelectorAll('.rec-card').forEach((card) => {
    card.addEventListener('click', () => {
      loadRecommendations(card.dataset.title);
    });
  });

  const recsSection = document.getElementById('recs-section');
  if (recsSection) recsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadRecommendationsOverHttp(title) {
  showLoadingBar();
  currentTitle = title;
  if (currentOffset === 0) {
    const recsStrip = document.getElementById('recs-strip');
    if (recsStrip) recsStrip.innerHTML = '';
  }
  try {
    const userId = getAnonymousUserId();
    const res = await fetch(
      `/api/recommend?title=${encodeURIComponent(title)}&limit=${PAGE_SIZE}&offset=${currentOffset}&user_id=${encodeURIComponent(userId)}`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderRecommendations(data, currentOffset > 0);
    currentOffset = data.pagination.next_offset ?? currentOffset + PAGE_SIZE;
    hasMoreRecs = data.pagination.has_more;
    updateLoadMoreButton();
  } catch (err) {
    console.error('Recommendation HTTP fallback error:', err);
    showToast('Could not load recommendations.', 'error');
  } finally {
    hideLoadingBar();
  }
}

export async function loadRecommendations(title) {
  if (!state.modelReady) {
    showToast('Build models first to get recommendations', 'info');
    return;
  }

  const recsSection = document.getElementById('recs-section');
  const recsLoader = document.getElementById('recs-loader');
  const recsStrip = document.getElementById('recs-strip');
  if (!recsSection || !recsStrip) return;

  recsSection.hidden = false;
  if (recsLoader) recsLoader.hidden = false;

  // Reset pagination on new search
  currentTitle = title;
  currentOffset = 0;
  hasMoreRecs = false;
  recsStrip.innerHTML = '';
  recsStrip.hidden = true;
  updateLoadMoreButton();

  showLoadingBar();

  const sentRealtime = requestRealtimeRecommendations(title);

  if (sentRealtime) {
    clearTimeout(realtimeFallbackTimer);
    realtimeFallbackTimer = setTimeout(() => {
      fallbackRecommendationRequest(title);
    }, 2500);
    return;
  }

  try {
    const userId = getAnonymousUserId();
    const res = await fetch(
      `/api/recommend?title=${encodeURIComponent(title)}&limit=${PAGE_SIZE}&offset=0&user_id=${encodeURIComponent(userId)}`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderRecommendations(data, false);
    currentOffset = data.pagination.next_offset ?? PAGE_SIZE;
    hasMoreRecs = data.pagination.has_more;
    updateLoadMoreButton();
  } catch (err) {
    console.error('Recommendation fetch error:', err);
    try {
      await loadRecommendationsOverHttp(title);
    } catch {
      if (recsLoader) recsLoader.hidden = true;
      recsStrip.hidden = false;
      recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Could not load recommendations.</div>';
    }
  } finally {
    hideLoadingBar();
    if (recsLoader) recsLoader.hidden = true;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function renderStars(rating) {
  const full = Math.floor(rating);
  const half = rating - full >= 0.5;
  let html = '';
  for (let i = 0; i < 5; i++) {
    if (i < full) html += '<span class="star filled">★</span>';
    else if (i === full && half) html += '<span class="star filled">★</span>';
    else html += '<span class="star">★</span>';
  }
  return html;
}

function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

async function getCsrfToken() {
  const res = await fetch('/api/csrf-token');
  if (!res.ok) throw new Error(`CSRF token request failed: ${res.status}`);

  const data = await res.json();
  return data.csrfToken;
}

const API = {
  async post(url, data) {
    const csrfToken = await getCsrfToken();

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken,
      },
      body: JSON.stringify(data),
    });

    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
};

// ── Load More ──────────────────────────────────────────────────────────────────
function updateLoadMoreButton() {
  const btn = document.getElementById('load-more-btn');
  if (!btn) return;
  if (hasMoreRecs && currentTitle) {
    btn.hidden = false;
  } else {
    btn.hidden = true;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('load-more-btn');
  if (btn) {
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.textContent = 'Loading...';
      try {
        const userId = getAnonymousUserId();
        const res = await fetch(
          `/api/recommend?title=${encodeURIComponent(currentTitle)}&limit=${PAGE_SIZE}&offset=${currentOffset}&user_id=${encodeURIComponent(userId)}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderRecommendations(data, true);
        currentOffset = data.pagination.next_offset ?? currentOffset + PAGE_SIZE;
        hasMoreRecs = data.pagination.has_more;
        updateLoadMoreButton();
      } catch (err) {
        console.error('Load more error:', err);
        showToast('Failed to load more recommendations.', 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Load More Recommendations';
      }
    });
  }
});
