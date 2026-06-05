// =============================================================================
// recommendations.js — Hybrid Recommendations & WebSocket
// =============================================================================
import { state, setState, getAnonymousUserId } from './state.js';
import { renderProductCards, showToast, setLoadingState, showLoadingBar, hideLoadingBar } from './ui.js';

function getRealtimeUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/recommendations`;
}

let recommendationSocket = null;
let realtimeReady = false;
let realtimeFallbackTimer = null;
let pendingRecommendationTitle = null;

export function initRecommendationSocket() {
  if (!('WebSocket' in window) || recommendationSocket) return;

  const socket = new WebSocket(getRealtimeUrl());
  recommendationSocket = socket;

  socket.addEventListener('open', () => {
    realtimeReady = true;
  });

  socket.addEventListener('message', (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'recommendations') {
        renderRecommendations(data);
      } else if (data.type === 'error') {
        throw new Error(data.detail || 'Recommendation stream failed');
      }
    } catch (err) {
      console.warn('Realtime recommendation update failed:', err.message);
      fallbackRecommendationRequest(pendingRecommendationTitle);
    }
  });

  socket.addEventListener('close', () => {
    realtimeReady = false;
    recommendationSocket = null;
  });

  socket.addEventListener('error', () => {
    realtimeReady = false;
  });
}

function requestRealtimeRecommendations(title) {
  if (!realtimeReady || !recommendationSocket) return false;

  pendingRecommendationTitle = title;
  const userId = getAnonymousUserId();
  recommendationSocket.send(JSON.stringify({
    item_title: title,
    top_n: 12,
    user_id: userId,
  }));
  return true;
}

async function fallbackRecommendationRequest(title) {
  if (!title) return;

  clearTimeout(realtimeFallbackTimer);
  realtimeFallbackTimer = setTimeout(async () => {
    try {
      const data = await API.post('/api/realtime/behavior', {
        item_title: title,
        top_n: 12,
      });
      renderRecommendations(data);
    } catch {
      await loadRecommendationsOverHttp(title);
    }
  }, 250);
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

function renderRecommendations(data) {
  const recs = data.recommendations || [];

  _updateRecsHeading(!!data.has_history);

  const recsStrip = document.getElementById('recs-strip');
  const recsLoader = document.getElementById('recs-loader');
  if (!recsStrip) return;

  if (recsLoader) recsLoader.hidden = true;
  recsStrip.hidden = false;

  if (!recs.length) {
    recsStrip.innerHTML = `
      <div class="empty-recommendations">
        <span class="empty-icon" aria-hidden="true">🔍</span>
        <p>No recommendations found. Try a different product!</p>
      </div>
    `;
    return;
  }

  recsStrip.innerHTML = recs.map((r) => `
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
  try {
    const userId = getAnonymousUserId();
    const res = await fetch(
      `/api/recommend/${encodeURIComponent(title)}?top_n=12&user_id=${encodeURIComponent(userId)}`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderRecommendations(data);
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
  recsStrip.hidden = true;
  recsStrip.innerHTML = '';

  showLoadingBar();

  try {
    const userId = getAnonymousUserId();
    const res = await fetch(
      `/api/recommend?title=${encodeURIComponent(title)}&top_n=12&user_id=${encodeURIComponent(userId)}`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderRecommendations(data);
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

const API = {
  async post(url, data) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
};