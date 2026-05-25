// =============================================================================
// recommendations.js — Recommendation Display
// Hybrid rec fetch, α/β/γ sliders, recently viewed panel.
// =============================================================================

import { state, setState, addRecentlyViewed } from './state.js';
import { renderProductCards, setLoadingState, showToast, escapeHtml } from './ui.js';
import { isAuthenticated } from './auth.js';

let _sliderDebounce = null;
/** Fetch and display hybrid recommendations for a product title. */
export async function showRecommendations(title) {
  if (!title) return;
  setState({ currentItem: title, isLoadingRecs: true });
  setLoadingState('recommendations', true);
  addRecentlyViewed(title);

  try {
    const params = new URLSearchParams(state.weights);
    const res    = await fetch(`/api/recommend/${encodeURIComponent(title)}?${params}`);
    if (!res.ok) throw new Error(`Recommend error: ${res.status}`);

    const data  = await res.json();
    const items = data.recommendations ?? data ?? [];

    setState({ recommendations: items, isLoadingRecs: false });
    _renderRecommendationSection(title, items);

    if (isAuthenticated() && state.user?.id) {
      _recordViewEvent(title).catch(() => {}); // non-critical
    }
  } catch (err) {
    showToast('Could not load recommendations.', 'error');
    console.error('[recommendations]', err);
    setState({ isLoadingRecs: false });
  } finally {
    setLoadingState('recommendations', false);
  }
}

/** Bind α/β/γ sliders. Call once from app.js. */
export function initWeightSliders() {
  ['alpha', 'beta', 'gamma'].forEach(key => {
    const slider = document.getElementById(`slider-${key}`);
    const valEl  = document.getElementById(`val-${key}`);
    if (!slider) return;

    slider.value = state.weights[key];
    if (valEl) valEl.textContent = state.weights[key].toFixed(2);

    slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        setState({ weights: { ...state.weights, [key]: val } });
        if (valEl) valEl.textContent = val.toFixed(2);

        clearTimeout(_sliderDebounce);
        _sliderDebounce = setTimeout(async () => {
            _persistWeights(state.weights).catch(() => {});
            if (state.currentItem) await showRecommendations(state.currentItem);
        }, 400); // wait 400ms after user stops sliding
    });
  });
}

/** Re-render the recently viewed sidebar panel. */
export function renderRecentlyViewed() {
  const container = document.getElementById('recently-viewed-list');
  if (!container) return;

  if (!state.recentlyViewed.length) {
    container.innerHTML = '<p class="empty-state">No items viewed yet.</p>';
    return;
  }

  container.innerHTML = state.recentlyViewed.map(title => `
    <div class="recent-item" data-title="${escapeHtml(title)}" role="button" tabindex="0">
      <span class="recent-item__title">${escapeHtml(title)}</span>
      <span class="recent-item__arrow">→</span>
    </div>
  `).join('');

  container.querySelectorAll('.recent-item').forEach(el => {
    el.addEventListener('click', () => showRecommendations(el.dataset.title));
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        showRecommendations(el.dataset.title);
      }
    });
  });
}

// ── Internal ──────────────────────────────────────────────────────────────────

function _renderRecommendationSection(sourceTitle, items) {
  const section   = document.getElementById('recommendations-section');
  const titleEl   = document.getElementById('recommendations-title');
  const container = document.getElementById('recommendations-grid');
  if (!section || !container) return;

  section.style.display = 'block';
  if (titleEl) titleEl.textContent = `Because you viewed: ${sourceTitle}`;

  items.length
    ? renderProductCards(items, { context: 'recommendations', sourceTitle })
    : (container.innerHTML = '<p class="empty-state">No recommendations found.</p>');
}

async function _recordViewEvent(title) {
  await fetch('/api/purchases', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ user_id: state.user.id, item_title: title, event_type: 'view' }),
  });
}

async function _persistWeights(weights) {
  await fetch('/api/weights', {
    method:  'PUT',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(weights),
  });
}
