// state is imported read-only for rendering context (e.g. perPage, weights display).
// ui.js never calls setState() — all writes go through the calling module.
import { getStars } from './utils.js';
import { state } from './state.js';
// ── Search History Dropdown (issue #22) ─────────────────────────────────────
import { getSearchHistory, clearSearchHistory } from './state.js';
import { runSearch } from './search.js';

let historyVisible = false;

function renderSearchHistory() {
  const historyList = document.getElementById('history-list');
  if (!historyList) return;
  const history = getSearchHistory();
  if (history.length === 0) {
    historyList.innerHTML = '<li class="history-empty">No recent searches</li>';
  } else {
    historyList.innerHTML = history.map(query => `<li class="history-item" data-query="${escapeHtml(query)}">${escapeHtml(query)}</li>`).join('');
    document.querySelectorAll('.history-item').forEach(item => {
      item.addEventListener('click', () => {
        const query = item.dataset.query;
        if (query) {
          const searchInput = document.getElementById('search-input');
          if (searchInput) searchInput.value = query;
          runSearch(query);
          hideHistoryDropdown();
        }
      });
    });
  }
}

export function toggleHistoryDropdown() {
  const dropdown = document.getElementById('search-history-dropdown');
  if (!dropdown) return;
  if (historyVisible) {
    dropdown.style.display = 'none';
    historyVisible = false;
  } else {
    renderSearchHistory();
    dropdown.style.display = 'block';
    historyVisible = true;
  }
}

export function hideHistoryDropdown() {
  const dropdown = document.getElementById('search-history-dropdown');
  if (dropdown) dropdown.style.display = 'none';
  historyVisible = false;
}

export function initSearchHistory() {
  const historyBtn = document.getElementById('search-history-btn');
  const clearBtn = document.getElementById('clear-history-btn');
  if (historyBtn) {
    historyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleHistoryDropdown();
    });
  }
  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      clearSearchHistory();
      renderSearchHistory();
    });
  }
  document.addEventListener('click', (e) => {
    if (historyVisible && !e.target.closest('#search-history-dropdown') && !e.target.closest('#search-history-btn')) {
      hideHistoryDropdown();
    }
  });
}

// Auto‑initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSearchHistory);
} else {
  initSearchHistory();
}

// ── Toast Notifications ───────────────────────────────────────────────────────

export function showToast(message, type = 'info', duration = 3500) {
  const container = _getOrCreateToastContainer();
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.setAttribute('role', 'alert');
  toast.innerHTML = `
    <span class="toast__icon">${{ success:'✓', error:'✕', info:'ℹ', warning:'⚠' }[type] ?? 'ℹ'}</span>
    <span class="toast__message">${_esc(message)}</span>
    <button class="toast__close" aria-label="Dismiss">×</button>
  `;
  toast.querySelector('.toast__close').addEventListener('click', () => _dismiss(toast));
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('toast--visible'));
  setTimeout(() => _dismiss(toast), duration);
}

function _dismiss(toast) {
  toast.classList.remove('toast--visible');
  toast.addEventListener('transitionend', () => toast.remove(), { once: true });
}

function _getOrCreateToastContainer() {
  let c = document.getElementById('toast-container');
  if (!c) {
    c = document.createElement('div');
    c.id = 'toast-container';
    document.body.appendChild(c);
  }
  return c;
}

// ── Modals ────────────────────────────────────────────────────────────────────

export function showModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.setAttribute('aria-hidden', 'false');
  modal.classList.add('modal--visible');
  modal.querySelector('[autofocus], input, button')?.focus();
  document.body.classList.add('modal-open');
}

export function hideModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.setAttribute('aria-hidden', 'true');
  modal.classList.remove('modal--visible');
  document.body.classList.remove('modal-open');
}

export function initModalDismiss() {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape')
      document.querySelectorAll('.modal--visible').forEach(m => hideModal(m.id));
  });
  document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal--visible')) hideModal(e.target.id);
  });
}

// ── Loading States ────────────────────────────────────────────────────────────

export function setLoadingState(area, active) {
  document.querySelectorAll(`[data-loading-area="${area}"]`)
    .forEach(el => el.classList.toggle('is-loading', active));
  document.querySelectorAll(`[data-loading-btn="${area}"]`)
    .forEach(btn => (btn.disabled = active));
}

// ── Product Card Rendering ────────────────────────────────────────────────────

export function renderProductCards(items, opts = {}) {
  const gridId = opts.context === 'recommendations' ? 'recommendations-grid' : 'products-grid';
  const grid   = document.getElementById(gridId);
  if (!grid) return;

  if (!items.length) {
    grid.innerHTML = `<div class="empty-state"><p>${
      opts.context === 'search' && opts.query
        ? `No results for "<strong>${_esc(opts.query)}</strong>"`
        : 'No items to display.'
    }</p></div>`;
    return;
  }

  grid.innerHTML = items.map(item => _buildCard(item, opts.context)).join('');

  grid.querySelectorAll('.product-card').forEach(card => {
    card.addEventListener('click', () => {
      // Lazy import breaks any circular dep at parse time
      import('./recommendations.js').then(({ showRecommendations }) => {
        showRecommendations(card.dataset.title);
      });
    });
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault(); // stop space from scrolling the page
        card.click();
      }
    });
  });
}

function _buildCard(item, context) {
  const title    = _esc(item.title ?? item.product_name ?? 'Untitled');
  const category = _esc(item.category ?? '');
  const price    = item.price != null ? `$${parseFloat(item.price).toFixed(2)}` : '';
  const rating   = item.rating != null ? parseFloat(item.rating).toFixed(1) : null;
  const score    = item.hybrid_score ?? item.score ?? null;

  return `
    <article class="product-card" data-title="${title}"
             role="button" tabindex="0" aria-label="View recommendations for ${title}">
      <div class="card__body">
        <p class="card__category">${category}</p>
        <h3 class="card__title">${title}</h3>
        ${rating ? `
          <div class="card-rating">
            ${getStars(item.rating || 0)}
            <span class="review-count">(${item.review_count || 0} reviews)</span>
          </div>
          ` : ''}
        <div class="card__footer">
          ${price ? `<span class="card__price">${price}</span>` : ''}
          ${score != null && context === 'recommendations'
            ? `<span class="card__score">Match: ${(score*100).toFixed(0)}%</span>` : ''}
        </div>
      </div>
    </article>
  `;
}

// ── Pagination ────────────────────────────────────────────────────────────────

export function renderPagination(current, total, perPage, onPageChange) {
  const container = document.getElementById('pagination');
  if (!container) return;
  const pages = Math.ceil(total / perPage);
  if (pages <= 1) { container.innerHTML = ''; return; }

  const range = _pageRange(current, pages);
  container.innerHTML = range.map(p =>
    p === '…'
      ? `<span class="page-ellipsis">…</span>`
      : `<button class="page-btn ${p === current ? 'page-btn--active' : ''}"
               data-page="${p}" aria-label="Page ${p}">${p}</button>`
  ).join('');

  container.querySelectorAll('.page-btn').forEach(btn =>
    btn.addEventListener('click', () => onPageChange(+btn.dataset.page))
  );
}

function _pageRange(c, t) {
  if (t <= 7) return Array.from({ length: t }, (_, i) => i + 1);
  if (c <= 4) return [1,2,3,4,5,'…',t];
  if (c >= t-3) return [1,'…',t-4,t-3,t-2,t-1,t];
  return [1,'…',c-1,c,c+1,'…',t];
}

// ── Upload & Build ────────────────────────────────────────────────────────────

export function bindUploadHandler(onSuccess) {
  const input     = document.getElementById('file-upload');
  const uploadBtn = document.getElementById('upload-btn');
  if (!input || !uploadBtn) return;

  uploadBtn.addEventListener('click', () => input.click());
  input.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!['.csv','.json'].some(ext => file.name.toLowerCase().endsWith(ext))) {
      showToast('Only CSV and JSON files are supported.', 'error'); return;
    }
    setLoadingState('upload', true);
    showLoadingBar();   // 👈 start loading bar
    try {
      const form = new FormData();
      form.append('file', file);
      const res  = await fetch('/api/upload', { method: 'POST', body: form });
      if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail ?? `Error ${res.status}`);
      const data = await res.json();
      showToast(`Uploaded ${data.rows ?? ''} rows successfully.`, 'success');
      onSuccess?.(data);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setLoadingState('upload', false);
      hideLoadingBar();   // 👈 hide loading bar
      input.value = '';
    }
  });
}

export function bindBuildModelsHandler(onSuccess) {
  document.getElementById('build-btn')
    ?.addEventListener('click', async () => {
      setLoadingState('build', true);
      showToast('Building models… this may take a moment.', 'info', 8000);
      showLoadingBar();   // 👈 start loading bar
      try {
        const res = await fetch('/api/build', { method: 'POST' });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail ?? `Error ${res.status}`);
        showToast('Models built! Start searching.', 'success');
        onSuccess?.(await res.json());
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        setLoadingState('build', false);
        hideLoadingBar();   // 👈 hide loading bar
      }
    });
}

// ── Status Poller ─────────────────────────────────────────────────────────────

export function startStatusPoller(intervalMs = 30_000) {
  const poll = async () => {
    try {
      const res  = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();
      const countEl  = document.getElementById('product-count');
      const statusEl = document.getElementById('model-status');
      if (countEl) countEl.textContent = `${(data.product_count ?? 0).toLocaleString()} products`;
      if (statusEl) {
        statusEl.textContent = data.models_built ? 'Models ready' : 'Models not built';
        statusEl.className   = `status-badge status-badge--${data.models_built ? 'ok' : 'warn'}`;
      }
    } catch { /* silent */ }
  };
  poll();
  return setInterval(poll, intervalMs);
}

// ── Escape helper ─────────────────────────────────────────────────────────────

export function escapeHtml(str) {
  return String(str ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Global loading bar (issue #236) ────────────────────────────────────
let loadingBarTimeout = null;

export function showLoadingBar() {
  let bar = document.getElementById('global-loading-bar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'global-loading-bar';
    bar.className = 'loading-bar';
    document.body.appendChild(bar);
  }
  // Force reflow to restart animation
  bar.offsetHeight;
  bar.style.width = '0%';
  bar.style.opacity = '1';
  bar.style.display = 'block';
  // quickly animate to 90%
  setTimeout(() => {
    if (bar.style.width !== '100%') bar.style.width = '90%';
  }, 50);
}

export function hideLoadingBar() {
  const bar = document.getElementById('global-loading-bar');
  if (!bar) return;
  bar.style.width = '100%';
  clearTimeout(loadingBarTimeout);
  loadingBarTimeout = setTimeout(() => {
    bar.style.opacity = '0';
    setTimeout(() => {
      if (bar) bar.style.display = 'none';
    }, 200);
  }, 200);
// ── Skeleton Loading Cards ────────────────────────────────────────────────────

export function showSkeletonCards(count = 8) {
    const grid = document.getElementById('product-grid');
    if (!grid) return;
    grid.innerHTML = Array.from({ length: count }, () => `
        <div class="product-card skeleton-card">
            <div class="skeleton skeleton-image"></div>
            <div class="product-info">
                <div class="skeleton skeleton-title"></div>
                <div class="skeleton skeleton-text"></div>
                <div class="skeleton skeleton-text short"></div>
                <div class="skeleton-footer">
                    <div class="skeleton skeleton-price"></div>
                    <div class="skeleton skeleton-button"></div>
                </div>
            </div>
        </div>
    `).join('');
}

export function hideSkeletonCards() {
    document.querySelectorAll('.skeleton-card').forEach(el => el.remove());
}
