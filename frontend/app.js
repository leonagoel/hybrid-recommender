

// ===== THEME TOGGLE =====
const themeToggle = document.getElementById('theme-toggle');
const root = document.documentElement;

function initThemeToggle() {
  if (!themeToggle) return;

  const savedTheme = localStorage.getItem('theme') || 'dark';

  root.setAttribute('data-theme', savedTheme);
  themeToggle.textContent = savedTheme === 'dark' ? '🌙' : '☀️';

  themeToggle.setAttribute(
    'aria-label',
    savedTheme === 'dark'
      ? 'Switch to light mode'
      : 'Switch to dark mode'
  );

  themeToggle.addEventListener('click', () => {
    const current = root.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';

    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    themeToggle.textContent = next === 'dark' ? '🌙' : '☀️';

    themeToggle.setAttribute(
      'aria-label',
      next === 'dark'
        ? 'Switch to light mode'
        : 'Switch to dark mode'
    );
  });

  themeToggle.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      themeToggle.click();
    }
  });
}

document.addEventListener('DOMContentLoaded', initThemeToggle);

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

/**
 * HybridRec — Frontend Application v3
 * Supabase Auth + PostgreSQL FTS Search + Modern UI
 */

// ── CSRF Token ──────────────────────────────────────────────────────
// Fetched once from /api/csrf-token and kept in memory.
// Every mutating request (POST / PUT / PATCH / DELETE) must include it
// as the X-CSRF-Token header to satisfy the Double Submit Cookie check.
let _csrfToken = null;

async function initCsrf() {
    try {
        const res = await fetch('/api/csrf-token');
        if (!res.ok) throw new Error(`CSRF fetch failed: ${res.status}`);
        const data = await res.json();
        _csrfToken = data.csrfToken || null;
    } catch (e) {
        console.warn('CSRF init failed:', e.message);
    }
}

function _csrfHeaders() {
    return _csrfToken ? { 'X-CSRF-Token': _csrfToken } : {};
}

// ── Supabase Client ─────────────────────────────────────────────────
// Loaded dynamically from backend — no hardcoded credentials
let sbClient = null;

async function initSupabase() {
    try {
        const resp = await fetch('/api/config');
        if (!resp.ok) return null;
        const config = await resp.json();
        const { createClient } = window.supabase || {};
        if (createClient && config.supabase_url && config.supabase_anon_key) {
            sbClient = createClient(config.supabase_url, config.supabase_anon_key);
        }
    } catch (e) {
        console.warn('Supabase init skipped:', e.message);
    }
    return sbClient;
}


// ── State ───────────────────────────────────────────────────────────


// ── DOM Elements ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const els = {
    searchInput: $('search-input'),
    searchContainer: $('search-container'),
    searchDropdown: $('search-dropdown'),
    searchSpinner: $('search-spinner'),
    searchShortcut: $('search-shortcut'),
    authBtn: $('auth-btn'),
    authLabel: $('auth-label'),
    authModal: $('auth-modal'),
    authForm: $('auth-form'),
    authEmail: $('auth-email'),
    authPassword: $('auth-password'),
    authSubmit: $('auth-submit'),
    authError: $('auth-error'),
    authToggleBtn: $('auth-toggle-btn'),
    authToggleText: $('auth-toggle-text'),
    modalTitle: $('modal-title'),
    modalClose: $('modal-close'),
    statusDot: $('status-dot'),
    statusText: $('status-text'),
    uploadBtn: $('upload-btn'),
    buildBtn: $('build-btn'),
    fileInput: $('file-input'),
    productGrid: $('product-grid'),
    productsTitle: $('products-title'),
    productCount: $('product-count'),
    trendingSection: $('trending-section'),
    trendingGrid: $('trending-grid'),
    skeletonLoader: $('skeleton-loader'),
    scrollSentinel: $('scroll-sentinel'),
    infiniteLoader: $('infinite-scroll-loader'),
    infiniteEnd: $('infinite-scroll-end'),
    recsSection: $('recs-section'),
    recsLoader: $('recs-loader'),
    recsStrip: $('recs-strip'),
    heatmapSection: $('heatmap-section'),
    heatmapLoader: $('heatmap-loader'),
    heatmapContainer: $('heatmap-container'),
    heatmapCloseBtn: $('heatmap-close-btn'),
    toastContainer: $('toast-container'),
    weightAlpha: $('weight-alpha'),
    weightBeta: $('weight-beta'),
    weightGamma: $('weight-gamma'),
    productModal: $('product-modal'),
    productModalClose: $('product-modal-close'),
    modalProductTitle: $('modal-product-title'),
    modalProductCategory: $('modal-product-category'),
    modalProductRating: $('modal-product-rating'),
    modalProductSentiment: $('modal-product-sentiment'),
    modalProductDescription: $('modal-product-description'),
    modalProductScore: $('modal-product-score'),
    modalRecommendationsList: $('modal-recommendations-list'),
    categoryFilter: $('category-filter'),
    sortFilter: $('sort-filter'),
    ratingFilter: $('rating-filter'),
    sentimentFilter: $('sentiment-filter'),
    clearFiltersBtn: $('clear-filters'),
};
// ===== CONFIG=====
const CONFIG = {
  TOAST_DURATION_MS: 3500,
  TOAST_EXIT_MS: 300,
  SEARCH_DEBOUNCE_MS: 300,
  SENTIMENT_POSITIVE: 0.05,
  SENTIMENT_NEGATIVE: -0.05,
  SEARCH_LIMIT: 5,
  MAX_COMPARE_ITEMS: 20
};

function loadPreferences() {
    const saved = localStorage.getItem('userPreferences');

    if (!saved) return;

    try {
        const prefs = JSON.parse(saved);

        state.filters.category = prefs.category || '';
        state.filters.rating = prefs.rating || '';
        state.filters.sentiment = prefs.sentiment || '';

        els.categoryFilter.value = state.filters.category;
        els.ratingFilter.value = state.filters.rating;
        els.sentimentFilter.value = state.filters.sentiment;

    } catch (err) {
        console.warn('Failed to load preferences:', err);
    }
}
// ── Utilities ───────────────────────────────────────────────────────
function setPageMeta(title, description) {
    if (title) {
        document.title = `${title} — HybridRec`;
    } else {
        document.title = 'HybridRec — Smart Recommendations';
    }
    const metaDesc = document.querySelector('meta[name="description"]');
    if (metaDesc && description) {
        metaDesc.setAttribute('content', description);
    }
}

function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    els.toastContainer.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(100%)';
        el.style.transition = `${CONFIG.TOAST_EXIT_MS}ms ease`;
        setTimeout(() => el.remove(), CONFIG.TOAST_EXIT_MS);
    }, CONFIG.TOAST_DURATION_MS);
}

function createSkeletonCard() {
    return `
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
    `;
}

