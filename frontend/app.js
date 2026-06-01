import { initBenchmarkingDashboard } from './js/benchmarking.js';

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
'use strict';

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
const state = {
    user: null,
    isGuest: true,
    products: [],
    allProducts: [],
    trending: [],
    page: 1,
    perPage: 20,
    totalProducts: 0,
    isLoading: false,
    hasMore: true,
    searchTimer: null,
    searchRequestId: 0,
    isSearchLoading: false,
    autocompleteResults: [],
    selectedSearchIdx: -1,
    isAuthSignUp: false,
    modelReady: false,
    scrollObserver: null,
    compareList: [],
    heatmapSelected: [],
    activeChips: new Set(['all']),
    filters: {
        category: '',
        rating: '',
        sentiment: '',
    },
};

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
        el.style.transition = '${CONFIG.TOAST_EXIT_MS}ms ease';
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

function showSkeletons(container, count = 8) {
    container.innerHTML = Array(count)
        .fill("")
        .map(() => createSkeletonCard())
        .join("");
}

function setSearchLoading(isLoading) {
    state.isSearchLoading = isLoading;
    els.searchContainer.classList.toggle('is-loading', isLoading);
    els.searchSpinner.hidden = !isLoading;
    els.searchInput.setAttribute('aria-busy', String(isLoading));
}

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

function formatReviewCount(count) {
    if (!count || count === 0) {
        return "No reviews yet";
    }

    if (count >= 1000) {
        return `(${(count / 1000).toFixed(1)}k reviews)`;
    }

    return `(${count} reviews)`;
}

function sentimentBadge(score) {
    if (score > CONFIG.SENTIMENT_POSITIVE) return '<span class="product-card__sentiment sentiment-positive">Positive</span>';
    if (score < CONFIG.SENTIMENT_NEGATIVE) return '<span class="product-card__sentiment sentiment-negative">Negative</span>';
    return '<span class="product-card__sentiment sentiment-neutral">Neutral</span>';
}

function applyFilters(products) {
    // Pre-calculate chip states to avoid recomputing for every product
    const hasAll = state.activeChips.has('all');
    const activeCategories = Array.from(state.activeChips).filter(c => c.startsWith('category:')).map(c => c.split(':')[1]);
    const hasTopRated = state.activeChips.has('rating:top-rated');
    const hasPositive = state.activeChips.has('sentiment:positive');
    const hasTrending = state.activeChips.has('special:trending');

    return products.filter((p) => {

        const matchesCategory =
            !state.filters.category ||
            p.category === state.filters.category;

        const matchesRating =
            !state.filters.rating ||
            (p.rating || 0) >= Number(state.filters.rating);

        let sentiment = 'neutral';

        if ((p.avg_sentiment || 0) > CONFIG.SENTIMENT_POSITIVE) {
            sentiment = 'positive';
        } else if ((p.avg_sentiment || 0) < CONFIG.SENTIMENT_NEGATIVE) {
            sentiment = 'negative';
        }

        const matchesSentiment =
            !state.filters.sentiment ||
            sentiment === state.filters.sentiment;

        let traditionalMatch = matchesCategory && matchesRating && matchesSentiment;

        // Chip logic
        if (hasAll) {
            return traditionalMatch;
        }

        let pass = true;

        // Categories OR logic
        if (activeCategories.length > 0) {
            if (!activeCategories.includes(p.category)) pass = false;
        }

        // Ratings & Sentiments AND logic
        if (hasTopRated && (p.rating || 0) < 4.0) pass = false;
        if (hasPositive && sentiment !== 'positive') pass = false;
        if (hasTrending && (p.rating || 0) < 4.2) pass = false;

        return traditionalMatch && pass;
    });
}

function sortProducts(products, sortType) {
    const sorted = [...products];

    switch (sortType) {
        case 'price-low':
            return sorted.sort((a, b) => (a.price || 0) - (b.price || 0));

        case 'price-high':
            return sorted.sort((a, b) => (b.price || 0) - (a.price || 0));

        case 'rating':
            return sorted.sort((a, b) => (b.rating || 0) - (a.rating || 0));

        case 'relevance':
        default:
            return sorted;
    }
}

function applySorting() {
    const sortType = els.sortFilter?.value || 'relevance';
    const sortedProducts = sortProducts(state.allProducts || [], sortType);
    renderProducts(sortedProducts, { append: false, skipSorting: true });
}

function getSelectedSort() {
    return encodeURIComponent(els.sortFilter?.value || 'relevance');
}

function categoryIcon(cat) {
    const c = (cat || '').toLowerCase();
    if (c.includes('book') || c.includes('fiction') || c.includes('literature')) return '📚';
    if (c.includes('tech') || c.includes('computer') || c.includes('electro')) return '💻';
    if (c.includes('music') || c.includes('audio')) return '🎵';
    if (c.includes('movie') || c.includes('film') || c.includes('video')) return '🎬';
    if (c.includes('game') || c.includes('toy')) return '🎮';
    if (c.includes('food') || c.includes('kitchen') || c.includes('cook')) return '🍳';
    if (c.includes('sport') || c.includes('fitness')) return '⚽';
    if (c.includes('health') || c.includes('beauty')) return '💊';
    if (c.includes('cloth') || c.includes('fashion')) return '👕';
    if (c.includes('home') || c.includes('garden')) return '🏡';
    return '📦';
}

// ── Wishlist ────────────────────────────────────────────────────────
function getWishlist() {
    return JSON.parse(localStorage.getItem('wishlist')) || [];
}

function saveWishlist(items) {
    localStorage.setItem('wishlist', JSON.stringify(items));
}

function isWishlisted(title) {
    return getWishlist().some(item => item.title === title);
}

function toggleWishlist(product) {
    let wishlist = getWishlist();

    const exists = wishlist.some(item => item.title === product.title);

    if (exists) {
        wishlist = wishlist.filter(item => item.title !== product.title);
        toast('Removed from wishlist', 'info');
    } else {
        wishlist.push(product);
        toast('Added to wishlist', 'success');
    }

    saveWishlist(wishlist);

    renderProducts(state.allProducts, { append: false });
}

// ── API Helpers ─────────────────────────────────────────────────────
const API = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
    },
    async post(url, data) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ..._csrfHeaders() },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
    },
    async put(url, data) {
        const res = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', ..._csrfHeaders() },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
    },
};

// ── Auth ────────────────────────────────────────────────────────────
async function initAuth() {
    if (!sbClient) {
        console.warn('Supabase client unavailable — auth disabled');
        els.authLabel.textContent = 'Sign In';
        return;
    }
    try {
        const { data: { session } } = await sbClient.auth.getSession();

        if (session) {
            setUser(session.user);
        } else {
            // Auto guest sign-in
            const { data, error } = await sbClient.auth.signInAnonymously();
            if (error) {
                console.warn('Guest login failed:', error.message);
                els.authLabel.textContent = 'Sign In';
            } else {
                setUser(data.user);
            }
        }
    } catch (err) {
        console.warn('Auth init failed:', err.message);
        els.authLabel.textContent = 'Sign In';
    }
}

function setUser(user) {
    state.user = user;
    state.isGuest = user?.is_anonymous || !user?.email;

    if (state.isGuest) {
        els.authLabel.textContent = 'Guest';
    } else {
        els.authLabel.textContent = user.email?.split('@')[0] || 'User';
    }
}

async function handleAuth(e) {
    e.preventDefault();
    els.authError.hidden = true;
    els.authSubmit.disabled = true;
    els.authSubmit.textContent = 'Please wait...';

    const email = els.authEmail.value.trim();
    const password = els.authPassword.value;

    try {
        let result;
        if (state.isAuthSignUp) {
            result = await sbClient.auth.signUp({
                email,
                password,
                options: { data: { display_name: email.split('@')[0] } },
            });
        } else {
            result = await sbClient.auth.signInWithPassword({ email, password });
        }

        if (result.error) throw result.error;

        setUser(result.data.user);
        els.authModal.hidden = true;
        toast(state.isAuthSignUp ? 'Account created!' : 'Signed in!', 'success');
    } catch (err) {
        els.authError.textContent = err.message;
        els.authError.hidden = false;
    } finally {
        els.authSubmit.disabled = false;
        els.authSubmit.textContent = state.isAuthSignUp ? 'Sign Up' : 'Sign In';
    }
}

function toggleAuthMode() {
    state.isAuthSignUp = !state.isAuthSignUp;
    els.modalTitle.textContent = state.isAuthSignUp ? 'Create Account' : 'Sign In';
    els.authSubmit.textContent = state.isAuthSignUp ? 'Sign Up' : 'Sign In';
    els.authToggleText.textContent = state.isAuthSignUp ? 'Already have an account?' : "Don't have an account?";
    els.authToggleBtn.textContent = state.isAuthSignUp ? 'Sign In' : 'Sign Up';
    els.authError.hidden = true;
}

// ── Type-to-Search (Global Keyboard Capture) ────────────────────────
function initTypeToSearch() {
    document.addEventListener('keydown', (e) => {
        const activeElement = document.activeElement;
        const tag = activeElement?.tagName;

        const isTypingField =
            tag === 'INPUT' ||
            tag === 'TEXTAREA' ||
            tag === 'SELECT' ||
            activeElement?.isContentEditable;

        if (isTypingField) return;
        if (e.ctrlKey || e.altKey || e.metaKey) return;
        if (e.key !== '/') return;

        e.preventDefault();
        els.searchInput.focus();
    });
}

// ── Search Dropdown ──────────────────────────────────────────────────

function renderSearchDropdown(results, query) {
    if (!results.length) {
        closeSearchDropdown();
        return;
    }

    els.searchDropdown.innerHTML = results
        .map((title, index) => {
            const safeTitle = escapeHtml(title);
            return `
            <div
                class="search-result ${index === state.selectedSearchIdx ? 'active' : ''}"
                data-title="${safeTitle}"
                data-idx="${index}"
            >
                <span class="search-result__icon">🔍</span>
                <div class="search-result__info">
                    <div class="search-result__title">
                        ${highlightMatch(title, query)}
                    </div>
                </div>
            </div>
        `;
        })
        .join('');

    els.searchDropdown.classList.add('active');

    // Click suggestion
    els.searchDropdown.querySelectorAll('.search-result').forEach((el) => {
        el.addEventListener('click', () => {
            const title = el.dataset.title;
            selectSearchResult(title);
        });
    });
}

function highlightMatch(text, query) {
    const safeText = escapeHtml(text);
    if (!query) return safeText;
    const safeQuery = escapeHtml(query);
    const regex = new RegExp(`(${safeQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return safeText.replace(regex, '<strong>$1</strong>');
}

function selectSearchResult(title) {
    els.searchInput.value = title;

    closeSearchDropdown();

    // Trigger actual search
    loadSearchResults(title);

    // Optional recommendation refresh
    loadRecommendations(title);
}


function closeSearchDropdown() {
    els.searchDropdown.classList.remove('active');
    state.selectedSearchIdx = -1;
}

// Close dropdown when clicking outside
window.addEventListener('click', (e) => {
    const container = document.getElementById('search-container');

    if (!container.contains(e.target)) {
        closeSearchDropdown();
    }
});

function handleSearchKeydown(e) {
    const results = state.autocompleteResults;

    if (e.key === 'Enter') {
        e.preventDefault();

        if (state.selectedSearchIdx >= 0 && results.length && els.searchDropdown.classList.contains('active')) {
            const selected = results[state.selectedSearchIdx];
            selectSearchResult(selected);
        } else if (els.searchInput.value.trim().length > 0) {
            selectSearchResult(els.searchInput.value.trim());
        }
        return;
    }

    if (e.key === 'Escape') {
        closeSearchDropdown();
        return;
    }

    if (!results.length || !els.searchDropdown.classList.contains('active')) {
        return;
    }

    if (e.key === 'ArrowDown') {
        e.preventDefault();

        state.selectedSearchIdx = Math.min(
            state.selectedSearchIdx + 1,
            results.length - 1
        );

        renderSearchDropdown(results, els.searchInput.value);
    }

    else if (e.key === 'ArrowUp') {
        e.preventDefault();

        state.selectedSearchIdx = Math.max(
            state.selectedSearchIdx - 1,
            0
        );

        renderSearchDropdown(results, els.searchInput.value);
    }
}



function handleSearch(query) {
    if (!query || query.trim().length < 1) {
        state.searchRequestId++;
        setSearchLoading(false);
        closeSearchDropdown();
        return;
    }

    clearTimeout(state.searchTimer);

    // 300ms debounce
    state.searchTimer = setTimeout(async () => {
        const requestId = ++state.searchRequestId;
        setSearchLoading(true);
        try {
            const data = await API.get(
                `/api/autocomplete?q=${encodeURIComponent(query)}&limit=${CONFIG.SEARCH_LIMIT}`
            );

            if (requestId !== state.searchRequestId) return;
            state.autocompleteResults = data.suggestions || [];
            state.selectedSearchIdx = -1;

            renderSearchDropdown(state.autocompleteResults, query);
        } catch (err) {
            if (requestId === state.searchRequestId) {
                console.error('Autocomplete failed:', err);
                closeSearchDropdown();
            }
        } finally {
            if (requestId === state.searchRequestId) setSearchLoading(false);
        }
    }, CONFIG.SEARCH_DEBOUNCE_MS);
}

// ── Product Loading ─────────────────────────────────────────────────
// ── Product Loading (Infinite Scroll) ───────────────────────────────

async function loadProducts(append = false) {
    // Guard: prevent duplicate requests and loading past end
    if (state.isLoading) return;
    if (append && !state.hasMore) return;

    state.isLoading = true;

    if (!append) {
        setPageMeta(
            'All Products',
            'Browse all products on HybridRec — personalised recommendations just for you.'
        );
    }

    if (!append) {
        els.productGrid.innerHTML = '';
        showSkeletons(els.productGrid, 8);
        els.skeletonLoader.hidden = false;
        els.infiniteEnd.hidden = true;
        state.page = 1;
        state.hasMore = true;
        state.products = [];
    } else {
        els.infiniteLoader.hidden = false;
    }

    try {
        const data = await API.get(
            `/api/items?page=${state.page}&limit=${state.perPage}`
        );
        const products = data.items || [];
        state.totalProducts = data.total || 0;
        state.hasMore = data.has_more ?? products.length >= state.perPage;

        if (!append) {
            state.allProducts = [...products];
            els.skeletonLoader.hidden = true;
        } else {
            state.allProducts = [...(state.allProducts || []), ...products];
        }

        renderProducts(products, { append });
        els.productCount.textContent = `${state.products.length} of ${state.totalProducts} products`;

        if (!state.hasMore) {
            els.infiniteEnd.hidden = state.products.length === 0;
        }

        // Advance page for next fetch
        state.page++;
    } catch (err) {
        els.skeletonLoader.hidden = true;
        toast('Failed to load products', 'error');
    } finally {
        state.isLoading = false;
        els.infiniteLoader.hidden = true;
    }
}

async function loadTrending(days = 7, limit = 10) {
    els.trendingSection.hidden = true;
    els.trendingGrid.innerHTML = '';

    try {
        const data = await API.get(`/api/trending?days=${days}&limit=${limit}`);
        const items = data.results || [];
        if (!items.length) {
            return;
        }

        state.trending = items;
        renderTrending(items);
        els.trendingSection.hidden = false;
    } catch (err) {
        console.warn('Trending load failed:', err.message || err);
    }
}

function renderTrending(items) {
    els.trendingGrid.innerHTML = '';
    const fragment = document.createDocumentFragment();

    items.forEach((item, index) => {
        const title = item.title || 'Untitled';
        const safeTitle = escapeHtml(title);
        const safeCategory = escapeHtml(item.category || '');
        const safeDescription = escapeHtml(item.description || 'No description available.');
        const card = document.createElement('div');
        card.className = 'product-card trending-card';
        card.style.animationDelay = `${index * 35}ms`;
        card.innerHTML = `
            <div class="product-card__image">
                ${categoryIcon(item.category)}
            </div>
            <div class="product-card__body">
                ${item.category ? `<span class="product-card__category">${safeCategory}</span>` : ''}
                <h3 class="product-card__title">${safeTitle}</h3>
                <p class="product-card__desc">${safeDescription}</p>
                <div class="product-card__footer">
                    <div class="product-card__rating">
                        <div class="star-rating">${renderStars(item.rating || 0)}</div>
                        <span class="rating-value">${(item.rating || 0).toFixed(1)}</span>
                    </div>
                    ${sentimentBadge(item.avg_sentiment || 0)}
                </div>
            </div>
            <div class="product-card__actions">
                <button class="btn--add-cart" data-title="${safeTitle}">
                    View Trending
                </button>
            </div>
        `;

        const actionButton = card.querySelector('.btn--add-cart');
        if (actionButton) {
            actionButton.addEventListener('click', (e) => {
                e.stopPropagation();
                loadRecommendations(title);
                toast(`Showing recommendations for trending product "${title.substring(0, 40)}"`, 'info');
            });
        }

        card.addEventListener('click', () => loadRecommendations(title));
        fragment.appendChild(card);
    });

    els.trendingGrid.appendChild(fragment);
}

async function loadSearchResults(query) {
    // Pause infinite scroll during search
    destroyScrollObserver();

    const requestId = ++state.searchRequestId;
    setSearchLoading(true);
    els.productGrid.innerHTML = '';
    els.skeletonLoader.hidden = false;
    els.productsTitle.textContent = `Results for "${query}"`;
    setPageMeta(`Search: ${query}`, `Showing results for "${query}" on HybridRec.`);
    els.infiniteEnd.hidden = true;

    try {
        const data = await API.get(`/api/search?q=${encodeURIComponent(query)}&limit=40&sort=${getSelectedSort()}`);
        const products = data.results || [];
        els.skeletonLoader.hidden = true;
        els.productCount.textContent = `${data.count ?? products.length} results`;
        state.products = [];
        state.hasMore = false;
        state.allProducts = [...products];
        renderProducts(products, { append: false, ignoreFilters: true });
    } catch {
        els.skeletonLoader.hidden = true;
        toast('Search failed', 'error');
    } finally {
        if (requestId === state.searchRequestId) setSearchLoading(false);
    }
}

// ── Lazy Loading ────────────────────────────────────────────────────
const lazyObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const img = entry.target;
        img.src = img.dataset.src;
        img.onload = () => img.classList.add('loaded');
        lazyObserver.unobserve(img);
    });
}, { rootMargin: '200px 0px', threshold: 0.01 });

function createLazyImage(src, alt) {
    const img = document.createElement('img');
    img.alt = alt || '';
    img.setAttribute('loading', 'lazy');

    if ('IntersectionObserver' in window) {
        img.dataset.src = src;
        img.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"%3E%3Crect width="400" height="300" fill="%23232f3e"/%3E%3C/svg%3E';
        lazyObserver.observe(img);
    } else {
        img.src = src;
        img.classList.add('loaded');
    }

    img.addEventListener('error', () => img.classList.add('error'));
    return img;
}

function renderProducts(products, options = {}) {
    const append = options.append || false;
    const ignoreFilters = options.ignoreFilters || false;

    if (!ignoreFilters) {
        products = applyFilters(products);
    }
    els.productCount.textContent = `${products.length} products`;
    if (!append) {
    els.productGrid.innerHTML = '';
}
    if (!append) state.products = [];
    if (!products.length) {
        els.productGrid.innerHTML = `
            <div class="no-results animate-fade-in">
                <svg class="no-results-svg" width="180" height="180" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="blue-grad" x1="0" y1="0" x2="200" y2="200" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stop-color="var(--primary)" stop-opacity="0.8"/>
                            <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.1"/>
                        </linearGradient>
                        <linearGradient id="amber-grad" x1="0" y1="0" x2="200" y2="200" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stop-color="var(--accent)"/>
                            <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.3"/>
                        </linearGradient>
                    </defs>
                    <circle cx="100" cy="100" r="70" fill="url(#blue-grad)" filter="blur(8px)" opacity="0.15" />
                    <circle cx="120" cy="80" r="40" fill="url(#amber-grad)" filter="blur(6px)" opacity="0.1" />

                    <path d="M50 80 L65 140 H135 L150 80" stroke="var(--text-muted)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                    <path d="M40 80 H160" stroke="var(--text-muted)" stroke-width="4" stroke-linecap="round" />

                    <circle cx="130" cy="65" r="28" stroke="var(--primary)" stroke-width="2" stroke-dasharray="5 5" opacity="0.6"/>

                    <g class="search-glass">
                        <circle cx="130" cy="65" r="16" stroke="var(--accent)" stroke-width="3.5" fill="var(--bg-card)"/>
                        <path d="M142 77 L158 93" stroke="var(--accent)" stroke-width="3.5" stroke-linecap="round"/>
                    </g>

                    <path d="M129 60 C129 57.5, 131 56, 133 57.5 C135 59, 132 62, 132 64 M132 67 H132.01" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />

                    <circle cx="65" cy="105" r="2.5" fill="var(--accent)" opacity="0.6"/>
                    <circle cx="85" cy="105" r="3.5" fill="var(--primary)" opacity="0.5"/>
                    <circle cx="145" cy="125" r="2" fill="var(--text-muted)" opacity="0.4"/>
                </svg>
                <h3 class="no-results__title">No products found</h3>
                <p class="no-results__subtitle">Try adjusting your search keywords or clearing active filters to find what you're looking for.</p>
                <button class="btn btn--primary btn--clear-search" id="empty-state-clear-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:8px; display:inline-block; vertical-align:middle;">
                        <path d="M21 12a9 9 0 11-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
                        <polyline points="21 3 21 8 16 8"/>
                    </svg>
                    Clear Search & Filters
                </button>
            </div>
        `;

        const clearBtn = document.getElementById('empty-state-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', resetAllFiltersAndSearch);
        }
        return;
    }

    const fragment = document.createDocumentFragment();

    products.forEach((p, i) => {
        state.products.push(p);
        const title = p.title || 'Untitled';
        const safeTitle = escapeHtml(title);
        const safeCategory = escapeHtml(p.category || '');
        const safeDescription = escapeHtml(p.description || 'No description available.');
        const card = document.createElement('div');
        card.className = p.image ? 'product-card' : 'product-card product-card--skeleton';
        card.style.animationDelay = `${i * 50}ms`;
        const isChecked = state.heatmapSelected.includes(title);
        card.innerHTML = `
           <div class="product-card__image">
            <button class="wishlist-btn" data-title="${safeTitle}">
                ${isWishlisted(title) ? '❤️' : '🤍'}
            </button>

            ${categoryIcon(p.category)}
            </div>
            <div class="product-card__body">
                ${p.category ? `<span class="product-card__category">${safeCategory}</span>` : ''}
                <h3 class="product-card__title" title="${safeTitle}">
                ${safeTitle}
                </h3>
                <p class="product-card__desc">${safeDescription}</p>
                <div class="product-card__price">
                ₹${p.price || 0}
                </div>
                <div class="product-card__footer">
                    <div class="product-card__rating">
                        <div class="star-rating">${renderStars(p.rating || 0)}</div>
                        <span class="rating-value">${(p.rating || 0).toFixed(1)}</span>
                    </div>
                    ${sentimentBadge(p.avg_sentiment || 0)}
                </div>
            </div>
            <div class="product-card__actions">
                <label class="compare-label">
                    <input type="checkbox" class="compare-checkbox" data-title="${safeTitle}" ${isChecked ? 'checked' : ''}>
                    Heatmap
                </label>
                <label class="compare-label">
                    <input type="checkbox" class="side-compare-checkbox" data-title="${safeTitle}">
                    Compare
                </label>
                <button class="btn--add-cart" data-title="${safeTitle}">
                    Get Recommendations
                </button>
            </div>
        `;
        if (p.image) {
            const imgEl = createLazyImage(p.image, title);
            card.querySelector('.product-card__image').appendChild(imgEl);
        }

        // Wishlist button
        const wishlistBtn = card.querySelector('.wishlist-btn');

        if (wishlistBtn) {
            wishlistBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleWishlist(p);
            });
        }

        // Click → get recommendations
        card.querySelector('.btn--add-cart').addEventListener('click', (e) => {
            const wishlistBtn = card.querySelector('.wishlist-btn');

            wishlistBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleWishlist(p);
            });

            const title = e.target.dataset.title;
            loadRecommendations(title);
            toast(`Finding recommendations for "${title.substring(0, 40)}..."`, 'info');
        });

        // Compare checkbox
        const checkbox = card.querySelector('.compare-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                const title = checkbox.dataset.title;
                if (checkbox.checked) {
                    if (state.heatmapSelected.length >= CONFIG.MAX_COMPARE_ITEMS) {
                        checkbox.checked = false;
                        toast('Maximum 20 items for comparison', 'error');
                        return;
                    }
                    if (!state.heatmapSelected.includes(title)) {
                        state.heatmapSelected.push(title);
                    }
                } else {
                    state.heatmapSelected = state.heatmapSelected.filter(t => t !== title);
                }
                updateCompareCount();
            });
        }

        // Side-by-side compare checkbox
        const sideCheckbox = card.querySelector('.side-compare-checkbox');
        if (sideCheckbox) {
            sideCheckbox.addEventListener('change', (e) => {
                e.stopPropagation();
                const success = toggleCompare(p, sideCheckbox.checked);
                if (!success) sideCheckbox.checked = false;
            });
        }

        card.addEventListener('click', () => {
            loadRecommendations(title);
        });

        fragment.appendChild(card);
    });

    els.productGrid.appendChild(fragment);
}

// ── Recommendations ─────────────────────────────────────────────────
function getRealtimeUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/recommendations`;
}

function initRecommendationSocket() {
    if (!('WebSocket' in window) || state.recommendationSocket) return;

    const socket = new WebSocket(getRealtimeUrl());
    state.recommendationSocket = socket;

    socket.addEventListener('open', () => {
        state.realtimeReady = true;
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
            fallbackRecommendationRequest(state.pendingRecommendationTitle);
        }
    });

    socket.addEventListener('close', () => {
        state.realtimeReady = false;
        state.recommendationSocket = null;
    });

    socket.addEventListener('error', () => {
        state.realtimeReady = false;
    });
}

function requestRealtimeRecommendations(title) {
    if (!state.realtimeReady || !state.recommendationSocket) return false;

    state.pendingRecommendationTitle = title;
    state.recommendationSocket.send(JSON.stringify({
        item_title: title,
        top_n: 12,
    }));
    return true;
}

async function fallbackRecommendationRequest(title) {
    if (!title) return;

    clearTimeout(state.realtimeFallbackTimer);
    state.realtimeFallbackTimer = setTimeout(async () => {
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

function renderRecommendations(data) {
    const recs = data.results || data.recommendations || [];

    els.recsLoader.hidden = true;
    els.recsStrip.hidden = false;

    if (!recs.length) {
        els.recsStrip.innerHTML = "";
        document.getElementById("empty-state").hidden = false;
        return;
    }

    document.getElementById("empty-state").hidden = true;

    els.recsStrip.innerHTML = recs.map((r) => {
        const title = r.title || 'Untitled';
        const safeTitle = escapeHtml(title);
        return `
        <div class="rec-card" data-title="${safeTitle}">
            <div class="rec-card__title">${safeTitle}</div>
            <div class="rec-card__rating">
                <div class="star-rating">${renderStars(r.rating || 0)}</div>
                <span class="rating-value">${(r.rating || 0).toFixed(1)}</span>
            </div>
            <div class="rec-card__score">
                Score: ${(r.hybrid_score || 0).toFixed(3)}
                · Content: ${(r.content_score || 0).toFixed(2)}
                · Collab: ${(r.collab_score || 0).toFixed(2)}
            </div>
        </div>
    `;
    }).join('');

    els.recsStrip.querySelectorAll('.rec-card').forEach((card) => {
        card.addEventListener('click', () => {
            loadRecommendations(card.dataset.title);
        });
    });

    els.recsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadRecommendationsOverHttp(title) {
    const data = await API.get(`/api/recommend/${encodeURIComponent(title)}?top_n=12`);
    renderRecommendations(data);
}

async function loadRecommendations(title) {
    if (!state.modelReady) {
        toast('Build models first to get recommendations', 'info');
        return;
    }

    els.recsSection.hidden = false;
    setPageMeta(`Recommendations for ${title}`, `Products similar to "${title}" using hybrid filtering.`);
    els.recsLoader.hidden = false;
    document
.getElementById(
"empty-state"
)
.hidden=true;

els.recsStrip.innerHTML=`
<div class="recommendation-loading">

<div class="loading-card"></div>

<div class="loading-card"></div>

<div class="loading-card"></div>

</div>
`;
    els.recsStrip.hidden = true;
    els.recsStrip.innerHTML = '';

    try {
        const data = await API.get(`/api/recommend?title=${encodeURIComponent(title)}&top_n=12`);
        const recs = data.results || data.recommendations || [];

        els.recsLoader.hidden = true;
        els.recsStrip.hidden = false;

        if (!recs.length) {
    els.recsStrip.innerHTML = `
        <div class="empty-recommendations">
            <span class="empty-icon" aria-hidden="true">🔍</span>
            <p>No recommendations found. Try a different product!</p>
        </div>
    `;
    return;
}
    } catch {
        try {
            await loadRecommendationsOverHttp(title);
        } catch {
            els.recsLoader.hidden = true;
            els.recsStrip.hidden = false;
            els.recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Could not load recommendations.</div>';
        }
    }
}

// ── Upload & Build ──────────────────────────────────────────────────
async function handleUpload(file) {
    toast(`Uploading ${file.name}...`, 'info');
    const form = new FormData();
    form.append('file', file);

    try {
        // FormData POST — Content-Type is set automatically by the browser.
        // We only inject the CSRF header manually.
        const res = await fetch('/api/upload', {
            method: 'POST',
            headers: { ..._csrfHeaders() },
            body: form,
        });
        if (!res.ok) throw new Error('Upload failed');
        const data = await res.json();
        toast(`Imported ${data.imported?.toLocaleString()} products!`, 'success');
        checkStatus();
    } catch (err) {
        toast('Upload failed: ' + err.message, 'error');
    }
}

async function handleBuild() {
    els.buildBtn.disabled = true;
    els.buildBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
        </svg>
        Building...`;

    try {
        const data = await API.post('/api/build', {});
        state.modelReady = true;
        toast(`Models built in ${data.build_time_seconds}s — ${data.items?.toLocaleString()} items`, 'success');
        updateStatus('ready', `Ready — ${data.items?.toLocaleString()} products`);
        initRecommendationSocket();
        loadProducts();
        setupScrollObserver();
    } catch (err) {
        toast('Build failed: ' + err.message, 'error');
    } finally {
        els.buildBtn.disabled = false;
        els.buildBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Build Models`;
    }
}

// ── Status ──────────────────────────────────────────────────────────
async function checkStatus() {
    try {
        const data = await API.get('/api/status');
        const count = data.product_count || 0;

        if (data.model_ready) {
            state.modelReady = true;
            updateStatus('ready', `Ready — ${count.toLocaleString()} products`);
            initRecommendationSocket();
            loadProducts();
            setupScrollObserver();
        } else if (count > 0) {
            updateStatus('has-data', `${count.toLocaleString()} products — Build models to start`);
            loadProducts();
            setupScrollObserver();
        } else {
            updateStatus('', 'No data — Upload a CSV or JSON dataset');
            els.skeletonLoader.hidden = true;
            els.productGrid.innerHTML = `
                <div style="grid-column:1/-1;text-align:center;padding:60px 20px;color:var(--text-muted);">
                    <div style="font-size:48px;margin-bottom:16px;">📦</div>
                    <div style="font-size:16px;font-weight:600;margin-bottom:8px;color:var(--text-secondary);">No products yet</div>
                    <div style="font-size:13px;">Upload a CSV or JSON dataset to get started</div>
                </div>`;
        }
    } catch {
        updateStatus('error', 'Backend offline');
    }
}

function updateStatus(cls, text) {
    els.statusDot.className = `status-dot ${cls}`;
    els.statusText.textContent = text;
}

// ── Weight Controls ─────────────────────────────────────────────────
async function handleWeightChange() {
    const a = parseInt(els.weightAlpha.value);
    const b = parseInt(els.weightBeta.value);
    const g = parseInt(els.weightGamma.value);

    try {
        await API.put('/api/weights', { alpha: a / 100, beta: b / 100, gamma: g / 100 });
    } catch {}
}

async function openProductModal(product) {
    els.modalProductTitle.textContent = product.title || 'Untitled';

    els.modalProductCategory.textContent =
        `Category: ${product.category || 'Unknown'}`;

    els.modalProductRating.textContent =
        `Rating: ${(product.rating || 0).toFixed(1)}`;

    els.modalProductSentiment.textContent =
        `Sentiment: ${(product.avg_sentiment || 0).toFixed(2)}`;

    els.modalProductDescription.textContent =
        product.description || 'No description available.';

    els.modalProductScore.textContent =
        (product.hybrid_score || 0).toFixed(3);

    els.modalRecommendationsList.innerHTML =
        '<li>Loading recommendations...</li>';

    els.productModal.hidden = false;

    // Fetch top recommendations
    try {
        const data = await API.get(
            `/api/recommend/${encodeURIComponent(product.title)}?top_n=5`
        );

        const recs = data.results || data.recommendations || [];

        els.modalRecommendationsList.innerHTML = recs.map((r) => `
            <li>${r.title}</li>
        `).join('');
    } catch {
        els.modalRecommendationsList.innerHTML =
            '<li>No recommendations available.</li>';
    }
}

function closeProductModal() {
    els.productModal.hidden = true;
}
// ── Event Listeners ─────────────────────────────────────────────────
function bindEvents() {
    // Search
    els.searchInput.addEventListener('input', (e) => handleSearch(e.target.value));
    els.searchInput.addEventListener('keydown', handleSearchKeydown);
    els.searchInput.addEventListener('focus', () => {
        if (els.searchInput.value) handleSearch(els.searchInput.value);
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.header__search')) closeSearchDropdown();
    });

    // Auth
    els.authBtn.addEventListener('click', () => {
        if (state.isGuest) {
            els.authModal.hidden = false;
        } else {
            // Logged in → sign out
            sbClient.auth.signOut().then(() => {
                state.user = null;
                state.isGuest = true;
                els.authLabel.textContent = 'Sign In';
                toast('Signed out', 'info');
                initAuth(); // Re-login as guest
            });
        }
    });

    els.authForm.addEventListener('submit', handleAuth);
    els.authToggleBtn.addEventListener('click', toggleAuthMode);
    els.modalClose.addEventListener('click', () => { els.authModal.hidden = true; });
    els.authModal.addEventListener('click', (e) => {
        if (e.target === els.authModal) els.authModal.hidden = true;
    });

    // Upload
    els.uploadBtn.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleUpload(e.target.files[0]);
        e.target.value = '';
    });

    // Build
    els.buildBtn.addEventListener('click', handleBuild);

    // Weights
    [els.weightAlpha, els.weightBeta, els.weightGamma].forEach((slider) => {
        slider.addEventListener('change', handleWeightChange);
    });

    if (els.sortFilter) {
        els.sortFilter.addEventListener('change', applySorting);
    }

    // Heatmap close
    els.heatmapCloseBtn.addEventListener('click', () => {
        els.heatmapSection.hidden = true;
    });
}

// ── Similarity Heatmap ──────────────────────────────────────────────
function updateCompareCount() {
    const count = state.heatmapSelected.length;
    // Show/hide the floating compare button
    let fab = document.getElementById('compare-fab');
    if (count >= 2) {
        if (!fab) {
            fab = document.createElement('button');
            fab.id = 'compare-fab';
            fab.className = 'compare-fab';
            fab.addEventListener('click', loadHeatmap);
            document.body.appendChild(fab);
        }
        fab.textContent = `Compare ${count} Products`;
        fab.hidden = false;
    } else if (fab) {
        fab.hidden = true;
    }
}

async function loadHeatmap() {
    if (state.heatmapSelected.length < 2) {
        toast('Select at least 2 products to compare', 'info');
        return;
    }
    if (!state.modelReady) {
        toast('Build models first to compare products', 'info');
        return;
    }

    els.heatmapSection.hidden = false;
    els.heatmapLoader.hidden = false;
    els.heatmapContainer.innerHTML = '';

    try {
        const itemsParam = state.heatmapSelected.map(t => encodeURIComponent(t)).join(',');
        const data = await API.get(`/api/similarity-matrix?items=${itemsParam}`);
        els.heatmapLoader.hidden = true;

        if (data.not_found && data.not_found.length) {
            toast(`${data.not_found.length} item(s) not found in model`, 'info');
        }

        renderHeatmap(data.labels, data.matrix);
        els.heatmapSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        els.heatmapLoader.hidden = true;
        els.heatmapContainer.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Could not compute similarity matrix.</div>';
        toast('Heatmap failed: ' + err.message, 'error');
    }
}

function renderHeatmap(labels, matrix) {
    const n = labels.length;
    const gridSize = n + 1; // +1 for axis labels

    // Truncate long labels for display
    const shortLabels = labels.map(l => l.length > 25 ? l.substring(0, 22) + '…' : l);

    let html = `<div class="heatmap-grid" style="grid-template-columns: 140px repeat(${n}, 1fr); grid-template-rows: auto repeat(${n}, 1fr);">`;

    // Top-left empty corner cell
    html += '<div class="heatmap-cell heatmap-corner"></div>';

    // Top axis labels (column headers)
    for (let j = 0; j < n; j++) {
        html += `<div class="heatmap-cell heatmap-col-label" title="${labels[j]}">${shortLabels[j]}</div>`;
    }

    // Rows
    for (let i = 0; i < n; i++) {
        // Row label
        html += `<div class="heatmap-cell heatmap-row-label" title="${labels[i]}">${shortLabels[i]}</div>`;

        for (let j = 0; j < n; j++) {
            const score = matrix[i][j];
            const pct = Math.round(score * 100);
            // Color: white (0) → green (1)
            const r = Math.round(255 - score * 200);
            const g = Math.round(255 - score * 55);
            const b = Math.round(255 - score * 200);
            const bg = `rgb(${r}, ${g}, ${b})`;
            const textColor = score > 0.6 ? '#fff' : 'var(--text)';

            html += `<div class="heatmap-cell heatmap-value" style="background:${bg};color:${textColor};" title="${labels[i]} × ${labels[j]}: ${score.toFixed(4)}">
                ${score === 1 ? '1.0' : score.toFixed(2)}
            </div>`;
        }
    }

    html += '</div>';
    els.heatmapContainer.innerHTML = html;
}

// ── Infinite Scroll (Intersection Observer) ─────────────────────────
function setupScrollObserver() {
    // Tear down any previous observer to avoid duplicates / leaks
    destroyScrollObserver();

    if (!els.scrollSentinel) return;

    state.scrollObserver = new IntersectionObserver(
        (entries) => {
            const entry = entries[0];
            if (entry.isIntersecting && !state.isLoading && state.hasMore) {
                loadProducts(true);
            }
        },
        {
            // Fire when sentinel is within 200px of the viewport bottom
            rootMargin: '0px 0px 200px 0px',
            threshold: 0,
        }
    );

    state.scrollObserver.observe(els.scrollSentinel);
}

// ── Search ──────────────────────────────────────────────────────────
async function handleSearch(query) {
    if (!query || query.length < 1) {
        els.typingIndicator.hidden = true;
        closeSearchDropdown();
        return;
    }

    clearTimeout(state.searchTimer);
    els.typingIndicator.hidden = false;
    state.searchTimer = setTimeout(async () => {
        try {
            const data = await API.get(`/api/search?q=${encodeURIComponent(query)}&limit=8&sort=${getSelectedSort()}`);
            state.searchResults = data.results || [];
            state.selectedSearchIdx = -1;
            renderSearchDropdown(state.searchResults, query);
            els.typingIndicator.hidden = true;
        } catch {
            closeSearchDropdown();
            els.typingIndicator.hidden = true;
        }
    }, 300);
}

function renderSearchDropdown(results, query) {
    if (!results.length) {
        els.searchDropdown.innerHTML = `
            <div style="padding:20px;text-align:center;color:var(--text-muted);font-size:13px;">
                No results for "${escapeHtml(query)}"
            </div>`;
        els.searchDropdown.classList.add('active');
        return;
    }

    els.searchDropdown.innerHTML = results.map((r, i) => {
        const title = r.title || '';
        const safeTitle = escapeHtml(title);
        const safeCategory = escapeHtml(r.category || '');
        return `
        <div class="search-result ${i === state.selectedSearchIdx ? 'active' : ''}"
             data-title="${safeTitle}" data-idx="${i}">
            <span style="font-size:20px;">${categoryIcon(r.category)}</span>
            <div class="search-result__info">
                <div class="search-result__title">${highlightMatch(title, query)}</div>
                <div class="search-result__meta">
                    ★ ${(r.rating || 0).toFixed(1)}
                    ${r.category ? `· <span class="search-result__category">${r.category}</span>` : ''}
                </div>
            </div>
        </div>
    `;
    }).join('');
    els.searchDropdown.classList.add('active');

    // Click handlers
    els.searchDropdown.querySelectorAll('.search-result').forEach((el) => {
        el.addEventListener('click', () => {
            const title = el.dataset.title;
            selectSearchResult(title);
        });
    });
}

function highlightMatch(text, query) {
    const safeText = escapeHtml(text);
    if (!query) return safeText;
    const safeQuery = escapeHtml(query);
    const regex = new RegExp(`(${safeQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return safeText.replace(regex, '<strong>$1</strong>');
}

function selectSearchResult(title) {
    addToSearchHistory(title);
    els.searchInput.value = title;
    closeSearchDropdown();
    loadSearchResults(title);
    loadRecommendations(title);
}

function closeSearchDropdown() {
    els.searchDropdown.classList.remove('active');
    state.selectedSearchIdx = -1;
}

function handleSearchKeydown(e) {
    const results = state.searchResults;
    if (!results.length || !els.searchDropdown.classList.contains('active')) return;

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        state.selectedSearchIdx = Math.min(state.selectedSearchIdx + 1, results.length - 1);
        renderSearchDropdown(results, els.searchInput.value);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        state.selectedSearchIdx = Math.max(state.selectedSearchIdx - 1, -1);
        renderSearchDropdown(results, els.searchInput.value);
    } else if (e.key === 'Enter' && state.selectedSearchIdx >= 0) {
        e.preventDefault();
        selectSearchResult(results[state.selectedSearchIdx].title);
    } else if (e.key === 'Escape') {
        closeSearchDropdown();
    }
}

// ── Product Loading ─────────────────────────────────────────────────
async function loadProducts(append = false) {
    if (!append) {
        els.productGrid.innerHTML = '';
        els.skeletonLoader.hidden = true;
        state.page = 1;
    }

    try {
        const data = await API.get(`/api/search?q=&limit=${state.perPage}&offset=${(state.page - 1) * state.perPage}&sort=${getSelectedSort()}`);
        const products = data.results || [];
        state.totalProducts = data.total || products.length;
        state.allProducts = append
            ? [...(state.allProducts || []), ...products]
            : [...products];

        if (!append) {
            els.skeletonLoader.hidden = true;
        }

        if (append) {
            renderProducts(products, { append });
        } else {
            applySorting();
        }
        const visibleCount = applyFilters(products).length;
        els.productCount.textContent = `${visibleCount} products loaded`;

        // Show load more if there might be more
        if (els.loadMoreContainer) {
            els.loadMoreContainer.hidden = products.length < state.perPage;
        }
    } catch (err) {
        els.skeletonLoader.hidden = true;
        toast('Failed to load products', 'error');
    }
}

async function loadSearchResults(query) {
    els.productGrid.innerHTML = '';
    els.skeletonLoader.hidden = false;
    els.productsTitle.textContent = `Results for "${query}"`;

    try {
        const data = await API.get(`/api/search?q=${encodeURIComponent(query)}&limit=40&sort=${getSelectedSort()}`);
        const products = data.results || [];
        els.skeletonLoader.hidden = true;
        els.productCount.textContent = `${data.count ?? products.length} results`;
        state.products = [];
        state.allProducts = [...products];
        applySorting();
        els.searchInput.select();
        els.productGrid.classList.remove('fade-in');

        requestAnimationFrame(() => {
        els.productGrid.classList.add('fade-in');
        });
        if (els.loadMoreContainer) {
            els.loadMoreContainer.hidden = true;
        }
    } catch {
        els.skeletonLoader.hidden = true;
        toast('Search failed', 'error');
    }
}

function renderProducts(products, options = {}) {
    const append = typeof options === 'object' ? options.append || false : Boolean(options);
    const skipSorting = typeof options === 'object' ? options.skipSorting || false : false;

    products = applyFilters(products || []);

    if (!skipSorting) {
        products = sortProducts(products, els.sortFilter?.value || 'relevance');
    }

    if (!append) {
        els.productGrid.innerHTML = '';
    }

    if (!append) state.products = [];

    const fragment = document.createDocumentFragment();
    const filteredProducts = products;
    filteredProducts.forEach((p, i) => {
        state.products.push(p);
        const title = p.title || 'Untitled';
        const safeTitle = escapeHtml(title);
        const safeCategory = escapeHtml(p.category || '');
        const safeDescription = escapeHtml(p.description || 'No description available.');
        const safeImage = escapeHtml(p.image || '');
        const card = document.createElement('div');
        card.className = 'product-card';
        card.style.animationDelay = `${i * 50}ms`;
        card.innerHTML = `
            <div class="product-card__image">
            ${
                !p.image || p.image === 'undefined'
                ? `<div class="product-placeholder">${categoryIcon(p.category)}</div>`
                : `<img src="${safeImage}" alt="${safeTitle}" class="product-image" />`
             }
            </div>
            <div class="product-card__body">
                ${p.category ? `<span class="product-card__category">${safeCategory}</span>` : ''}
                <h3 class="product-card__title">${safeTitle}</h3>
                <p class="product-card__desc">${safeDescription}</p>
                <div class="product-card__footer">
                    <div class="product-card__rating">
                        <div class="star-rating">${renderStars(p.rating || 0)}</div>
                        <span class="rating-value">${(p.rating || 0).toFixed(1)}</span>
                    </div>
                    <div class="product-review-count">
                        ${formatReviewCount(p.review_count)}
                    </div>
                    ${sentimentBadge(p.avg_sentiment || 0)}
                </div>
            </div>
            <div class="product-card__actions">
                <button class="btn--add-cart" data-title="${safeTitle}">
                    Get Recommendations
                </button>
            </div>
        `;

        // Click → get recommendations
        card.querySelector('.btn--add-cart').addEventListener('click', (e) => {
            e.stopPropagation();
            const title = e.target.dataset.title;
            loadRecommendations(title);
            toast(`Finding recommendations for "${title.substring(0, 40)}..."`, 'info');
        });

        card.addEventListener('click', () => {
            openProductModal(p);
        });

        fragment.appendChild(card);
    });

    els.productGrid.appendChild(fragment);
}

// ── Recommendations ─────────────────────────────────────────────────
async function loadRecommendations(title) {
    if (!state.modelReady) {
        toast('Build models first to get recommendations', 'info');
        return;
    }

    els.recsSection.hidden = false;
    els.recsSection.classList.remove('slide-up');

        requestAnimationFrame(() => {
    els.recsSection.classList.add('slide-up');
    });
    els.recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:13px;">Loading recommendations...</div>';

    try {
        const data = await API.get(`/api/recommend/${encodeURIComponent(title)}?top_n=12`);
        const recs = data.results || data.recommendations || [];

        if (!recs.length) {
            els.recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);">No recommendations found.</div>';
            return;
        }

        els.recsStrip.innerHTML = recs.map((r) => {
            const title = r.title || 'Untitled';
            const safeTitle = escapeHtml(title);
            return `
            <div class="rec-card" data-title="${safeTitle}">
                <div class="rec-card__title">${safeTitle}</div>
                <div class="rec-card__rating">
                    <div class="star-rating">${renderStars(r.rating || 0)}</div>
                    <span class="rating-value">${(r.rating || 0).toFixed(1)}</span>
                </div>
                <div class="rec-card__score">
                    Score: ${(r.hybrid_score || 0).toFixed(3)}
                    · Content: ${(r.content_score || 0).toFixed(2)}
                    · Collab: ${(r.collab_score || 0).toFixed(2)}
                </div>
            </div>
        `;
        }).join('');

        // Click to chain recommendations
        els.recsStrip.querySelectorAll('.rec-card').forEach((card) => {
            card.addEventListener('click', () => {
                loadRecommendations(card.dataset.title);
            });
        });

        // Scroll to recs
        els.recsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch {
        els.recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Could not load recommendations.</div>';
    }
}



// ── CSS spin animation ──────────────────────────────────────────────
const spinStyle = document.createElement('style');
spinStyle.textContent = `@keyframes spin { to { transform: rotate(360deg); } } .spin { animation: spin 1s linear infinite; }`;
document.head.appendChild(spinStyle);

// ── Init ────────────────────────────────────────────────────────────
async function init() {
    bindEvents();
    initDebugMode();
    loadSavedWeights();
    initTypeToSearch();
    initFilterChips();

    // Fetch CSRF token first — must complete before any mutating request.
    await initCsrf();

    // Initialize Supabase client from backend config (no hardcoded keys)
    await initSupabase();
    loadCategories();
    // Run auth and status independently — neither blocks the other
    initAuth().catch((e) => console.warn('Auth error:', e));
    checkStatus().catch((e) => console.warn('Status error:', e));

    // Benchmarking dashboard
    initBenchmarkingDashboard();
}

async function loadCategories() {
    try {
        const data = await API.get('/api/categories');
        const categories = data.categories || [];

        els.categoryFilter.innerHTML = `
            <option value="All Categories">All Categories</option>
            ${categories.map(cat => `
                <option value="${cat}">${cat}</option>
            `).join('')}
        `;
    } catch (err) {
        console.error('Failed to load categories', err);
    }
}

// Store previous scroll position
let previousScrollPosition = 0;

// Create back button dynamically
const backButton = document.createElement("button");
backButton.id = "backToResultsBtn";
backButton.innerHTML = "← Back to Results";
document.body.appendChild(backButton);

// Hide initially
backButton.style.display = "none";

// Product grid container
const productGrid = document.querySelector(".product-grid");

// Example function when opening product detail
function openProductDetail(productId) {
    // Save current scroll position
    previousScrollPosition = window.scrollY;

    // Open detail logic
    const detailView = document.querySelector(".product-detail");
    detailView.classList.add("active");

    // Show button
    backButton.style.display = "flex";
}

// Close detail function
function closeProductDetail() {
    const detailView = document.querySelector(".product-detail");
    detailView.classList.remove("active");

    // Hide button
    backButton.style.display = "none";

    // Restore scroll position smoothly
    window.scrollTo({
        top: previousScrollPosition,
        behavior: "smooth"
    });
}

// Back button click
backButton.addEventListener("click", () => {
    closeProductDetail();
});

// Example existing product card click listeners
document.querySelectorAll(".product-card").forEach(card => {
    card.addEventListener("click", () => {
        const productId = card.dataset.id;
        openProductDetail(productId);
    });
});

document.addEventListener('DOMContentLoaded', init);
document.addEventListener('DOMContentLoaded', init);

// ── Language Toggle ─────────────────────────────────────────────────
let currentLang = 'EN';

function toggleLanguage() {
    currentLang = currentLang === 'EN' ? 'HI' : 'EN';
    document.getElementById('lang-toggle').textContent = currentLang;

    if (currentLang === 'HI') {
        document.getElementById('search-input').placeholder = 'हिंदी में खोजें...';
        document.getElementById('hindi-indicator').style.display = 'inline';
        document.getElementById('search-shortcut').style.display = 'none';
    } else {
        document.getElementById('search-input').placeholder = 'Search products...';
        document.getElementById('hindi-indicator').style.display = 'none';
        document.getElementById('search-shortcut').style.display = 'block';
    }
}

// -- Filter Chips ----------------------------------------------------
function initFilterChips() {
    const chipsContainer = document.getElementById('filter-chips');
    if (!chipsContainer) return;

    const chips = chipsContainer.querySelectorAll('.chip');

    chips.forEach(chip => {
        chip.addEventListener('click', (e) => {
            const filterVal = e.currentTarget.dataset.filter;

            if (filterVal === 'all') {
                state.activeChips.clear();
                state.activeChips.add('all');
            } else {
                state.activeChips.delete('all');

                if (state.activeChips.has(filterVal)) {
                    state.activeChips.delete(filterVal);
                } else {
                    state.activeChips.add(filterVal);
                }

                if (state.activeChips.size === 0) {
                    state.activeChips.add('all');
                }
            }

            // Update UI
            chips.forEach(c => {
                if (state.activeChips.has(c.dataset.filter)) {
                    c.classList.add('active');
                } else {
                    c.classList.remove('active');
                }
            });

            // Re-render
            renderProducts(state.allProducts, false);
        });
    });
}
