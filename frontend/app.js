/**
 * HybridRec — Frontend Application v3
 * Supabase Auth + PostgreSQL FTS Search + Modern UI
 */

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
    page: 1,
    perPage: 20,
    totalProducts: 0,
    searchTimer: null,
    searchResults: [],
    selectedSearchIdx: -1,
    isAuthSignUp: false,
    modelReady: false,
    searchHistory: [],
    selectedCategory: 'All Categories',
};

// ── DOM Elements ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const els = {
    searchInput: $('search-input'),
    searchDropdown: $('search-dropdown'),
    searchHistory: $('search-history'),
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
    skeletonLoader: $('skeleton-loader'),
    loadMoreBtn: $('load-more-btn'),
    loadMoreContainer: $('load-more-container'),
    recsSection: $('recs-section'),
    recsStrip: $('recs-strip'),
    toastContainer: $('toast-container'),
    weightAlpha: $('weight-alpha'),
    weightBeta: $('weight-beta'),
    weightGamma: $('weight-gamma'),
    categoryFilter: $('category-filter'),
};

// ── Utilities ───────────────────────────────────────────────────────
function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    els.toastContainer.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(100%)';
        el.style.transition = '300ms ease';
        setTimeout(() => el.remove(), 300);
    }, 3500);
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

function sentimentBadge(score) {
    if (score > 0.05) return '<span class="product-card__sentiment sentiment-positive">Positive</span>';
    if (score < -0.05) return '<span class="product-card__sentiment sentiment-negative">Negative</span>';
    return '<span class="product-card__sentiment sentiment-neutral">Neutral</span>';
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
    },
    async put(url, data) {
        const res = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
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
        const tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (e.key === ' ' || e.key === 'Escape' || e.ctrlKey || e.altKey || e.metaKey) return;

        if (e.key === 'Backspace') {
            els.searchInput.focus();
            return;
        }

        if (e.key.length === 1) {
            els.searchInput.focus();
            // The character will naturally be typed into the input
        }
    });
}

// ── Search ──────────────────────────────────────────────────────────
async function handleSearch(query) {
    if (!query || query.length < 1) {
        closeSearchDropdown();
        return;
    }

    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(async () => {
        try {
            const data = await API.get(`/api/search?q=${encodeURIComponent(query)}&limit=8`);
            state.searchResults = data.results || [];
            state.selectedSearchIdx = -1;
            renderSearchDropdown(state.searchResults, query);
        } catch {
            closeSearchDropdown();
        }
    }, 200);
}

function addToSearchHistory(query) {
    if (!query || !query.trim()) return;

    state.searchHistory = [
        query,
        ...state.searchHistory.filter(item => item !== query)
    ].slice(0, 5);

    renderSearchHistory();
}

function renderSearchHistory() {
    if (!state.searchHistory.length) {
        els.searchHistory.innerHTML = '';
        els.searchHistory.classList.remove('active');
        return;
    }

    els.searchHistory.innerHTML = `
        <div class="search-history__header">
            <span>Recent Searches</span>
            <button class="clear-history-btn" id="clear-history-btn">
                Clear
            </button>
        </div>

        ${state.searchHistory.map(item => `
            <div class="search-history__item" data-query="${item}">
                ${item}
            </div>
        `).join('')}
    `;

    els.searchHistory.classList.add('active');

    // Click history item
    els.searchHistory.querySelectorAll('.search-history__item')
        .forEach((el) => {
            el.addEventListener('click', () => {
                const query = el.dataset.query;
                els.searchInput.value = query;
                loadSearchResults(query);
                handleSearch(query);
            });
        });

    // Clear history
    const clearBtn = document.getElementById('clear-history-btn');

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            state.searchHistory = [];
            renderSearchHistory();
        });
    }
}

function renderSearchDropdown(results, query) {
    if (!results.length) {
        els.searchDropdown.innerHTML = `
            <div style="padding:20px;text-align:center;color:var(--text-muted);font-size:13px;">
                No results for "${query}"
            </div>`;
        els.searchDropdown.classList.add('active');
        return;
    }

    els.searchDropdown.innerHTML = results.map((r, i) => `
        <div class="search-result ${i === state.selectedSearchIdx ? 'active' : ''}"
             data-title="${r.title}" data-idx="${i}">
            <span style="font-size:20px;">${categoryIcon(r.category)}</span>
            <div class="search-result__info">
                <div class="search-result__title">${highlightMatch(r.title, query)}</div>
                <div class="search-result__meta">
                    ★ ${(r.rating || 0).toFixed(1)}
                    ${r.category ? `· <span class="search-result__category">${r.category}</span>` : ''}
                </div>
            </div>
        </div>
    `).join('');
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
    if (!query) return text;
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<strong>$1</strong>');
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
        els.skeletonLoader.hidden = false;
        state.page = 1;
    }

    try {
        const data = await API.get(`/api/search?q=&limit=${state.perPage}&offset=${(state.page - 1) * state.perPage}`);
        const products = data.results || [];
        state.totalProducts = data.total || products.length;

        if (!append) {
            els.skeletonLoader.hidden = true;
        }

        renderProducts(products, append);
        const visibleCount =
    state.selectedCategory === 'All Categories'
        ? products.length
        : products.filter(
            p => p.category === state.selectedCategory
        ).length;

els.productCount.textContent = `${visibleCount} products loaded`;

        // Show load more if there might be more
        els.loadMoreContainer.hidden = products.length < state.perPage;
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
        const data = await API.get(`/api/search?q=${encodeURIComponent(query)}&limit=40`);
        const products = data.results || [];
        els.skeletonLoader.hidden = true;
        els.productCount.textContent = `${products.length} results`;
        state.products = [];
        renderProducts(products, false);
        els.loadMoreContainer.hidden = true;
    } catch {
        els.skeletonLoader.hidden = true;
        toast('Search failed', 'error');
    }
}

function renderProducts(products, append) {
    if (!append) state.products = [];

    const fragment = document.createDocumentFragment();
    const filteredProducts =
    state.selectedCategory === 'All Categories'
        ? products
        : products.filter(
            p => p.category === state.selectedCategory
        );
    filteredProducts.forEach((p, i) => {
        state.products.push(p);
        const card = document.createElement('div');
        card.className = 'product-card';
        card.style.animationDelay = `${i * 50}ms`;
        card.innerHTML = `
            <div class="product-card__image">
                ${categoryIcon(p.category)}
            </div>
            <div class="product-card__body">
                ${p.category ? `<span class="product-card__category">${p.category}</span>` : ''}
                <h3 class="product-card__title">${p.title || 'Untitled'}</h3>
                <p class="product-card__desc">${p.description || 'No description available.'}</p>
                <div class="product-card__footer">
                    <div class="product-card__rating">
                        <div class="star-rating">${renderStars(p.rating || 0)}</div>
                        <span class="rating-value">${(p.rating || 0).toFixed(1)}</span>
                    </div>
                    ${sentimentBadge(p.avg_sentiment || 0)}
                </div>
            </div>
            <div class="product-card__actions">
                <button class="btn--add-cart" data-title="${p.title}">
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
            loadRecommendations(p.title);
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
    els.recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:13px;">Loading recommendations...</div>';

    try {
        const data = await API.get(`/api/recommend/${encodeURIComponent(title)}?top_n=12`);
        const recs = data.recommendations || [];

        if (!recs.length) {
            els.recsStrip.innerHTML = '<div style="padding:16px;color:var(--text-muted);">No recommendations found.</div>';
            return;
        }

        els.recsStrip.innerHTML = recs.map((r) => `
            <div class="rec-card" data-title="${r.title}">
                <div class="rec-card__title">${r.title}</div>
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
        `).join('');

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

// ── Upload & Build ──────────────────────────────────────────────────
async function handleUpload(file) {
    toast(`Uploading ${file.name}...`, 'info');
    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: form });
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
        loadProducts();
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
            loadProducts();
        } else if (count > 0) {
            updateStatus('has-data', `${count.toLocaleString()} products — Build models to start`);
            loadProducts();
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

// ── Event Listeners ─────────────────────────────────────────────────
function bindEvents() {
    // Search
    els.searchInput.addEventListener('input', (e) => handleSearch(e.target.value));
    els.searchInput.addEventListener('keydown', handleSearchKeydown);
    els.searchInput.addEventListener('focus', () => {
    if (els.searchInput.value) {
        handleSearch(els.searchInput.value);
    } else {
        renderSearchHistory();
    }
});
    els.categoryFilter.addEventListener('change', (e) => {
    state.selectedCategory = e.target.value;

    if (els.searchInput.value.trim()) {
        loadSearchResults(els.searchInput.value);
    } else {
        loadProducts();
    }
});

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
       if (!e.target.closest('.header__search')) {
    closeSearchDropdown();
    els.searchHistory.classList.remove('active');
}
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

    // Load more
    els.loadMoreBtn.addEventListener('click', () => {
        state.page++;
        loadProducts(true);
    });

    // Weights
    [els.weightAlpha, els.weightBeta, els.weightGamma].forEach((slider) => {
        slider.addEventListener('change', handleWeightChange);
    });
}

// ── CSS spin animation ──────────────────────────────────────────────
const spinStyle = document.createElement('style');
spinStyle.textContent = `@keyframes spin { to { transform: rotate(360deg); } } .spin { animation: spin 1s linear infinite; }`;
document.head.appendChild(spinStyle);

// ── Init ────────────────────────────────────────────────────────────
async function init() {
    bindEvents();
    initTypeToSearch();

    // Initialize Supabase client from backend config (no hardcoded keys)
    await initSupabase();
    loadCategories();
    // Run auth and status independently — neither blocks the other
    initAuth().catch((e) => console.warn('Auth error:', e));
    checkStatus().catch((e) => console.warn('Status error:', e));
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

document.addEventListener('DOMContentLoaded', init);
