/**
 * HybridRec — Frontend Application v3
 * Entry point: imports and initialisation only.
 * All logic lives in frontend/js/*.js
 */

import { els, state } from './js/state.js';
import { initSupabase, initAuth, handleAuth, toggleAuthMode } from './js/auth.js';
import {
    initTheme, initBackToTop, injectSpinStyle,
    loadPreferences, savePreferences, debounce,
    applyFilters, populateCategoryFilter,
} from './js/ui.js';
import {
    handleSearch, handleSearchKeydown, closeSearchDropdown, initTypeToSearch,
} from './js/search.js';
import {
    loadProducts, checkStatus, handleUpload, handleBuild,
    handleWeightChange, renderProducts, resetAllFiltersAndSearch,
    setupScrollObserver, destroyScrollObserver,
} from './js/recommendations.js';

// ── Bootstrap ────────────────────────────────────────────────────────
injectSpinStyle();
initTheme();

// ── Event Bindings ───────────────────────────────────────────────────
function bindEvents() {
    // Search
    els.searchInput.addEventListener('input', (e) => handleSearch(e.target.value));
    els.searchInput.addEventListener('keydown', handleSearchKeydown);
    els.searchInput.addEventListener('focus', () => {
        if (els.searchInput.value) handleSearch(els.searchInput.value);
    });
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.header__search')) closeSearchDropdown();
    });

    // Auth
    els.authBtn.addEventListener('click', () => {
        if (state.isGuest) {
            els.authModal.hidden = false;
        } else {
            import('./js/auth.js').then(({ sbClient }) => {
                sbClient.auth.signOut().then(() => {
                    state.user = null;
                    state.isGuest = true;
                    els.authLabel.textContent = 'Sign In';
                    import('./js/ui.js').then(({ toast }) => toast('Signed out', 'info'));
                    initAuth();
                });
            });
        }
    });
    els.authForm.addEventListener('submit', handleAuth);
    els.authToggleBtn.addEventListener('click', toggleAuthMode);
    els.modalClose.addEventListener('click', () => { els.authModal.hidden = true; });
    els.authModal.addEventListener('click', (e) => {
        if (e.target === els.authModal) els.authModal.hidden = true;
    });

    // Upload & Build
    els.uploadBtn.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleUpload(e.target.files[0]);
        e.target.value = '';
    });
    els.buildBtn.addEventListener('click', handleBuild);

    // Weights
    [els.weightAlpha, els.weightBeta, els.weightGamma].forEach((slider) => {
        slider.addEventListener('change', handleWeightChange);
    });

    // Heatmap close
    els.heatmapCloseBtn.addEventListener('click', () => {
        els.heatmapSection.hidden = true;
    });

    // Filters
    const debouncedSave = debounce(savePreferences, 500);

    els.categoryFilter.addEventListener('change', (e) => {
        state.filters.category = e.target.value;
        renderProducts(state.allProducts, false);
        debouncedSave();
    });
    els.ratingFilter.addEventListener('change', (e) => {
        state.filters.rating = e.target.value;
        renderProducts(state.allProducts, false);
        debouncedSave();
    });
    els.sentimentFilter.addEventListener('change', (e) => {
        state.filters.sentiment = e.target.value;
        renderProducts(state.allProducts, false);
        debouncedSave();
    });
    els.clearFiltersBtn.addEventListener('click', resetAllFiltersAndSearch);
}

// ── Init ─────────────────────────────────────────────────────────────
async function init() {
    bindEvents();
    loadPreferences();
    initTypeToSearch();
    initBackToTop();

    await initSupabase();

    initAuth().catch((e) => console.warn('Auth error:', e));
    checkStatus().catch((e) => console.warn('Status error:', e));
}

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
