// =============================================================================
// search.js — Search Functionality
// Debounced FTS, global keyboard capture, category filter, pagination.
// =============================================================================

import { state, setState, addToSearchHistory } from './state.js';
import { renderProductCards, setLoadingState, showToast, renderPagination, showLoadingBar, hideLoadingBar } from './ui.js';

const DEBOUNCE_MS = 300;
let _debounceTimer = null;

/** Bind search input, keyboard capture, category filter. Call once from app.js. */
export function initSearch() {
  _bindSearchInput();
  _bindGlobalKeyCapture();
  _bindCategoryFilter();
}

/** Run a search query against /api/search. */
export async function runSearch(query, limit = 20) {
  const q = query.trim();
  setState({ lastQuery: q, isSearching: true });
  setLoadingState('search', true);
  showLoadingBar();

  try {
    const params = new URLSearchParams({ q, limit });
    if (state.activeCategory) params.set('category', state.activeCategory);

    const res  = await fetch(`/api/search?${params}`);
    if (!res.ok) throw new Error(`Search error: ${res.status}`);

    const data    = await res.json();
    const results = data.results ?? [];

    setState({ searchResults: results, isSearching: false });
    addToSearchHistory(q);   // <-- ADD SEARCH TO HISTORY
    renderProductCards(results, { context: 'search', query: q });
    renderPagination(1, data.count ?? data.total ?? results.length, state.perPage, loadProducts);
  } catch (err) {
    showToast('Search failed. Please try again.', 'error');
    console.error('[search]', err);
    setState({ isSearching: false });
  } finally {
    setLoadingState('search', false);
    hideLoadingBar();
  }
}

/** Load paginated product listing (no query). */
export async function loadProducts(page = 1) {
  setLoadingState('products', true);
  setState({ currentPage: page });
  showLoadingBar();

  try {
    const params = new URLSearchParams({ page, per_page: state.perPage });
    if (state.activeCategory) params.set('category', state.activeCategory);

    const res   = await fetch(`/api/items?${params}`);
    if (!res.ok) throw new Error(`Items error: ${res.status}`);

    const data  = await res.json();
    const items = data.items ?? data ?? [];

    setState({ searchResults: items });
    renderProductCards(items, { context: 'browse', page });
    renderPagination(page, data.total ?? items.length, state.perPage, loadProducts);
  } catch (err) {
    showToast('Failed to load products.', 'error');
    console.error('[search]', err);
  } finally {
    setLoadingState('products', false);
    hideLoadingBar();
  }
}

/** Fetch categories and populate the dropdown. */
export async function loadCategories() {
  showLoadingBar();
  try {
    const res  = await fetch('/api/categories');
    if (!res.ok) return;
    const data = await res.json();
    const cats = data.categories ?? data ?? [];
    setState({ categories: cats });
    _renderCategoryOptions(cats);
  } catch (err) {
    console.warn('[search] loadCategories:', err);
  } finally {
    hideLoadingBar();
  }
}

// ── Internal ──────────────────────────────────────────────────────────────────

function _bindSearchInput() {
  const input = document.getElementById('search-input');
  const counter = document.getElementById('search-char-counter');
  if (!input) return;

  // Update counter function
  const updateCounter = () => {
    if (!counter) return;
    const len = input.value.length;
    counter.textContent = `${len}/100`;
  };

  // Show/hide counter on focus/blur
  input.addEventListener('focus', () => {
    if (counter) {
      counter.style.display = 'block';
      updateCounter();
    }
  });
  input.addEventListener('blur', () => {
    if (counter) counter.style.display = 'none';
  });

  // Existing debounced search (keep it, but add counter update)
  input.addEventListener('input', (e) => {
    clearTimeout(_debounceTimer);
    const q = e.target.value;
    // Update character counter
    updateCounter();
    // Existing behaviour
    if (!q.trim()) { loadProducts(1); return; }
    _debounceTimer = setTimeout(() => runSearch(q), DEBOUNCE_MS);
  });
}
function _bindSearchInput() {
  const input = document.getElementById('search-input');
  const clearBtn = document.getElementById('clear-search-btn');
  if (!input) return;

  // === Existing debounced search logic ===
  input.addEventListener('input', (e) => {
    clearTimeout(_debounceTimer);
    const q = e.target.value;
    if (!q.trim()) { loadProducts(1); return; }
    _debounceTimer = setTimeout(() => runSearch(q), DEBOUNCE_MS);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { input.value = ''; loadProducts(1); }
  });

  // === Clear button logic (new) ===
  if (clearBtn) {
    const toggleClearButton = () => {
      clearBtn.style.display = input.value.length > 0 ? 'block' : 'none';
    };
    toggleClearButton();
    input.addEventListener('input', toggleClearButton);
    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      input.value = '';
      toggleClearButton();
      loadProducts(1);
      input.focus();
    });
    // Also handle Escape (already handled above, but we need to update button)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') setTimeout(toggleClearButton, 0);
    });
  }
}
function _bindGlobalKeyCapture() {
  document.addEventListener('keydown', (e) => {
    const tag = document.activeElement?.tagName?.toLowerCase();
    if (['input', 'textarea', 'select'].includes(tag)) return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.key.length !== 1) return;
    document.getElementById('search-input')?.focus();
  });
}

function _bindCategoryFilter() {
  document.getElementById('category-filter')
    ?.addEventListener('change', (e) => {
      setState({ activeCategory: e.target.value || null, currentPage: 1 });
      state.lastQuery ? runSearch(state.lastQuery) : loadProducts(1);
    });
}

function _renderCategoryOptions(categories) {
  const select = document.getElementById('category-filter');
  if (!select) return;
  select.innerHTML = '<option value="">All Categories</option>';
  categories.forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat; opt.textContent = cat;
    select.appendChild(opt);
  });
}
