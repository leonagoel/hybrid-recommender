// =============================================================================
// state.js — Global State Management
// Single source of truth. All modules read/write state only through here.
// =============================================================================

export const state = {
  // Auth
  user: null,
  session: null,
  isGuest: false,

  // Dataset
  datasetLoaded: false,
  modelsBuilt: false,
  productCount: 0,
  categories: [],

  // Hybrid weights (alpha=content, beta=collab, gamma=sentiment)
  weights: { alpha: 0.4, beta: 0.4, gamma: 0.2 },

  // Search
  lastQuery: '',
  searchResults: [],
  isSearching: false,
  searchHistory: JSON.parse(localStorage.getItem('searchHistory') || '[]'),

  // Recommendations
  currentItem: null,
  recommendations: [],
  isLoadingRecs: false,

  // UI / Pagination
  currentPage: 1,
  perPage: 50,
  activeCategory: null,
  recentlyViewed: [],   // max 10 items
};

// ── Pub/Sub ──────────────────────────────────────────────────────────────────
const _listeners = {};

/**
 * Subscribe to changes on a top-level state key.
 * @param {string} key
 * @param {Function} cb  called with (newValue, oldValue)
 * @returns {Function}   unsubscribe
 */
export function subscribe(key, cb) {
  if (!_listeners[key]) _listeners[key] = new Set();
  _listeners[key].add(cb);
  return () => _listeners[key].delete(cb);
}

/**
 * Update state keys and notify subscribers.
 * @param {Partial<typeof state>} patch
 */
export function setState(patch) {
  for (const [key, newVal] of Object.entries(patch)) {
    const old = state[key];
    state[key] = newVal;
    _listeners[key]?.forEach(cb => cb(newVal, old));
  }
}

/**
 * Add a title to recently viewed (no duplicates, max 10).
 * @param {string} title
 */
export function addRecentlyViewed(title) {
  const list = state.recentlyViewed.filter(t => t !== title);
  list.unshift(title);
  setState({ recentlyViewed: list.slice(0, 10) });
}

// ── Search History Helpers (issue #22) ──────────────────────────────────────

/**
 * Add a query to search history (no duplicates, max 5, most recent first).
 * @param {string} query
 */
export function addToSearchHistory(query) {
  if (!query || query.trim() === '') return;
  let history = [...state.searchHistory];
  history = history.filter(item => item !== query);
  history.unshift(query);
  history = history.slice(0, 5);
  state.searchHistory = history;
  localStorage.setItem('searchHistory', JSON.stringify(history));
}

/**
 * Clear the entire search history.
 */
export function clearSearchHistory() {
  state.searchHistory = [];
  localStorage.setItem('searchHistory', '[]');
}

/**
 * Get a copy of the current search history.
 * @returns {string[]}
 */
export function getSearchHistory() {
  return [...state.searchHistory];
}

// ── Anonymous Session ID (Issue #64) ────────────────────────────────────────

/**
 * Get or create a persistent anonymous user ID for this browser session.
 * Uses sessionStorage so it is scoped to the tab — no cross-session
 * fingerprinting. Generated once on first visit, reused on every
 * /api/recommend call so the collaborative model can build history.
 * @returns {string} UUID v4
 */
export function getAnonymousUserId() {
  let id = sessionStorage.getItem('anon_user_id');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('anon_user_id', id);
  }
  return id;
}