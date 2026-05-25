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