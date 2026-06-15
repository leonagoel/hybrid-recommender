/**
 * HybridRec — Global State Engine Module
 * Extracted cleanly from monolithic core app framework layout
 */
export const state = {
    user: null,
    isGuest: true,
    products: [],    
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
    filters: { category: '', rating: '', sentiment: '' },

    // Core application feature baseline parameters
    activeChips: new Set(),
    heatmapSelected: [],
    allProducts: [],
    searchResults: [],
    recommendationSocket: null,
    pendingRecommendationTitle: null,
    realtimeReady: false
};

export function setState(changes) {
    Object.assign(state, changes);
}

export function getAnonymousUserId() {
    if (state.user && state.user.id) {
        return state.user.id;
    }
    let token = localStorage.getItem('guest_session_token');
    if (!token) {
        token = 'guest_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        localStorage.setItem('guest_session_token', token);
    }
    return token;
}

