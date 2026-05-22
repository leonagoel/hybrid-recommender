import { state, els } from './state.js';
import { API } from './api.js';
import {
    toast, renderStars, sentimentBadge, categoryIcon,
    applyFilters, isWishlisted, saveWishlist, getWishlist,
    setPageMeta, updateStatus, renderHeatmap, updateCompareBar,
    openComparePage, debounce, savePreferences, isValidUploadFile,
    esc,
} from './ui.js';

// ── Lazy Image Loading ───────────────────────────────────────────────
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

// ── Wishlist Toggle ──────────────────────────────────────────────────
function toggleWishlist(product) {
    let wishlist = getWishlist();
    const exists = wishlist.some((item) => item.title === product.title);
    if (exists) {
        wishlist = wishlist.filter((item) => item.title !== product.title);
        toast('Removed from wishlist', 'info');
    } else {
        wishlist.push(product);
        toast('Added to wishlist', 'success');
    }
    saveWishlist(wishlist);
    renderProducts(state.allProducts, false);
}

// ── Product Rendering ────────────────────────────────────────────────
export function renderProducts(products, append) {
    products = applyFilters(products);
    els.productCount.textContent = `${products.length} products`;
    if (!append) els.productGrid.innerHTML = '';
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
                    </defs>
                    <circle cx="100" cy="100" r="70" fill="url(#blue-grad)" filter="blur(8px)" opacity="0.15" />
                    <g class="search-glass">
                        <circle cx="130" cy="65" r="16" stroke="var(--accent)" stroke-width="3.5" fill="var(--bg-card)"/>
                        <path d="M142 77 L158 93" stroke="var(--accent)" stroke-width="3.5" stroke-linecap="round"/>
                    </g>
                </svg>
                <h3 class="no-results__title">No products found</h3>
                <p class="no-results__subtitle">Try adjusting your search keywords or clearing active filters.</p>
                <button class="btn btn--primary btn--clear-search" id="empty-state-clear-btn">
                    Clear Search &amp; Filters
                </button>
            </div>
        `;
        document.getElementById('empty-state-clear-btn')
            ?.addEventListener('click', resetAllFiltersAndSearch);
        return;
    }

    const fragment = document.createDocumentFragment();

    products.forEach((p, i) => {
        state.products.push(p);
        const card = document.createElement('div');
        card.className = p.image ? 'product-card' : 'product-card product-card--skeleton';
        card.style.animationDelay = `${i * 50}ms`;
        // Use tabindex so keyboard users can reach cards
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.setAttribute('aria-label', `Get recommendations for ${p.title}`);

        const isChecked = state.heatmapSelected.includes(p.title);
        card.innerHTML = `
           <div class="product-card__image">
            <button class="wishlist-btn" data-title="${esc(p.title)}" aria-label="${isWishlisted(p.title) ? 'Remove from wishlist' : 'Add to wishlist'}">
                ${isWishlisted(p.title) ? '❤️' : '🤍'}
            </button>
            ${categoryIcon(p.category)}
            </div>
            <div class="product-card__body">
                ${p.category ? `<span class="product-card__category">${esc(p.category)}</span>` : ''}
                <h3 class="product-card__title">${esc(p.title || 'Untitled')}</h3>
                <p class="product-card__desc">${esc(p.description || 'No description available.')}</p>
                <div class="product-card__price">₹${esc(String(p.price || 0))}</div>
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
                    <input type="checkbox" class="compare-checkbox" data-title="${esc(p.title)}" ${isChecked ? 'checked' : ''}>
                    Heatmap
                </label>
                <label class="compare-label">
                    <input type="checkbox" class="side-compare-checkbox" data-title="${esc(p.title)}">
                    Compare
                </label>
                <button class="btn--add-cart" data-title="${esc(p.title)}">
                    Get Recommendations
                </button>
            </div>
        `;

        if (p.image) {
            card.querySelector('.product-card__image').appendChild(createLazyImage(p.image, p.title));
        }

        card.querySelector('.wishlist-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            toggleWishlist(p);
        });

        card.querySelector('.btn--add-cart').addEventListener('click', (e) => {
            e.stopPropagation();
            const title = e.target.dataset.title;
            loadRecommendations(title);
            toast(`Finding recommendations for "${title.substring(0, 40)}..."`, 'info');
        });

        // FIX: Space key on card — call preventDefault() to stop page scroll
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                loadRecommendations(p.title);
            } else if (e.key === ' ') {
                e.preventDefault(); // prevents page scroll on Space
                loadRecommendations(p.title);
            }
        });

        const checkbox = card.querySelector('.compare-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                const title = checkbox.dataset.title;
                if (checkbox.checked) {
                    if (state.heatmapSelected.length >= 20) {
                        checkbox.checked = false;
                        toast('Maximum 20 items for comparison', 'error');
                        return;
                    }
                    if (!state.heatmapSelected.includes(title)) state.heatmapSelected.push(title);
                } else {
                    state.heatmapSelected = state.heatmapSelected.filter((t) => t !== title);
                }
                updateCompareCount();
            });
        }

        const sideCheckbox = card.querySelector('.side-compare-checkbox');
        if (sideCheckbox) {
            sideCheckbox.addEventListener('change', (e) => {
                e.stopPropagation();
                const success = toggleCompare(p, sideCheckbox.checked);
                if (!success) sideCheckbox.checked = false;
            });
        }

        card.addEventListener('click', () => loadRecommendations(p.title));
        fragment.appendChild(card);
    });

    els.productGrid.appendChild(fragment);
}

// ── Product Loading (Infinite Scroll) ───────────────────────────────
export async function loadProducts(append = false) {
    if (state.isLoading) return;
    if (append && !state.hasMore) return;

    state.isLoading = true;

    if (!append) {
        setPageMeta('All Products', 'Browse all products on HybridRec — personalised recommendations just for you.');
        els.productGrid.innerHTML = '';
        els.skeletonLoader.hidden = false;
        els.infiniteEnd.hidden = true;
        state.page = 1;
        state.hasMore = true;
        state.products = [];
    } else {
        els.infiniteLoader.hidden = false;
    }

    try {
        const data = await API.get(`/api/items?page=${state.page}&limit=${state.perPage}`);
        const products = data.items || [];
        state.totalProducts = data.total || 0;
        state.hasMore = data.has_more ?? products.length >= state.perPage;

        if (!append) {
            state.allProducts = [...products];
            els.skeletonLoader.hidden = true;
        } else {
            state.allProducts = [...(state.allProducts || []), ...products];
        }

        renderProducts(products, append);
        els.productCount.textContent = `${state.products.length} of ${state.totalProducts} products`;

        if (!state.hasMore) els.infiniteEnd.hidden = state.products.length === 0;

        state.page++;
    } catch (err) {
        els.skeletonLoader.hidden = true;
        toast('Failed to load products', 'error');
    } finally {
        state.isLoading = false;
        els.infiniteLoader.hidden = true;
    }
}

export async function loadSearchResults(query) {
    destroyScrollObserver();
    els.productGrid.innerHTML = '';
    els.skeletonLoader.hidden = false;
    els.productsTitle.textContent = `Results for "${esc(query)}"`;
    setPageMeta(`Search: ${query}`, `Showing results for "${query}" on HybridRec.`);
    els.infiniteEnd.hidden = true;

    try {
        const data = await API.get(`/api/search?q=${encodeURIComponent(query)}&limit=40`);
        const products = data.results || data.items || [];
        els.skeletonLoader.hidden = true;
        els.productCount.textContent = `${products.length} results`;
        state.products = [];
        state.hasMore = false;
        state.allProducts = [...products];
        renderProducts(products, false);
    } catch {
        els.skeletonLoader.hidden = true;
        toast('Search failed', 'error');
    }
}

// ── Trending ─────────────────────────────────────────────────────────
export async function loadTrending(days = 7, limit = 10) {
    if (!els.trendingSection || !els.trendingGrid) return;
    els.trendingSection.hidden = true;
    els.trendingGrid.innerHTML = '';

    try {
        const data = await API.get(`/api/trending?days=${days}&limit=${limit}`);
        const items = data.results || [];
        if (!items.length) return;
        state.trending = items;
        renderTrending(items);
        els.trendingSection.hidden = false;
    } catch (err) {
        console.warn('Trending load failed:', err.message || err);
    }
}

export function renderTrending(items) {
    els.trendingGrid.innerHTML = '';
    const fragment = document.createDocumentFragment();

    items.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'product-card trending-card';
        card.style.animationDelay = `${index * 35}ms`;
        card.innerHTML = `
            <div class="product-card__image">${categoryIcon(item.category)}</div>
            <div class="product-card__body">
                ${item.category ? `<span class="product-card__category">${esc(item.category)}</span>` : ''}
                <h3 class="product-card__title">${esc(item.title || 'Untitled')}</h3>
                <p class="product-card__desc">${esc(item.description || 'No description available.')}</p>
                <div class="product-card__footer">
                    <div class="product-card__rating">
                        <div class="star-rating">${renderStars(item.rating || 0)}</div>
                        <span class="rating-value">${(item.rating || 0).toFixed(1)}</span>
                    </div>
                    ${sentimentBadge(item.avg_sentiment || 0)}
                </div>
            </div>
            <div class="product-card__actions">
                <button class="btn--add-cart" data-title="${esc(item.title)}">View Trending</button>
            </div>
        `;

        card.querySelector('.btn--add-cart').addEventListener('click', (e) => {
            e.stopPropagation();
            loadRecommendations(item.title);
            toast(`Showing recommendations for "${item.title.substring(0, 40)}"`, 'info');
        });

        card.addEventListener('click', () => loadRecommendations(item.title));
        fragment.appendChild(card);
    });

    els.trendingGrid.appendChild(fragment);
}

// ── Recommendations ──────────────────────────────────────────────────
function getRealtimeUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/recommendations`;
}

export function initRecommendationSocket() {
    if (!('WebSocket' in window) || state.recommendationSocket) return;

    const socket = new WebSocket(getRealtimeUrl());
    state.recommendationSocket = socket;

    socket.addEventListener('open', () => { state.realtimeReady = true; });
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
    socket.addEventListener('error', () => { state.realtimeReady = false; });
}

// FIX: debounced to avoid flooding the API on rapid calls
const debouncedFallback = debounce(async (title) => {
    try {
        const data = await API.post('/api/realtime/behavior', { item_title: title, top_n: 12 });
        renderRecommendations(data);
    } catch {
        await loadRecommendationsOverHttp(title);
    }
}, 250);

async function fallbackRecommendationRequest(title) {
    if (!title) return;
    debouncedFallback(title);
}

export function renderRecommendations(data) {
    const recs = data.recommendations || [];
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

    els.recsStrip.innerHTML = recs.map((r) => `
        <div class="rec-card" data-title="${esc(r.title)}" tabindex="0" role="button" aria-label="Get recommendations for ${esc(r.title)}">
            <div class="rec-card__title">${esc(r.title)}</div>
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

    els.recsStrip.querySelectorAll('.rec-card').forEach((card) => {
        card.addEventListener('click', () => loadRecommendations(card.dataset.title));
        // FIX: Space key preventDefault to stop page scroll
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                loadRecommendations(card.dataset.title);
            } else if (e.key === ' ') {
                e.preventDefault();
                loadRecommendations(card.dataset.title);
            }
        });
    });

    els.recsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadRecommendationsOverHttp(title) {
    const data = await API.get(`/api/recommend/${encodeURIComponent(title)}?top_n=12`);
    renderRecommendations(data);
}

export async function loadRecommendations(title) {
    if (!state.modelReady) {
        toast('Build models first to get recommendations', 'info');
        return;
    }

    els.recsSection.hidden = false;
    setPageMeta(`Recommendations for ${title}`, `Products similar to "${title}" using hybrid filtering.`);
    els.recsLoader.hidden = false;
    els.recsStrip.hidden = true;
    els.recsStrip.innerHTML = '';

    try {
        const data = await API.get(`/api/recommend?title=${encodeURIComponent(title)}&top_n=12`);
        const recs = data.recommendations || [];
        els.recsLoader.hidden = true;
        els.recsStrip.hidden = false;

        if (!recs.length) {
            els.recsStrip.innerHTML = `
                <div class="empty-recommendations">
                    <span class="empty-icon" aria-hidden="true">🔍</span>
                    <p>No recommendations found. Try a different product!</p>
                </div>
            `;
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

// ── Upload & Build ───────────────────────────────────────────────────
export async function handleUpload(file) {
    // FIX: case-insensitive file validation (DATA.CSV now works)
    if (!isValidUploadFile(file.name)) {
        toast('Only CSV and JSON files are supported', 'error');
        return;
    }

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

export async function handleBuild() {
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

// ── Status ───────────────────────────────────────────────────────────
export async function checkStatus() {
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

// ── Heatmap ──────────────────────────────────────────────────────────
export function updateCompareCount() {
    const count = state.heatmapSelected.length;
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

export async function loadHeatmap() {
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
        const itemsParam = state.heatmapSelected.map((t) => encodeURIComponent(t)).join(',');
        const data = await API.get(`/api/similarity-matrix?items=${itemsParam}`);
        els.heatmapLoader.hidden = true;
        if (data.not_found?.length) toast(`${data.not_found.length} item(s) not found in model`, 'info');
        renderHeatmap(data.labels, data.matrix);
        els.heatmapSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        els.heatmapLoader.hidden = true;
        els.heatmapContainer.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Could not compute similarity matrix.</div>';
        toast('Heatmap failed: ' + err.message, 'error');
    }
}

// ── Side-by-Side Compare ─────────────────────────────────────────────
export function toggleCompare(product, checked) {
    if (checked) {
        if (state.compareList.length >= 3) {
            toast('Maximum 3 products can be compared', 'error');
            return false;
        }
        if (!state.compareList.find((p) => p.title === product.title)) {
            state.compareList.push(product);
        }
    } else {
        state.compareList = state.compareList.filter((p) => p.title !== product.title);
    }
    updateCompareBar();
    return true;
}

export function removeFromCompare(title) {
    state.compareList = state.compareList.filter((p) => p.title !== title);
    document.querySelectorAll('.side-compare-checkbox').forEach((cb) => {
        if (cb.dataset.title === title) cb.checked = false;
    });
    updateCompareBar();
}

export function clearCompare() {
    state.compareList = [];
    document.querySelectorAll('.side-compare-checkbox').forEach((cb) => { cb.checked = false; });
    updateCompareBar();
}

// ── Infinite Scroll ──────────────────────────────────────────────────
export function setupScrollObserver() {
    destroyScrollObserver();
    if (!els.scrollSentinel) return;

    state.scrollObserver = new IntersectionObserver(
        (entries) => {
            const entry = entries[0];
            if (entry.isIntersecting && !state.isLoading && state.hasMore) {
                loadProducts(true);
            }
        },
        { rootMargin: '0px 0px 200px 0px', threshold: 0 }
    );

    state.scrollObserver.observe(els.scrollSentinel);
}

export function destroyScrollObserver() {
    if (state.scrollObserver) {
        state.scrollObserver.disconnect();
        state.scrollObserver = null;
    }
}

// ── Weights (debounced to avoid API flood on slider drag) ────────────
const debouncedWeightUpdate = debounce(async () => {
    const a = parseInt(els.weightAlpha.value);
    const b = parseInt(els.weightBeta.value);
    const g = parseInt(els.weightGamma.value);
    try {
        await API.put('/api/weights', { alpha: a / 100, beta: b / 100, gamma: g / 100 });
    } catch {}
}, 400);

export function handleWeightChange() {
    debouncedWeightUpdate();
}

// ── Reset Filters ────────────────────────────────────────────────────
export function resetAllFiltersAndSearch() {
    els.searchInput.value = '';
    els.categoryFilter.value = '';
    els.ratingFilter.value = '';
    els.sentimentFilter.value = '';
    state.filters = { category: '', rating: '', sentiment: '' };
    els.productsTitle.textContent = 'Top Products';
    savePreferences();
    loadProducts(false);
    setupScrollObserver();
}

// Expose globals needed by inline onclick handlers in compare bar HTML
window._removeFromCompare = removeFromCompare;
window._clearCompare = clearCompare;
window._openComparePage = openComparePage;
