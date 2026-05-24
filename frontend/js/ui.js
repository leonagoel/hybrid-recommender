import { state, els } from './state.js';

// ── Toast ────────────────────────────────────────────────────────────
export function toast(message, type = 'info') {
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

// ── HTML Escaping (centralised — used by ui.js and recommendations.js) ──
export function esc(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Stars & Badges ───────────────────────────────────────────────────
export function renderStars(rating) {
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

export function sentimentBadge(score) {
    if (score > 0.05) return '<span class="product-card__sentiment sentiment-positive">Positive</span>';
    if (score < -0.05) return '<span class="product-card__sentiment sentiment-negative">Negative</span>';
    return '<span class="product-card__sentiment sentiment-neutral">Neutral</span>';
}

export function categoryIcon(cat) {
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

// ── Skeleton Cards ───────────────────────────────────────────────────
export function createSkeletonCard() {
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

export function showSkeletons(container, count = 8) {
    container.innerHTML = Array(count).fill('').map(() => createSkeletonCard()).join('');
}

// ── Page Meta ────────────────────────────────────────────────────────
export function setPageMeta(title, description) {
    document.title = title
        ? `${title} — HybridRec`
        : 'HybridRec — Smart Recommendations';
    const metaDesc = document.querySelector('meta[name="description"]');
    if (metaDesc && description) metaDesc.setAttribute('content', description);
}

// ── Filters ──────────────────────────────────────────────────────────
export function applyFilters(products) {
    return products.filter((p) => {
        const matchesCategory =
            !state.filters.category || p.category === state.filters.category;

        const matchesRating =
            !state.filters.rating || (p.rating || 0) >= Number(state.filters.rating);

        let sentiment = 'neutral';
        if ((p.avg_sentiment || 0) > 0.05) sentiment = 'positive';
        else if ((p.avg_sentiment || 0) < -0.05) sentiment = 'negative';

        const matchesSentiment =
            !state.filters.sentiment || sentiment === state.filters.sentiment;

        return matchesCategory && matchesRating && matchesSentiment;
    });
}

export function populateCategoryFilter(products) {
    const categories = [...new Set(products.map((p) => p.category).filter(Boolean))];
    els.categoryFilter.innerHTML = `
        <option value="">All Categories</option>
        ${categories.map((cat) => `<option value="${esc(cat)}">${esc(cat)}</option>`).join('')}
    `;
}

export function loadPreferences() {
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

export function savePreferences() {
    const prefs = {
        category: state.filters.category,
        rating: state.filters.rating,
        sentiment: state.filters.sentiment,
    };
    localStorage.setItem('userPreferences', JSON.stringify(prefs));
}

// ── Wishlist ─────────────────────────────────────────────────────────
export function getWishlist() {
    return JSON.parse(localStorage.getItem('wishlist')) || [];
}

export function saveWishlist(items) {
    localStorage.setItem('wishlist', JSON.stringify(items));
}

export function isWishlisted(title) {
    return getWishlist().some((item) => item.title === title);
}

// ── Status ───────────────────────────────────────────────────────────
export function updateStatus(cls, text) {
    els.statusDot.className = `status-dot ${cls}`;
    els.statusText.textContent = text;
}

// ── Heatmap ──────────────────────────────────────────────────────────
export function renderHeatmap(labels, matrix) {
    const n = labels.length;
    const shortLabels = labels.map((l) => (l.length > 25 ? l.substring(0, 22) + '…' : l));

    let html = `<div class="heatmap-grid" style="grid-template-columns: 140px repeat(${n}, 1fr); grid-template-rows: auto repeat(${n}, 1fr);">`;
    html += '<div class="heatmap-cell heatmap-corner"></div>';

    for (let j = 0; j < n; j++) {
        html += `<div class="heatmap-cell heatmap-col-label" title="${esc(labels[j])}">${esc(shortLabels[j])}</div>`;
    }

    for (let i = 0; i < n; i++) {
        html += `<div class="heatmap-cell heatmap-row-label" title="${esc(labels[i])}">${esc(shortLabels[i])}</div>`;
        for (let j = 0; j < n; j++) {
            const score = matrix[i][j];
            const r = Math.round(255 - score * 200);
            const g = Math.round(255 - score * 55);
            const b = Math.round(255 - score * 200);
            const bg = `rgb(${r}, ${g}, ${b})`;
            const textColor = score > 0.6 ? '#fff' : 'var(--text)';
            html += `<div class="heatmap-cell heatmap-value" style="background:${bg};color:${textColor};"
                title="${esc(labels[i])} × ${esc(labels[j])}: ${score.toFixed(4)}">
                ${score === 1 ? '1.0' : score.toFixed(2)}
            </div>`;
        }
    }

    html += '</div>';
    els.heatmapContainer.innerHTML = html;
}

// ── Compare Bar (Side-by-Side) ───────────────────────────────────────
export function updateCompareBar() {
    let bar = document.getElementById('compare-bar');
    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'compare-bar';
        bar.className = 'compare-bar';
        document.body.appendChild(bar);
    }
    if (state.compareList.length === 0) {
        bar.hidden = true;
        return;
    }
    bar.hidden = false;
    bar.innerHTML = `
        <div class="compare-bar__items">
            ${state.compareList.map((p) => `
                <div class="compare-bar__item">
                    <span>${esc(p.title.substring(0, 25))}${p.title.length > 25 ? '...' : ''}</span>
                    <button onclick="window._removeFromCompare('${esc(p.title.replace(/'/g, "\\'"))}')">✕</button>
                </div>
            `).join('')}
        </div>
        <div class="compare-bar__actions">
            <span class="compare-bar__count">${state.compareList.length}/3 selected</span>
            <button class="compare-bar__btn" onclick="window._openComparePage()"
                ${state.compareList.length < 2 ? 'disabled' : ''}>
                Compare Now
            </button>
            <button class="compare-bar__clear" onclick="window._clearCompare()">Clear</button>
        </div>
    `;
}

export function openComparePage() {
    if (state.compareList.length < 2) {
        toast('Select at least 2 products to compare', 'info');
        return;
    }

    let modal = document.getElementById('compare-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'compare-modal';
        modal.className = 'modal-overlay';
        document.body.appendChild(modal);
    }

    const products = state.compareList;
    modal.innerHTML = `
        <div class="modal" style="max-width:900px;width:95%;">
            <button class="modal__close" onclick="document.getElementById('compare-modal').hidden=true">&times;</button>
            <h2 class="modal__title">Product Comparison</h2>
            <div style="overflow-x:auto;margin-top:16px;">
                <table class="compare-table">
                    <thead>
                        <tr>
                            <th style="min-width:120px;">Attribute</th>
                            ${products.map((p) => `<th style="min-width:180px;">${esc(p.title)}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Category</strong></td>
                            ${products.map((p) => `<td>${esc(p.category || 'N/A')}</td>`).join('')}
                        </tr>
                        <tr>
                            <td><strong>Rating</strong></td>
                            ${products.map((p) => `<td>⭐ ${(p.rating || 0).toFixed(1)}</td>`).join('')}
                        </tr>
                        <tr>
                            <td><strong>Sentiment</strong></td>
                            ${products.map((p) => {
                                const s = p.avg_sentiment || 0;
                                const label = s > 0.05 ? '😊 Positive' : s < -0.05 ? '😞 Negative' : '😐 Neutral';
                                return `<td>${label}</td>`;
                            }).join('')}
                        </tr>
                        <tr>
                            <td><strong>Description</strong></td>
                            ${products.map((p) => `<td style="font-size:12px;">${esc((p.description || 'N/A').substring(0, 100))}...</td>`).join('')}
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;

    modal.hidden = false;
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.hidden = true;
    });
}

// ── Back To Top ──────────────────────────────────────────────────────
export function initBackToTop() {
    const backToTop = document.getElementById('backToTop');
    if (!backToTop) return;
    backToTop.style.display = 'none';
    window.addEventListener('scroll', () => {
        backToTop.style.display = window.scrollY > 700 ? 'block' : 'none';
    });
    backToTop.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// ── Theme Toggle ─────────────────────────────────────────────────────
export function initTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;
    const root = document.documentElement;
    const savedTheme = localStorage.getItem('theme') || 'dark';
    root.setAttribute('data-theme', savedTheme);
    themeToggle.textContent = savedTheme === 'dark' ? '🌙' : '☀️';

    themeToggle.addEventListener('click', () => {
        const current = root.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        themeToggle.textContent = next === 'dark' ? '🌙' : '☀️';
    });
}

// ── CSS Spin Animation ───────────────────────────────────────────────
export function injectSpinStyle() {
    const spinStyle = document.createElement('style');
    spinStyle.textContent = `@keyframes spin { to { transform: rotate(360deg); } } .spin { animation: spin 1s linear infinite; }`;
    document.head.appendChild(spinStyle);
}

// ── Debounce Helper ──────────────────────────────────────────────────
export function debounce(func, delay) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
}

// ── Upload file validation (case-insensitive) ────────────────────────
export function isValidUploadFile(filename) {
    // FIX: toLowerCase() ensures DATA.CSV and data.csv both pass
    const lower = filename.toLowerCase();
    return lower.endsWith('.csv') || lower.endsWith('.json');
}
