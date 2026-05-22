import { state, els } from './state.js';
import { API } from './api.js';

// ── Search Dropdown ──────────────────────────────────────────────────
export function renderSearchDropdown(results, query) {
    if (!results.length) {
        closeSearchDropdown();
        return;
    }

    els.searchDropdown.innerHTML = results
        .map((title, index) => `
            <div
                class="search-result ${index === state.selectedSearchIdx ? 'active' : ''}"
                data-title="${title}"
                data-idx="${index}"
            >
                <span class="search-result__icon">🔍</span>
                <div class="search-result__info">
                    <div class="search-result__title">
                        ${highlightMatch(title, query)}
                    </div>
                </div>
            </div>
        `)
        .join('');

    els.searchDropdown.classList.add('active');

    els.searchDropdown.querySelectorAll('.search-result').forEach((el) => {
        el.addEventListener('click', () => {
            selectSearchResult(el.dataset.title);
        });
    });
}

function highlightMatch(text, query) {
    if (!query) return text;
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<strong>$1</strong>');
}

export function selectSearchResult(title) {
    els.searchInput.value = title;
    closeSearchDropdown();
    // Lazy import to avoid circular dependency with recommendations.js
    import('./recommendations.js').then(({ loadRecommendations, loadSearchResults }) => {
        loadSearchResults(title);
        loadRecommendations(title);
    });
}

export function closeSearchDropdown() {
    els.searchDropdown.classList.remove('active');
    state.selectedSearchIdx = -1;
}

// ── Keyboard Handling ────────────────────────────────────────────────
export function handleSearchKeydown(e) {
    const results = state.autocompleteResults;

    if (e.key === 'Enter') {
        e.preventDefault();
        if (
            state.selectedSearchIdx >= 0 &&
            results.length &&
            els.searchDropdown.classList.contains('active')
        ) {
            selectSearchResult(results[state.selectedSearchIdx]);
        } else if (els.searchInput.value.trim().length > 0) {
            selectSearchResult(els.searchInput.value.trim());
        }
        return;
    }

    if (e.key === 'Escape') {
        closeSearchDropdown();
        return;
    }

    if (!results.length || !els.searchDropdown.classList.contains('active')) return;

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        state.selectedSearchIdx = Math.min(state.selectedSearchIdx + 1, results.length - 1);
        renderSearchDropdown(results, els.searchInput.value);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        state.selectedSearchIdx = Math.max(state.selectedSearchIdx - 1, 0);
        renderSearchDropdown(results, els.searchInput.value);
    }
}

// ── Debounced Search Input ───────────────────────────────────────────
export function handleSearch(query) {
    if (!query || query.trim().length < 1) {
        closeSearchDropdown();
        return;
    }

    clearTimeout(state.searchTimer);

    state.searchTimer = setTimeout(async () => {
        try {
            const data = await API.get(
                `/api/autocomplete?q=${encodeURIComponent(query)}&limit=5`
            );
            state.autocompleteResults = data.suggestions || [];
            state.selectedSearchIdx = -1;
            renderSearchDropdown(state.autocompleteResults, query);
        } catch (err) {
            console.error('Autocomplete failed:', err);
            closeSearchDropdown();
        }
    }, 300);
}

// ── Type-to-Search (Global / Shortcut) ──────────────────────────────
export function initTypeToSearch() {
    document.addEventListener('keydown', (e) => {
        const tag = document.activeElement?.tagName;
        const isTypingField =
            tag === 'INPUT' ||
            tag === 'TEXTAREA' ||
            tag === 'SELECT' ||
            document.activeElement?.isContentEditable;

        if (isTypingField) return;
        if (e.ctrlKey || e.altKey || e.metaKey) return;
        if (e.key !== '/') return;

        e.preventDefault();
        els.searchInput.focus();
    });

    // Close dropdown on outside click
    window.addEventListener('click', (e) => {
        const container = document.getElementById('search-container');
        if (!container.contains(e.target)) closeSearchDropdown();
    });
}
