// frontend/js/utils.js
export function getStars(rating) {
    if (!rating && rating !== 0) return '☆☆☆☆☆';
    const full = Math.round(rating);
    const empty = 5 - full;
    return '★'.repeat(full) + '☆'.repeat(empty);
}

/**
 * Check if a redirect URL is safe (relative path or same origin).
 * @param {string} url - The URL to validate.
 * @returns {boolean}
 */
export function isSafeRedirect(url) {
    if (!url) return false;
    // Allow relative URLs (starts with '/')
    if (url.startsWith('/')) return true;
    // Allow same‑origin absolute URLs
    try {
        const target = new URL(url, window.location.origin);
        return target.origin === window.location.origin;
    } catch {
        return false;
    }
}