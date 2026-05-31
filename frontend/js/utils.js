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
    if (!url || typeof url !== 'string') {
        return false;
    }

    // Allow only safe internal relative paths
    if (url.startsWith('/') && !url.startsWith('//')) {
        return true;
    }

    // Allow same-origin absolute URLs only
    try {
        const target = new URL(url, window.location.origin);
        return target.origin === window.location.origin;
    } catch {
        return false;
    }
}
/**
 * Securely merges objects to prevent prototype pollution.
 * @param {Object} target
 * @param {Object} source
 * @returns {Object}
 */
export function safeMerge(target, source) {
    for (const key in source) {
        if (!Object.prototype.hasOwnProperty.call(source, key)) continue;
        if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue;

        const sourceVal = source[key];
        const targetVal = target[key];

        if (typeof sourceVal === 'object' && sourceVal !== null && !Array.isArray(sourceVal)) {
            if (typeof targetVal !== 'object' || targetVal === null || Array.isArray(targetVal)) {
                target[key] = {};
            }
            safeMerge(target[key], sourceVal);
        } else {
            target[key] = sourceVal;
        }
    }
    return target;
}
