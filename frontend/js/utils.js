// frontend/js/utils.js
export function getStars(rating) {
    if (!rating && rating !== 0) return '☆☆☆☆☆';
    const full = Math.round(rating);
    const empty = 5 - full;
    return '★'.repeat(full) + '☆'.repeat(empty);
}