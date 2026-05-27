function getWishlist() {
    return JSON.parse(localStorage.getItem('wishlist')) || [];
}

function removeFromWishlist(title) {
    let wishlist = getWishlist();

    wishlist = wishlist.filter(item => item.title !== title);

    localStorage.setItem('wishlist', JSON.stringify(wishlist));

    renderWishlist();
}

function renderWishlist() {
    const grid = document.getElementById('wishlist-grid');

    const wishlist = getWishlist();

    if (!wishlist.length) {
        grid.innerHTML = '<p>No saved products yet.</p>';
        return;
    }

    grid.innerHTML = wishlist.map(p => `
        <div class="product-card">
            <div class="product-card__image">
                📦
            </div>

            <div class="product-card__body">
                <h3 class="product-card__title">${p.title}</h3>

                <p class="product-card__desc">
                    ${p.description || 'No description'}
                </p>

                <button onclick="removeFromWishlist('${p.title}')">
                    Remove
                </button>
            </div>
        </div>
    `).join('');
}

renderWishlist();