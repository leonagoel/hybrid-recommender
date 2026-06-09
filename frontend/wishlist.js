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

    // Reset container safely
    grid.innerHTML = '';

    if (!wishlist.length) {
        grid.innerHTML = '<p>No saved products yet.</p>';
        return;
    }

    // Securely generate cards using DOM methods instead of vulnerable strings
    wishlist.forEach(p => {
        // 1. Root Card Container
        const card = document.createElement('div');
        card.className = 'product-card';

        // 2. Visual Box Node
        const imageDiv = document.createElement('div');
        imageDiv.className = 'product-card__image';
        imageDiv.textContent = '📦';
        card.appendChild(imageDiv);

        // 3. Inner Content Body Node
        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'product-card__body';

        // 4. Secure Title: Handled entirely via textContent to block payload compilation
        const h3 = document.createElement('h3');
        h3.className = 'product-card__title';
        h3.textContent = p.title ?? 'Unknown Product';
        bodyDiv.appendChild(h3);

        // 5. Secure Description Handling
        const pDesc = document.createElement('p');
        pDesc.className = 'product-card__desc';
        pDesc.textContent = p.description || 'No description';
        bodyDiv.appendChild(pDesc);

        // 6. FIX SECURITY BUG: Safe explicit event binding context loop mapping
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        
        removeButton.addEventListener('click', () => {
            removeFromWishlist(p.title);
        });
        
        bodyDiv.appendChild(removeButton);
        card.appendChild(bodyDiv);
        
        // Push secure container tree to active page layout grid
        grid.appendChild(card);
    });
}

renderWishlist();
