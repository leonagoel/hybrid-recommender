// =============================================================================
// app.js — Main Entry Point (imports only)
// All business logic lives in frontend/js/*.
// This file: fetch config → init Supabase → wire modules together.
// =============================================================================

import { initAuth, bindAuthEvents, signInAsGuest } from './js/auth.js';
import { initSearch, loadProducts, loadCategories }  from './js/search.js';
import { initWeightSliders, renderRecentlyViewed }   from './js/recommendations.js';
import {
  initModalDismiss,
  bindUploadHandler,
  bindBuildModelsHandler,
  startStatusPoller,
} from './js/ui.js';
import { setState, subscribe } from './js/state.js';

async function bootstrap() {
  // 1. Fetch Supabase config from backend
  let supabaseClient = null;
  try {
    const res    = await fetch('/api/config');
    const config = await res.json();
    supabaseClient = window.supabase.createClient(
      config.supabase_url,
      config.supabase_anon_key
    );
  } catch (err) {
    console.warn('[app] Config fetch failed — running offline.', err);
  }

  // 2. Auth
  if (supabaseClient) {
    initAuth(supabaseClient);
    bindAuthEvents();
    const { data: { session } } = await supabaseClient.auth.getSession();
    if (!session) await signInAsGuest();
  }

  // 3. UI infrastructure
  initModalDismiss();
  startStatusPoller(30_000);

  // 4. Dataset management
  bindUploadHandler((data) => {
    setState({ datasetLoaded: true, productCount: data.rows ?? 0 });
    loadProducts(1);
    loadCategories();
  });
  bindBuildModelsHandler(() => setState({ modelsBuilt: true }));

  // 5. Search & browse
  initSearch();
  loadCategories();
  loadProducts(1);

  // 6. Recommendation weight sliders
  initWeightSliders();

  // 7. React to state changes
  subscribe('recentlyViewed', renderRecentlyViewed);

  console.log('[HybridRec] ✓ All modules loaded.');
}

document.readyState === 'loading'
  ? document.addEventListener('DOMContentLoaded', bootstrap)
  : bootstrap();
