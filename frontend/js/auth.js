// =============================================================================
// auth.js — Authentication Logic
// Supabase guest (anonymous) + email/password flows.
// =============================================================================

import { state, setState, getAnonymousUserId } from './state.js';
import { showToast, showModal, hideModal, setLoadingState } from './ui.js';

let _supabase = null;

/** Called once from app.js after Supabase client is created. */
export function initAuth(supabaseClient) {
  _supabase = supabaseClient;

  _supabase.auth.onAuthStateChange((event, session) => {
    const user = session?.user ?? null;
    setState({ user, session, isGuest: user?.is_anonymous ?? false });
    _syncAuthUI(user);
  });
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function signInAsGuest() {
  try {
    setLoadingState('auth', true);
    const { data, error } = await _supabase.auth.signInAnonymously();
    if (error) throw error;
    showToast('Browsing as guest', 'info');
    return data;
  } catch (err) {
    showToast(`Guest sign-in failed: ${err.message}`, 'error');
  } finally {
    setLoadingState('auth', false);
  }
}

export async function signInWithEmail(email, password) {
  try {
    setLoadingState('auth', true);
    const { data, error } = await _supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
    hideModal('auth-modal');
    showToast(`Welcome back, ${data.user.email}!`, 'success');
    return data;
  } catch (err) {
    showToast(`Sign-in failed: ${err.message}`, 'error');
  } finally {
    setLoadingState('auth', false);
  }
}

export async function signUpWithEmail(email, password) {
  try {
    setLoadingState('auth', true);
    const guestId = getAnonymousUserId();
    const { data, error } = await _supabase.auth.signUp({ email, password });
    if (error) throw error;

    if (data?.user?.id && guestId && guestId !== data.user.id) {
      try {
        const csrfToken = _getCookie('csrftoken');
        const headers = { 'Content-Type': 'application/json' };
        if (csrfToken) {
          headers['X-CSRF-Token'] = csrfToken;
        }
        await fetch('/api/register', {
          method: 'POST',
          headers: headers,
          body: JSON.stringify({
            guest_id: guestId,
            user_id: data.user.id,
          }),
        });
      } catch (mergeErr) {
        console.warn('Failed to merge guest history on register:', mergeErr);
      }
    }

    hideModal('auth-modal');
    showToast('Account created! Check your email to confirm.', 'success');
    return data;
  } catch (err) {
    showToast(`Sign-up failed: ${err.message}`, 'error');
  } finally {
    setLoadingState('auth', false);
  }
}

export async function signOut() {
  try {
    const { error } = await _supabase.auth.signOut();
    if (error) throw error;
    showToast('Signed out successfully.', 'info');
  } catch (err) {
    showToast(`Sign-out failed: ${err.message}`, 'error');
  }
}

export function isAuthenticated() {
  return !!state.user && !state.isGuest;
}

// ── Internal ──────────────────────────────────────────────────────────────────

function _syncAuthUI(user) {
  const authBtn    = document.getElementById('auth-btn');
  const signOutBtn = document.getElementById('signout-btn');
  const userLabel  = document.getElementById('user-label');
  if (!authBtn) return;

  if (user && !state.isGuest) {
    // Fully logged in
    authBtn.style.display    = 'none';
    signOutBtn.style.display = 'inline-flex';
    if (userLabel) userLabel.textContent = user.email;
  } else if (user && state.isGuest) {
    // Guest session
    authBtn.style.display    = 'inline-flex';
    authBtn.textContent      = 'Sign In';
    signOutBtn.style.display = 'none';
    if (userLabel) userLabel.textContent = 'Guest';
  } else {
    // Logged out
    authBtn.style.display    = 'inline-flex';
    authBtn.textContent      = 'Sign In';
    signOutBtn.style.display = 'none';
    if (userLabel) userLabel.textContent = '';
  }
}

/** Bind all auth-related DOM events. Called once from app.js. */
export function bindAuthEvents() {
  document.getElementById('auth-btn')
    ?.addEventListener('click', () => showModal('auth-modal'));

  document.getElementById('signout-btn')
    ?.addEventListener('click', signOut);

  document.getElementById('auth-form')
    ?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email    = document.getElementById('auth-email')?.value.trim();
      const password = document.getElementById('auth-password')?.value;
      const isSignUp = document.getElementById('auth-mode')?.dataset.mode === 'signup';
      isSignUp
        ? await signUpWithEmail(email, password)
        : await signInWithEmail(email, password);
    });

  document.getElementById('auth-toggle')
    ?.addEventListener('click', () => {
      const modeEl = document.getElementById('auth-mode');
      if (!modeEl) return;
      const next = modeEl.dataset.mode === 'signin' ? 'signup' : 'signin';
      modeEl.dataset.mode = next;
      modeEl.textContent  = next === 'signup' ? 'Create Account' : 'Sign In';
      const toggle = document.getElementById('auth-toggle');
      if (toggle) toggle.textContent = next === 'signup'
        ? 'Already have an account? Sign in'
        : "Don't have an account? Sign up";
    });
}

function _getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return '';
}