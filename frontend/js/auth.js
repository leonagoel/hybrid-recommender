import { state, els } from './state.js';
import { toast } from './ui.js';

// ── Supabase Client ─────────────────────────────────────────────────
export let sbClient = null;

export async function initSupabase() {
    try {
        const resp = await fetch('/api/config');
        if (!resp.ok) return null;
        const config = await resp.json();
        const { createClient } = window.supabase || {};
        if (createClient && config.supabase_url && config.supabase_anon_key) {
            sbClient = createClient(config.supabase_url, config.supabase_anon_key);
        }
    } catch (e) {
        console.warn('Supabase init skipped:', e.message);
    }
    return sbClient;
}

// ── Auth ────────────────────────────────────────────────────────────
export async function initAuth() {
    if (!sbClient) {
        console.warn('Supabase client unavailable — auth disabled');
        els.authLabel.textContent = 'Sign In';
        return;
    }
    try {
        const { data: { session } } = await sbClient.auth.getSession();

        if (session) {
            setUser(session.user);
        } else {
            const { data, error } = await sbClient.auth.signInAnonymously();
            if (error) {
                console.warn('Guest login failed:', error.message);
                els.authLabel.textContent = 'Sign In';
            } else {
                setUser(data.user);
            }
        }
    } catch (err) {
        console.warn('Auth init failed:', err.message);
        els.authLabel.textContent = 'Sign In';
    }
}

export function setUser(user) {
    state.user = user;
    state.isGuest = user?.is_anonymous || !user?.email;

    if (state.isGuest) {
        els.authLabel.textContent = 'Guest';
    } else {
        els.authLabel.textContent = user.email?.split('@')[0] || 'User';
    }
}

export async function handleAuth(e) {
    e.preventDefault();
    els.authError.hidden = true;
    els.authSubmit.disabled = true;
    els.authSubmit.textContent = 'Please wait...';

    const email = els.authEmail.value.trim();
    const password = els.authPassword.value;

    try {
        let result;
        if (state.isAuthSignUp) {
            result = await sbClient.auth.signUp({
                email,
                password,
                options: { data: { display_name: email.split('@')[0] } },
            });
        } else {
            result = await sbClient.auth.signInWithPassword({ email, password });
        }

        if (result.error) throw result.error;

        setUser(result.data.user);
        els.authModal.hidden = true;
        toast(state.isAuthSignUp ? 'Account created!' : 'Signed in!', 'success');
    } catch (err) {
        els.authError.textContent = err.message;
        els.authError.hidden = false;
    } finally {
        els.authSubmit.disabled = false;
        els.authSubmit.textContent = state.isAuthSignUp ? 'Sign Up' : 'Sign In';
    }
}

export function toggleAuthMode() {
    state.isAuthSignUp = !state.isAuthSignUp;
    els.modalTitle.textContent = state.isAuthSignUp ? 'Create Account' : 'Sign In';
    els.authSubmit.textContent = state.isAuthSignUp ? 'Sign Up' : 'Sign In';
    els.authToggleText.textContent = state.isAuthSignUp ? 'Already have an account?' : "Don't have an account?";
    els.authToggleBtn.textContent = state.isAuthSignUp ? 'Sign In' : 'Sign Up';
    els.authError.hidden = true;
}
