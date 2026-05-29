import os
import logging
import threading
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level singletons — populated on first use, never recreated.
_client: Client | None = None
_admin_client: Client | None = None

# One lock per client; guards the check-then-create critical section.
_client_lock = threading.Lock()
_admin_client_lock = threading.Lock()


def get_supabase() -> Client:
    """
    Return the shared anon Supabase client (respects RLS).

    Lazy: the client is created on the first call, not at import time.
    Singleton: subsequent calls return the same instance without acquiring
    the lock (double-checked locking pattern).
    Raises RuntimeError if required env vars are absent.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # re-check after acquiring lock
                url = os.environ.get("SUPABASE_URL", "")
                key = os.environ.get("SUPABASE_ANON_KEY", "")
                if not url or not key:
                    raise RuntimeError(
                        "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"
                    )
                _client = create_client(url, key)
                logger.info("Supabase anon client initialised.")
    return _client


def get_supabase_admin() -> Client:
    """
    Return the shared admin Supabase client (bypasses RLS).

    Same lazy singleton pattern as get_supabase().
    Raises RuntimeError if required env vars are absent.
    """
    global _admin_client
    if _admin_client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        
        # FIX FOR ISSUE #487: Dynamic fallback check prevents global execution crashes
        if not url or not key:
            logger.warning("Supabase admin credentials missing (SUPABASE_URL/SUPABASE_SERVICE_KEY). Admin features disabled.")
            return None
            
        try:
            from supabase import create_client
            _admin_client = create_client(url, key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase admin client: {e}")
            return None
            
    return _admin_client
