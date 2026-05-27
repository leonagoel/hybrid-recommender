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
        with _admin_client_lock:
            if _admin_client is None:  # re-check after acquiring lock
                url = os.environ.get("SUPABASE_URL", "")
                key = os.environ.get("SUPABASE_SERVICE_KEY", "")
                if not url or not key:
                    raise RuntimeError(
                        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
                    )
                _admin_client = create_client(url, key)
                logger.info("Supabase admin client initialised.")
    return _admin_client
