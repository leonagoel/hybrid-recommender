"""
Supabase client singleton.
Provides a single reusable client instance for the entire application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

_client = None
_admin_client = None


def get_supabase():
    """Get the Supabase client (uses anon key — respects RLS)."""
    global _client
    if _client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        _client = create_client(url, key)
    return _client


def get_supabase_admin():
    """Get the Supabase admin client (uses service role key — bypasses RLS)."""
    global _admin_client
    if _admin_client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        _admin_client = create_client(url, key)
    return _admin_client
