import os
import logging
from dotenv import load_dotenv
load_dotenv()   # loads .env file

from supabase import create_client, Client

logger = logging.getLogger(__name__)

_client = None
_admin_client = None


def get_supabase():
    """
    Get the Supabase client (uses anon key — respects RLS).
    Returns None gracefully if environment keys are missing to prevent import-time crashes.
    """
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")
        
        # FIX FOR ISSUE #487: Dynamic fallback check prevents global execution crashes
        if not url or not key:
            logger.warning("Supabase environment variables missing (SUPABASE_URL/SUPABASE_ANON_KEY). Running in offline mode.")
            return None
            
        try:
            from supabase import create_client
            _client = create_client(url, key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            return None
            
    return _client


def get_supabase_admin():
    """
    Get the Supabase admin client (uses service role key — bypasses RLS).
    Returns None gracefully if environment keys are missing to prevent import-time crashes.
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
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    return create_client(url, key)

def get_supabase_admin() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    return create_client(url, key)