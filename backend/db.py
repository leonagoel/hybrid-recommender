import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")

# Initialize clients only if credentials are available
supabase: Client = None
supabase_admin: Client = None

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client initialized")
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE:
        supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
        logger.info("✅ Supabase admin client initialized")
    if not (SUPABASE_URL and SUPABASE_KEY):
        logger.warning("⚠️  Supabase credentials not found. Using mock data mode.")
except Exception as e:
    logger.warning(f"⚠️  Failed to connect to Supabase: {e}. Using mock data mode.")

def get_supabase():
    """Get public Supabase client. Returns None in mock mode."""
    return supabase

def get_supabase_admin():
    """Get admin Supabase client. Returns None in mock mode."""
    return supabase_admin