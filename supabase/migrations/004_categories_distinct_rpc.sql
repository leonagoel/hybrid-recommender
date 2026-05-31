-- =============================================================================
-- Migration: Add get_distinct_categories RPC and supporting index
--
-- Previously the /api/categories endpoint fell back to fetching up to 5 000
-- full product rows and deduplicating in Python memory. At 25 000+ products
-- this transfers ~25× more data than necessary, and the hard limit silently
-- truncated the category list on larger catalogues.
--
-- This migration adds:
--   1. An index on products(category) to make DISTINCT fast.
--   2. A get_distinct_categories() function that returns exactly one row per
--      unique, non-empty category — payload size is O(distinct categories),
--      not O(product table size).
-- =============================================================================

-- Index to make the DISTINCT category scan efficient.
-- Without this, the function performs a sequential scan of the entire
-- products table on every cache miss.
CREATE INDEX IF NOT EXISTS idx_products_category
    ON products (category);

-- RPC called by the /api/categories endpoint.
--
-- Returns one row per distinct, non-empty category, sorted alphabetically.
-- STABLE tells PostgreSQL the result does not change within a transaction,
-- which is correct since categories only change on data upload.
CREATE OR REPLACE FUNCTION get_distinct_categories()
RETURNS TABLE (category TEXT)
LANGUAGE sql
STABLE
AS $$
    SELECT DISTINCT category
    FROM products
    WHERE category IS NOT NULL
      AND category <> ''
    ORDER BY category;
$$;

-- Grant execute to the roles used by Supabase PostgREST.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        GRANT EXECUTE ON FUNCTION get_distinct_categories() TO anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        GRANT EXECUTE ON FUNCTION get_distinct_categories() TO authenticated;
    END IF;
END
$$;
