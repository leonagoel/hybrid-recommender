-- =============================================================================
-- Extension & Indexing Setup (Tasks 1 & 2)
-- =============================================================================

-- Task 1: Enable pg_trgm extension in Supabase
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Task 2: Add GIN index on product titles for fast similarity lookups
CREATE INDEX IF NOT EXISTS idx_products_title_trgm 
    ON products USING gin (title pg_trgm_ops);


-- =============================================================================
-- Task 3: Create Fuzzy Search Function (/api/search/fuzzy)
-- =============================================================================
CREATE OR REPLACE FUNCTION fuzzy_search_products(q TEXT, threshold FLOAT DEFAULT 0.3)
RETURNS SETOF products 
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Set the local similarity threshold dynamically
    PERFORM set_limit(threshold);
    
    RETURN QUERY
    SELECT * FROM products
    WHERE title % q
    ORDER BY similarity(title, q) DESC;
END;
$$;


-- =============================================================================
-- Task 4: Create Search Function with Automatic Fallback Pipeline
-- =============================================================================
CREATE OR REPLACE FUNCTION standard_search_products(q TEXT)
RETURNS SETOF products 
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result_count INTEGER;
BEGIN
    -- 1. Attempt standard Full-Text Search (FTS) first
    -- This returns rows matching the plain text query syntax
    RETURN QUERY
    SELECT * FROM products
    WHERE to_tsvector('english', title) @@ plainto_tsquery('english', q);
    
    -- Check how many rows the standard FTS just returned
    GET DIAGNOSTICS result_count = ROW_COUNT;
    
    -- 2. Fallback Logic: If FTS returns less than 3 results, execute the fuzzy search
    IF result_count < 3 THEN
        RETURN QUERY
        SELECT * FROM fuzzy_search_products(q, 0.3);
    END IF;
END;
$$;