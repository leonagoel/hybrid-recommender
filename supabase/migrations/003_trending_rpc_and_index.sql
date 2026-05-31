-- =============================================================================
-- Migration: Add get_trending_products RPC and supporting index
--
-- This pushes purchase aggregation (GROUP BY, COUNT, AVG) down to PostgreSQL
-- so only the final top-N ranked rows are transferred over the network.
-- Without this, the trending endpoint fetched every purchase row in the window
-- and grouped them entirely in Python memory, causing OOM at scale.
-- =============================================================================

-- Index to make the purchased_at date-range filter efficient.
-- The trending query filters purchases by purchased_at >= cutoff_date; without
-- this index PostgreSQL performs a sequential scan of the entire purchases table
-- on every cache miss.
CREATE INDEX IF NOT EXISTS idx_purchases_purchased_at
    ON purchases (purchased_at);

-- Index to speed up the JOIN between purchases and products.
CREATE INDEX IF NOT EXISTS idx_purchases_product_id
    ON purchases (product_id);

-- RPC called by the trending endpoint.
--
-- Returns one row per product aggregated over purchases within the time window.
-- The caller requests limit_n * 3 rows so the Python layer has headroom for
-- Bayesian re-ranking before trimming to the final limit.
--
-- STABLE means PostgreSQL may cache the result within a single transaction;
-- it is correct here because trending results change only when new purchases
-- are inserted, not within a single request.
CREATE OR REPLACE FUNCTION get_trending_products(
    cutoff_date TIMESTAMPTZ,
    limit_n     INT
)
RETURNS TABLE (
    product_id     BIGINT,
    purchase_count BIGINT,
    avg_rating     NUMERIC,
    title          TEXT,
    category       TEXT,
    rating         DOUBLE PRECISION,
    avg_sentiment  DOUBLE PRECISION,
    review_count   INT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        p.product_id,
        COUNT(*)                    AS purchase_count,
        AVG(p.rating)               AS avg_rating,
        pr.title,
        pr.category,
        pr.rating,
        pr.avg_sentiment,
        pr.review_count
    FROM purchases p
    JOIN products pr ON pr.id = p.product_id
    WHERE p.purchased_at >= cutoff_date
    GROUP BY
        p.product_id,
        pr.title,
        pr.category,
        pr.rating,
        pr.avg_sentiment,
        pr.review_count
    ORDER BY purchase_count DESC
    LIMIT limit_n;
$$;

-- Grant execute to the anon and authenticated roles used by Supabase PostgREST.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        GRANT EXECUTE ON FUNCTION get_trending_products(TIMESTAMPTZ, INT) TO anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        GRANT EXECUTE ON FUNCTION get_trending_products(TIMESTAMPTZ, INT) TO authenticated;
    END IF;
END
$$;
