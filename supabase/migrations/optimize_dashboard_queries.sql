CREATE OR REPLACE FUNCTION get_total_users()
RETURNS integer
LANGUAGE sql
AS $$
    SELECT COUNT(DISTINCT user_id)::integer
    FROM purchases
    WHERE user_id IS NOT NULL;
$$;


CREATE OR REPLACE FUNCTION get_top_product_counts()
RETURNS TABLE (
    product_id bigint,
    interaction_count bigint
)
LANGUAGE sql
AS $$
    SELECT
        product_id,
        COUNT(*) AS interaction_count
    FROM purchases
    WHERE product_id IS NOT NULL
    GROUP BY product_id
    ORDER BY interaction_count DESC
    LIMIT 5;
$$;

CREATE OR REPLACE FUNCTION get_product_avg_rating()
RETURNS numeric
LANGUAGE sql
AS $$
    SELECT COALESCE(AVG(rating)::numeric, 0) FROM products WHERE rating IS NOT NULL AND rating > 0;
$$;

CREATE OR REPLACE FUNCTION get_product_avg_sentiment()
RETURNS numeric
LANGUAGE sql
AS $$
    SELECT COALESCE(AVG(avg_sentiment)::numeric, 0) FROM products WHERE avg_sentiment IS NOT NULL;
$$;
