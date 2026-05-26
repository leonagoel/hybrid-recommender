-- =============================================================================
-- Migration: Create/repair products table used by import, search, and model build
-- =============================================================================

CREATE TABLE IF NOT EXISTS products (
    id            BIGSERIAL   PRIMARY KEY,
    title         TEXT        NOT NULL,
    description   TEXT        NOT NULL DEFAULT '',
    category      TEXT        NOT NULL DEFAULT '',
    rating        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    avg_sentiment DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    review_count  INTEGER     NOT NULL DEFAULT 0,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'Untitled',
    ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS rating DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS avg_sentiment DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS review_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE products
    ALTER COLUMN title DROP DEFAULT;

UPDATE products
SET review_count = 0
WHERE review_count IS NULL;

ALTER TABLE products
    ALTER COLUMN review_count SET DEFAULT 0,
    ALTER COLUMN review_count SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'products_review_count_nonnegative'
          AND conrelid = 'products'::regclass
    ) THEN
        ALTER TABLE products
            ADD CONSTRAINT products_review_count_nonnegative
            CHECK (review_count >= 0);
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_products_title_unique
    ON products (title);

CREATE INDEX IF NOT EXISTS idx_products_rating_review_count
    ON products (rating DESC, review_count DESC);

ALTER TABLE products ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'products'
          AND policyname = 'Anyone can read products'
    ) THEN
        CREATE POLICY "Anyone can read products"
            ON products FOR SELECT
            USING (true);
    END IF;
END
$$;
