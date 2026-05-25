-- =============================================================================
-- Migration: Create benchmark_runs table
-- Run this in your Supabase SQL Editor before deploying the /api/evaluate endpoint
-- =============================================================================

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    k          INTEGER     NOT NULL CHECK (k BETWEEN 1 AND 100),
    mode       TEXT        NOT NULL DEFAULT 'all',
    weights    JSONB       NOT NULL DEFAULT '{"alpha": 0.4, "beta": 0.4, "gamma": 0.2}',
    results    JSONB       NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created_at
    ON benchmark_runs (created_at DESC);

ALTER TABLE benchmark_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read benchmark runs"
    ON benchmark_runs FOR SELECT
    USING (true);

CREATE POLICY "Service role can insert benchmark runs"
    ON benchmark_runs FOR INSERT
    WITH CHECK (true);
