-- =============================================================================
-- Migration: Create recommendation_history table for personalized histories
-- Run this in your Supabase SQL Editor to track user recommendation histories
-- =============================================================================

CREATE TABLE IF NOT EXISTS recommendation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT,
    query TEXT,
    recommended_items JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recommendation_history_created_at
    ON recommendation_history (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recommendation_history_user_id
    ON recommendation_history (user_id);

ALTER TABLE recommendation_history ENABLE ROW LEVEL SECURITY;
