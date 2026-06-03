-- =============================================================================
-- Migration: Create feedback_submissions table for persisted user feedback
-- Run this in your Supabase SQL Editor before deploying the /api/feedback endpoint
-- =============================================================================

CREATE TABLE IF NOT EXISTS feedback_submissions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT        NOT NULL,
    item        TEXT        NOT NULL,
    feedback    TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_submissions_created_at
    ON feedback_submissions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_feedback_submissions_user_id
    ON feedback_submissions (user_id);

ALTER TABLE feedback_submissions ENABLE ROW LEVEL SECURITY;
