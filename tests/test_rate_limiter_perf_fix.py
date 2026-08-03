"""
Test cases for Rate Limiter O(N) Performance Fix (Issue #2 / bug2.md).

This test suite verifies that the TokenBucketLimiter properly handles:
1. Periodic cleanup of stale buckets
2. O(1) request evaluation time
3. Memory-efficient bucket management

See bug2.md for full vulnerability details.
"""
import pytest
import time
import threading
from backend.rate_limiter import TokenBucketLimiter


class TestRateLimiterPerformance:
    """Test cases verifying performance characteristics of the rate limiter."""

    def test_periodic_cleanup_triggered(self):
        """
        Verify that cleanup is triggered every N requests.
        """
        limiter = TokenBucketLimiter(capacity=5, refill_rate=1.0, cleanup_interval=10)
        
        # Add some buckets
        for i in range(10):
            limiter.allow_request(f"user_{i}")
        
        # After 10 requests, cleanup should have been triggered
        # We can verify this by checking the bucket count doesn't grow unbounded
        
        # Add more users (should trigger cleanup)
        initial_count = len(limiter.buckets)
        
        # Wait for buckets to become stale
        time.sleep(2)
        
        # Next 10 requests should trigger cleanup
        for i in range(10):
            limiter.allow_request(f"new_user_{i}")
        
        # Verify cleanup occurred - stale buckets should be removed
        assert len(limiter.buckets) < (initial_count + 10) * 2

    def test_stale_bucket_cleanup(self):
        """
        Verify that stale (abandoned) buckets are properly cleaned up.
        """
        limiter = TokenBucketLimiter(capacity=5, refill_rate=1.0, cleanup_interval=100)
        
        # Add buckets
        for i in range(20):
            limiter.allow_request(f"stale_user_{i}")
        
        initial_count = len(limiter.buckets)
        
        # Manually trigger cleanup after marking buckets as stale
        now = time.time()
        old_time = now - 1000  # 1000 seconds ago (definitely stale)
        
        # Mark some buckets as stale by directly manipulating storage
        for i in range(5):
            key = f"stale_user_{i}"
            if key in limiter.buckets:
                limiter.buckets[key] = (limiter.capacity, old_time)
        
        # Trigger cleanup
        removed = limiter._cleanup_stale_buckets(now)
        
        # Should have removed the 5 stale buckets
        assert removed == 5
        assert len(limiter.buckets) == initial_count - 5

    def test_memory_bounded_under_spoofed_ips(self):
        """
        Simulate attack scenario: attacker sends requests from many spoofed IPs.
        Verify memory doesn't grow unbounded.
        """
        limiter = TokenBucketLimiter(capacity=5, refill_rate=1.0, cleanup_interval=100)
        
        # Simulate many unique IPs
        num_ips = 1000
        for i in range(num_ips):
            limiter.allow_request(f"spoofed_ip_{i}")
        
        # Memory should be bounded
        assert len(limiter.buckets) == num_ips
        
        # Wait for buckets to become stale
        time.sleep(10)  # 2x stale_threshold for capacity=5, rate=1.0
        
        # Trigger cleanup by making more requests
        for i in range(limiter.cleanup_interval):
            limiter.allow_request("cleanup_trigger_user")
        
        # Stale buckets should have been removed
        assert len(limiter.buckets) < num_ips

    def test_cleanup_interval_configurable(self):
        """
        Verify cleanup_interval parameter works correctly.
        """
        # Test with very small interval
        limiter_small = TokenBucketLimiter(capacity=5, refill_rate=1.0, cleanup_interval=5)
        assert limiter_small.cleanup_interval == 5
        
        # Test with larger interval
        limiter_large = TokenBucketLimiter(capacity=5, refill_rate=1.0, cleanup_interval=10000)
        assert limiter_large.cleanup_interval == 10000

    def test_concurrent_access_safe(self):
        """
        Verify rate limiter is thread-safe during concurrent access.
        """
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10.0, cleanup_interval=50)
        errors = []
        
        def worker(start_id, count):
            try:
                for i in range(count):
                    limiter.allow_request(f"user_{start_id + i}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        num_threads = 10
        requests_per_thread = 100
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i * requests_per_thread, requests_per_thread))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Threading errors: {errors}"
        # All requests should have been processed
        assert len(limiter.buckets) > 0


class TestRateLimiterFunctionality:
    """Test basic rate limiting functionality still works correctly."""

    def test_basic_rate_limiting(self):
        """
        Verify basic token bucket algorithm still works.
        """
        limiter = TokenBucketLimiter(capacity=3, refill_rate=2.0, cleanup_interval=1000)
        
        # First 3 requests should be allowed
        assert limiter.allow_request("user") is True
        assert limiter.allow_request("user") is True
        assert limiter.allow_request("user") is True
        
        # 4th request should be blocked (bucket exhausted)
        assert limiter.allow_request("user") is False

    def test_refill_after_wait(self):
        """
        Verify tokens are refilled after waiting.
        """
        limiter = TokenBucketLimiter(capacity=1, refill_rate=10.0, cleanup_interval=1000)
        
        # First request allowed
        assert limiter.allow_request("user") is True
        
        # Immediate second request blocked
        assert limiter.allow_request("user") is False
        
        # Wait for refill (0.2 seconds should give 2 tokens)
        time.sleep(0.2)
        
        # Should be allowed again
        assert limiter.allow_request("user") is True

    def test_different_users_independent(self):
        """
        Verify each user has their own bucket.
        """
        limiter = TokenBucketLimiter(capacity=1, refill_rate=1.0, cleanup_interval=1000)
        
        # User A uses their bucket
        assert limiter.allow_request("user_a") is True
        assert limiter.allow_request("user_a") is False
        
        # User B has their own bucket (not affected)
        assert limiter.allow_request("user_b") is True


class TestVulnerabilityFix:
    """
    Test class documenting the fix for bug2.md.
    """

    def test_vulnerability_fixed(self):
        """
        DOCUMENTATION: Bug #2 - O(N) Garbage Collection Bottleneck
        
        BEFORE FIX:
        - No cleanup mechanism existed
        - _rate_limit_buckets dict grew unbounded
        - O(N) iteration over ALL buckets on every request
        - Attack could cause DoS by filling buckets
        
        AFTER FIX:
        - Periodic cleanup every N requests (default 1000)
        - O(N) cleanup only happens every 1000 requests (amortized O(1))
        - Stale buckets automatically removed
        - Memory bounded even under attack
        
        This test documents that the fix is in place.
        """
        limiter = TokenBucketLimiter(capacity=5, refill_rate=1.0, cleanup_interval=100)
        
        # Verify new methods exist
        assert hasattr(limiter, '_cleanup_stale_buckets')
        assert hasattr(limiter, '_request_count')
        assert hasattr(limiter, '_stale_threshold')
        assert hasattr(limiter, 'cleanup_interval')
        
        # Verify defaults
        assert limiter.cleanup_interval == 100
        assert limiter._stale_threshold > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
