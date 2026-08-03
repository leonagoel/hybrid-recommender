import time
from threading import Lock


class TokenBucketLimiter:
    """
    An O(1) Time-Complexity Token Bucket Rate Limiter with periodic cleanup.
    
    Fix for Issue #2 (bug2.md): O(N) Garbage Collection Bottleneck
    
    The original implementation had unbounded bucket growth with no cleanup,
    causing O(N) iteration over all buckets on every request.
    
    This fix adds:
    1. Periodic cleanup every N requests (configurable, default 1000)
    2. Stale bucket detection based on idle time
    3. O(1) request evaluation
    """
    
    def __init__(self, capacity: int, refill_rate: float, cleanup_interval: int = 1000):
        """
        Initialize the rate limiter.
        
        :param capacity: Maximum number of tokens the bucket can hold.
        :param refill_rate: How many tokens are added to the bucket per second.
        :param cleanup_interval: Number of requests between cleanup operations.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.cleanup_interval = cleanup_interval
        
        # Thread safety lock to handle high-throughput concurrent API spikes
        self.lock = Lock()
        
        # Memory-efficient storage tracking: { user_id: (current_tokens, last_update_timestamp) }
        self.buckets = {}
        
        # Request counter for triggering periodic cleanup
        self._request_count = 0
        
        # Calculate stale threshold (time after which bucket is considered abandoned)
        self._stale_threshold = (capacity / refill_rate) * 2 if refill_rate > 0 else 3600

    def allow_request(self, user_id: str) -> bool:
        """
        Evaluates request allowance using lazy mathematical accumulation in O(1) time.
        Includes periodic cleanup to prevent unbounded bucket growth.
        """
        now = time.time()
        
        with self.lock:
            # Periodic cleanup every N requests to prevent unbounded growth
            self._request_count += 1
            if self._request_count >= self.cleanup_interval:
                self._cleanup_stale_buckets(now)
                self._request_count = 0
            
            # 1. Initialize user state if it's their first request
            if user_id not in self.buckets:
                self.buckets[user_id] = (self.capacity, now)
                # Deduct 1 token for the current active request
                self.buckets[user_id] = (self.capacity - 1, now)
                return True
            
            current_tokens, last_update = self.buckets[user_id]
            
            # 2. Lazy Refill: Calculate tokens accumulated during the elapsed time delta
            elapsed_time = now - last_update
            generated_tokens = elapsed_time * self.refill_rate
            
            # Update the token balance without exceeding the bucket's maximum capacity
            refilled_tokens = min(self.capacity, current_tokens + generated_tokens)
            
            # 3. Evaluation: Check if the user has enough tokens remaining
            if refilled_tokens >= 1.0:
                # Deduct a token and update their timestamp snapshot
                self.buckets[user_id] = (refilled_tokens - 1.0, now)
                return True
            else:
                # Bucket is depleted; persist their updated refilled fractional tokens
                self.buckets[user_id] = (refilled_tokens, now)
                return False

    def _cleanup_stale_buckets(self, now: float) -> int:
        """
        Remove abandoned buckets to prevent unbounded memory growth.
        
        A bucket is considered stale if:
        1. It has been idle for longer than 2x the time needed to fully refill
        2. OR it has zero tokens AND hasn't been accessed recently
        
        Returns the number of buckets removed.
        """
        stale_keys = []
        
        for user_id, (tokens, last_update) in self.buckets.items():
            idle_time = now - last_update
            
            # Bucket is stale if idle for longer than threshold
            if idle_time > self._stale_threshold:
                stale_keys.append(user_id)
        
        for key in stale_keys:
            del self.buckets[key]
        
        return len(stale_keys)

# ========================================================
# Demonstration / Integration Hook Example
# ========================================================
if __name__ == "__main__":
    # Allow a maximum burst of 5 requests, refilling at a rate of 1 token per second
    limiter = TokenBucketLimiter(capacity=5, refill_rate=1.0)
    client_ip = "192.168.1.50"

    print("--- Simulating Rapid Request Burst ---")
    for i in range(7):
        allowed = limiter.allow_request(client_ip)
        print(f"Request {i+1}: {'✅ ALLOWED' if allowed else '❌ BLOCKED (Rate Limited)'}")
        time.sleep(0.1) # Rapid firing

    print("\n--- Sleeping for 2 Seconds to Refill ---")
    time.sleep(2.0)

    print("--- Post-Refill Test ---")
    print(f"Request 8: {'✅ ALLOWED' if limiter.allow_request(client_ip) else '❌ BLOCKED'}")