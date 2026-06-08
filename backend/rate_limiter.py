import time
from threading import Lock

class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        """
        An O(1) Time-Complexity Token Bucket Rate Limiter.
        
        :param capacity: Maximum number of tokens the bucket can hold.
        :param refill_rate: How many tokens are added to the bucket per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        
        # Thread safety lock to handle high-throughput concurrent API spikes
        self.lock = Lock()
        
        # Memory-efficient storage tracking: { user_id: (current_tokens, last_update_timestamp) }
        self.buckets = {}

    def allow_request(self, user_id: str) -> bool:
        """
        Evaluates request allowance using lazy mathematical accumulation in O(1) time.
        """
        now = time.time()
        
        with self.lock:
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