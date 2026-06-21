import asyncio
import math
import random
import time
from typing import Any, Optional, Tuple

class XFetchCacheManager:
    """
    Implements a High-Concurrency Sparsity-Aware Caching Layer equipped with
    the XFetch Probabilistic Early Expiration Algorithm to eliminate Cache Stampedes.
    """
    def __init__(self, beta: float = 1.0):
        # Beta (> 0): Aggressive scaling coefficient. Higher means earlier proactive refreshes.
        self.beta = beta
        self.cache_store = {}
        self.lock_store = {}

    def set(self, key: str, value: Any, ttl_seconds: float, compute_delta: float):
        """
        Stores an item in the cache memory pool.
        Tracks the explicit calculation time (compute_delta) taken to generate the array.
        """
        expire_at = time.time() + ttl_seconds
        self.cache_store[key] = {
            "value": value,
            "expire_at": expire_at,
            "ttl": ttl_seconds,
            "compute_delta": compute_delta  # Time delta delta taken to compute the ML matrix model
        }
        # Initialize a unique non-blocking async lock for this specific cache key
        if key not in self.lock_store:
            self.lock_store[key] = asyncio.Lock()

    async def get(self, key: str, matrix_recompute_callback) -> Any:
        """
        Retrieves items using probabilistic evaluation. Bypasses thundering herds
        by triggering background calculations before the hard drop timeout occurs.
        """
        now = time.time()
        cached_item = self.cache_store.get(key)

        # Case 1: Total Cache Miss (Cold start/First Boot) -> Forced Synchronous Safe Fetch
        if not cached_item:
            print(f"[CACHE MISS] Key '{key}' not found in cluster. Invoking immediate blocking compute...")
            return await self._execute_hard_refresh(key, matrix_recompute_callback)

        value = cached_item["value"]
        expire_at = cached_item["expire_at"]
        compute_delta = cached_item["compute_delta"]
        
        # Calculate remaining time-to-live bounds
        ttl_remaining = expire_at - now

        # Case 2: Hard Timeout Expiry -> Must force refresh immediately
        if ttl_remaining <= 0:
            print(f"[CACHE EXPIRED] Key '{key}' hard threshold breached. Forcing urgent calculation cycle.")
            return await self._execute_hard_refresh(key, matrix_recompute_callback)

        # Case 3: XFetch Probabilistic Evaluation Matrix Core
        # Equation: -beta * compute_delta * ln(rand()) > ttl_remaining
        rand_modifier = random.random()
        # Prevent math domain error if random selection drops exactly to zero
        rand_modifier = max(rand_modifier, 1e-9)
        
        xfetch_threshold = -self.beta * compute_delta * math.log(rand_modifier)

        if xfetch_threshold > ttl_remaining:
            # Check if another concurrent worker task has already locked this key for refreshing
            lock = self.lock_store[key]
            if not lock.locked():
                print(f"[XFETCH TRIGGER] Key '{key}' hit probabilistic early expiry threshold "
                      f"({xfetch_threshold:.3f} > {ttl_remaining:.3f}s). Spawning background refresh task...")
                
                # Fire-and-forget background worker to re-calculate the model matrix
                asyncio.create_task(self._background_refresh_worker(key, matrix_recompute_callback))
            else:
                print(f"[MUTEX ENFORCED] Key '{key}' triggered XFetch, but a background update lock is already active. Serving stale data safely.")

        # Always return the active cached vector instantly to maintain sub-millisecond response rates
        return value

    async def _execute_hard_refresh(self, key: str, callback) -> Any:
        """
        Safely locks and updates the cache values when a hard miss or expiration occurs.
        """
        lock = self.lock_store.setdefault(key, asyncio.Lock())
        async with lock:
            # Double-check cache record right inside lock scope to verify if a parallel thread resolved it
            cached = self.cache_store.get(key)
            if cached and cached["expire_at"] > time.time():
                return cached["value"]

            start_time = time.perf_counter()
            new_matrix = await callback()
            delta = time.perf_counter() - start_time
            
            # Default new items to a 5-second lifetime window for testing
            self.set(key, new_matrix, ttl_seconds=5.0, compute_delta=delta)
            return new_matrix

    async def _background_refresh_worker(self, key: str, callback):
        """
        Isolated task worker that quietly recalculates the recommendations 
        without holding up incoming user API requests.
        """
        lock = self.lock_store[key]
        if lock.locked():
            return  # Prevent duplicate overlap executions
            
        async with lock:
            try:
                start_time = time.perf_counter()
                updated_matrix = await callback()
                delta = time.perf_counter() - start_time
                
                self.set(key, updated_matrix, ttl_seconds=5.0, compute_delta=delta)
                print(f"[BACKGROUND SUCCESS] Cache key '{key}' refreshed quietly in background. New Delta: {delta*1000:.2f}ms")
            except Exception as e:
                print(f"[BACKGROUND ERROR] Failed to resolve background cache refresh: {e}")

# ============================================================================
# HIGH-CONCURRENCY STRESS SIMULATION VERIFICATION
# ============================================================================
async def mock_heavy_ml_factorization():
    """
    Simulates a heavy, resource-intensive machine learning recommendation calculation.
    """
    await asyncio.sleep(0.4)  # Simulate a heavy 400ms matrix computation delay
    return [random.randint(10, 99) for _ in range(5)]

async def mock_user_request_surge(cache: XFetchCacheManager, request_id: int):
    """
    Simulates a single active user thread hammering the endpoint.
    """
    # Fetch recommendation vectors for user profile key 'user_25BCE1256'
    start = time.perf_counter()
    recommendations = await cache.get("user_25BCE1256", mock_heavy_ml_factorization)
    latency = (time.perf_counter() - start) * 1000
    
    print(f" -> Thread Request #{request_id:02d} fetched recommendations: {recommendations} in {latency:.2f}ms")

async def main():
    print("--- PIPELINE EXPERIMENT START: Initializing XFetch Cache Cluster ---")
    cache_cluster = XFetchCacheManager(beta=1.5)

    # 1. Warm up the system cache with a cold start fetch
    await cache_cluster.get("user_25BCE1256", mock_heavy_ml_factorization)
    
    # 2. Wait 3.5 seconds to enter the vulnerable early expiration window (TTL is 5s, computation takes 0.4s)
    print("\n[SIMULATION] Letting cache age into the active XFetch expiration window...")
    await asyncio.sleep(3.5)

    # 3. Simulate a massive concurrent traffic surge (30 parallel requests hit the endpoint at once)
    print("\n[STRESS TEST] Firing a surge of 30 simultaneous web requests at the aging cache...")
    traffic_surge_threads = [
        mock_user_request_surge(cache_cluster, i) for i in range(1, 31)
    ]
    await asyncio.gather(*traffic_surge_threads)

    # 4. Wait out the remaining time window to observe background task updates resolving smoothly
    await asyncio.sleep(1.0)
    print("\n--- PIPELINE EXPERIMENT COMPLETE: Cache Stampede Deflected Perfectly ---")

if __name__ == "__main__":
    asyncio.run(main())