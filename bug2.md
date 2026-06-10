## Bug Description
The rate limiter implementation in `backend/main.py` suffers from an O(N) garbage collection bottleneck inside a global lock on every single request. In `_apply_rate_limit`, after updating the rate limit buckets, the code executes a list comprehension across the entire `_rate_limit_buckets` dictionary: `empty_keys = [k for k, v in _rate_limit_buckets.items() if not v]`. Since this runs within `with _rate_limit_lock:`, an attacker can fill the buckets with thousands of spoofed IPs. The server will then spend an O(N) amount of time iterating through all buckets for *every single incoming request* across all threads, leading to an Algorithmic Complexity DoS (thread starvation and massive latency).

## Steps to Reproduce
1. Write a script to send thousands of requests from varying spoofed `X-Forwarded-For` IPs (if trusted) or via a botnet/proxies to any rate-limited API endpoint.
2. Observe the size of `_rate_limit_buckets` growing.
3. Send a legitimate request and measure the response time.
4. The response time will increase linearly with the number of tracked IPs, degrading throughput drastically.

## Expected Behavior
Garbage collection of empty rate limit buckets should be handled asynchronously (e.g., via a background thread, Celery beat task) or periodically (e.g., every 10,000 requests), not synchronously on every request within a global lock. Alternatively, a structure with automatic TTL eviction like Redis should be used.

## Actual Behavior
O(N) iteration over the global rate limit bucket dictionary happens synchronously under a lock on every single request, causing severe blocking and throughput degradation under load.

## Screenshots / Error Logs
N/A

## Environment
- OS: Any
- Python version: Any

## Additional Context
Advanced level performance and security (DoS) issue in `backend/main.py`.
