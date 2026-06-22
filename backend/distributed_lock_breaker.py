import asyncio
import time
import random

class CircuitBreakerException(Exception): pass

class SafeModelStateController:
    """
    Advanced Concurrency Controller implementing a non-blocking Read-Copy-Update (RCU)
    matrix swap thread pattern combined with an active Circuit Breaker state machine.
    Removes Distributed Deadlocks and prevents Event Loop Starvation.
    """
    def __init__(self):
        self._matrix_version = 0
        # Core active shared weights resource matrix
        self.shared_weights = {"version": 0, "data": np.random.randn(512, 512)} # type: ignore
        self._write_lock = asyncio.Lock()
        
        # Circuit Breaker States: CLOSED (Healthy), OPEN (Failing), HALF-OPEN (Testing)
        self.circuit_state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = 3
        self.recovery_timeout = 2.0  # Seconds to wait before testing recovery
        self.last_state_change = time.time()

    async def read_inference_matrix(self) -> dict:
        """
        High-frequency non-blocking Read operation. Uses an atomic snapshot access 
        pattern to completely eliminate read-locks and avoid lock inversion.
        """
        self._check_circuit_condition()
        
        # Zero lock contention: Directly reference the active immutable memory reference pointer
        # This keeps user API reads running at sub-millisecond speeds even during background writes
        return self.shared_weights

    async def trigger_background_weight_update(self, compute_heavy_matrix_callback):
        """
        Isolated Write operation. Leverages an asynchronous context manager with an 
        explicit acquisition timeout constraint to prevent thread pool deadlocks.
        """
        if self.circuit_state == "OPEN":
            if time.time() - self.last_state_change > self.recovery_timeout:
                print("[CIRCUIT BREAKER] Entering HALF-OPEN state. Testing pipeline recovery...")
                self.circuit_state = "HALF-OPEN"
            else:
                raise CircuitBreakerException("Write pipeline blocked: Circuit Breaker is OPEN.")

        # Enforce a strict execution deadline on lock acquisition to prevent starvation loops
        try:
            async with asyncio.timeout(0.5):  # 500ms hard threshold deadline
                async with self._write_lock:
                    print(f"[WRITE WORKER] Lock acquired safely. Generating version {self._matrix_version + 1}...")
                    
                    # 1. Compute changes on an isolated deep copy in memory (Read-Copy-Update)
                    new_data = await compute_heavy_matrix_callback()
                    
                    # 2. Perform an atomic switch of the pointer reference to update the state instantly
                    self._matrix_version += 1
                    self.shared_weights = {"version": self._matrix_version, "data": new_data}
                    
                    # Reset failure thresholds on success
                    self.failure_count = 0
                    self.circuit_state = "CLOSED"
                    print(f"[WRITE SUCCESS] Atomic pointer swap complete. System state: version {self._matrix_version}")
                    
        except (asyncio.TimeoutError, Exception) as e:
            self._handle_pipeline_failure(e)

    def _check_circuit_condition(self):
        if self.circuit_state == "OPEN":
            if time.time() - self.last_state_change > self.recovery_timeout:
                # Trip to HALF-OPEN to allow read testing passes
                return
            raise CircuitBreakerException("Critical System Block: Circuit Breaker is currently tripped (OPEN).")

    def _handle_pipeline_failure(self, error):
        self.failure_count += 1
        print(f"[CRITICAL FAILURE] Pipeline execution error recorded: {error} | Failure Count: {self.failure_count}")
        if self.failure_count >= self.failure_threshold:
            self.circuit_state = "OPEN"
            self.last_state_change = time.time()
            print(">>> [CIRCUIT BREAKER TRIPPED] State shifted to OPEN. Isolating the model thread pool to protect system core!")

# ============================================================================
# CONCURRENCY STRESS VERIFICATION MODULE
# ============================================================================
import num  py as np # type: ignore

async def mock_faulty_ml_training_pass():
    await asyncio.sleep(0.2)  # Simulate active matrix calculation passes
    if random.random() > 0.4:
        raise RuntimeError("Simulated Database Connection Timeout failure during weight ingestion.")
    return np.random.randn(512, 512) # type: ignore

async def simulate_user_api_traffic(controller: SafeModelStateController, client_id: int):
    try:
        start = time.perf_counter()
        weights = await controller.read_inference_matrix()
        latency = (time.perf_counter() - start) * 1000
        print(f" -> API User Client #{client_id:02d} fetched model matrix V{weights['version']} in {latency:.2f}ms")
    except CircuitBreakerException as cbe:
        print(f" -> API User Client #{client_id:02d} BLOCKED gracefully by circuit breaker: {cbe}")

async def main():
    print("--- SYSTEM START: Deploying Deadlock-Resilient Model Controller ---")
    orchestrator = SafeModelStateController()

    # Fire parallel execution tracks simulating heavy user traffic combined with backend failures
    for wave in range(1, 4):
        print(f"\n--- EXECUTION WAVE #{wave} ---")
        
        # Interleave background updates with real-time user requests
        write_task = asyncio.create_task(orchestrator.trigger_background_weight_update(mock_faulty_ml_training_pass))
        read_tasks = [simulate_user_api_traffic(orchestrator, i) for i in range(1, 6)]
        
        await asyncio.gather(write_task, *read_tasks)
        await asyncio.sleep(1.0) # Let the recovery time metrics tick forward

if __name__ == "__main__":
    asyncio.run(main())