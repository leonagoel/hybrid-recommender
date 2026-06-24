import asyncio
import queue
import time
import random
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class StreamEvent:
    """
    Data payload container for incoming streaming interactions.
    Automatically sorts by priority level (lower values processed first) and timestamp.
    """
    priority: int
    timestamp: float = field(compare=True)
    payload: dict = field(compare=False)

class AsyncInteractionPipeline:
    """
    Thread-safe, asynchronous ingestion engine that buffers real-time ratings/clicks
    into a priority queue and dispatches them to parallel model worker loops.
    """
    def __init__(self, max_buffer_size=1000):
        # Thread-safe heap priority queue structure
        self.event_queue = queue.PriorityQueue(maxsize=max_buffer_size)
        self.is_running = False

    def ingest_event(self, event_type: str, user_id: int, item_id: int, rating: float = 1.0, priority: int = 2):
        """
        Synchronous, non-blocking ingestion endpoint to capture live streaming traffic.
        Higher-priority events (like checkout ratings) bypass routine telemetry clicks.
        """
        payload = {
            "type": event_type,
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating
        }
        # Priority level 1 = Urgent, 2 = Standard, 3 = Low/Background Telemetry
        stream_item = StreamEvent(priority=priority, timestamp=time.time(), payload=payload)
        
        try:
            self.event_queue.put_nowait(stream_item)
            return True
        except queue.Full:
            print(f"[PIPELINE ALERT] Buffer Overflow! Drop event: {payload}")
            return False

    async def _async_worker_consumer(self, worker_id: int):
        """
        Asynchronous consumer worker that continuously dequeues events from the buffer
        and updates active model embedding parameters.
        """
        print(f" -> Async Stream Consumer Worker_{worker_id} Spawned.")
        while self.is_running:
            try:
                # Retrieve from thread-safe queue without blocking the event loop
                # loop.run_in_executor allows running blocking queue reads in async workflows cleanly
                loop = asyncio.get_running_loop()
                event: StreamEvent = await loop.run_in_executor(
                    None, lambda: self.event_queue.get(timeout=0.1)
                )
                
                # --- SIMULATE MODEL MATRIX FACTORIZATION UPDATES ---
                data = event.payload
                start_calc = time.perf_counter()
                
                # Simulate dynamic math weight update delay
                await asyncio.sleep(0.05) 
                
                latency = (time.perf_counter() - start_calc) * 1000
                print(f"[Worker_{worker_id}] Processed [Prio {event.priority}] "
                      f"{data['type']} for User {data['user_id']} -> Latency: {latency:.2f}ms")
                
                # Signal queue task completion
                self.event_queue.task_done()
                
            except queue.Empty:
                # No items in pipeline buffer, loop back and await stream items
                await asyncio.sleep(0.01)

    async def start_pipeline(self, num_workers=3):
        """
        Activates the asynchronous processing loop and pools background workers.
        """
        self.is_running = True
        print(f"[PIPELINE CORE] Initializing Streaming Pipeline with {num_workers} Consumers...")
        
        # Instantiate concurrent async task workers
        self.workers = [
            asyncio.create_task(self._async_worker_consumer(i)) for i in range(num_workers)
        ]
        await asyncio.gather(*self.workers, return_exceptions=True)

    def stop_pipeline(self):
        """
        Gracefully terminates ingestion consumption workers.
        """
        print("[PIPELINE CORE] Shaking down connection lines. Halting processing loops.")
        self.is_running = False

# ============================================================================
# VERIFICATION MULTI-THREADED TESTING SUITE
# ============================================================================
async def mock_traffic_generator(pipeline: AsyncInteractionPipeline):
    """
    Simulates high-concurrency traffic bursts hitting your application API.
    """
    event_types = ["CLICK", "IMPRESSION", "RATING_SUBMIT", "WISHLIST"]
    
    print("\n--- BURST INGESTION: Simulating High-Volume User Actions ---")
    for _ in range(20):
        e_type = random.choice(event_types)
        # Assign highest priority (1) specifically to explicit Rating submissions
        prio = 1 if e_type == "RATING_SUBMIT" else random.choice([2, 3])
        
        pipeline.ingest_event(
            event_type=e_type,
            user_id=random.randint(1000, 9999),
            item_id=random.randint(500, 600),
            rating=float(random.randint(1, 5)),
            priority=prio
        )
    print(f"[TRAFFIC GEN] Staged {pipeline.event_queue.qsize()} streaming events into memory pool.\n")

async def main():
    stream_pipeline = AsyncInteractionPipeline(max_buffer_size=100)
    
    # Fire off live concurrent traffic and spawn the pipeline engine loop side-by-side
    traffic_task = asyncio.create_task(mock_traffic_generator(stream_pipeline))
    
    # Run pipeline consumers for a short window to demonstrate prioritized orchestration
    pipeline_task = asyncio.create_task(stream_pipeline.start_pipeline(num_workers=3))
    
    await traffic_task
    await asyncio.sleep(1.0) # Allow active consumers to clear the backlog buffer
    
    stream_pipeline.stop_pipeline()
    await pipeline_task

if __name__ == "__main__":
    asyncio.run(main())