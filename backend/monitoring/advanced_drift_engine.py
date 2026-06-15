"""
CharingMind Advanced Machine Learning Operations (MLOps) Engine
Enterprise-Grade Asynchronous Model Drift and Distribution Deviations Tracking System.
"""

import numpy as np # type: ignore
from scipy.stats import wasserstein_distance # type: ignore
import threading
import queue
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

# Setup high-fidelity telemetry logger configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)s) %(name)s: %(message)s'
)
logger = logging.getLogger("CharingMind_MLOps_Core")


class DriftAnalysisEngine:
    def __init__(self, num_bins: int = 10, laplace_smoothing: float = 1e-5):
        """
        Mathematical execution pipeline computing distributional distance vectors.
        """
        self.num_bins = num_bins
        self.epsilon = laplace_smoothing

    def compute_psi(self, baseline: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates Population Stability Index (PSI) between training data and active runtime vectors.
        Formula: PSI = SUM( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
        """
        # Purge undefined values
        b_clean = baseline[~np.isnan(baseline)]
        t_clean = target[~np.isnan(target)]
        
        if b_clean.size == 0 or t_clean.size == 0:
            return 0.0

        # Formulate quantile split points dynamically from training distribution matrix
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        bins = np.percentile(b_clean, percentiles)
        
        # Adjust outer boundary limits to handle extreme floating point vectors cleanly
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        # Compute counts via histograms
        base_counts, _ = np.histogram(b_clean, bins=bins)
        target_counts, _ = np.histogram(t_clean, bins=bins)

        # Convert frequencies to relative probability distributions
        base_prob = base_counts / b_clean.size
        target_prob = target_counts / t_clean.size

        # Apply Laplace Smoothing to protect against infinity/zero-division boundary crashes
        base_prob = np.where(base_prob == 0, self.epsilon, base_prob)
        target_prob = np.where(target_prob == 0, self.epsilon, target_prob)

        # Vectorized implementation of the core information theory logarithmic equation
        psi_val = np.sum((target_prob - base_prob) * np.log(target_prob / base_prob))
        return float(psi_val)

    def compute_wasserstein(self, baseline: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the first Earth Mover's Distance to measure work needed to transform
        the live data distribution shape back into the baseline reference profile shape.
        """
        b_clean = baseline[~np.isnan(baseline)]
        t_clean = target[~np.isnan(target)]
        
        if b_clean.size == 0 or t_clean.size == 0:
            return 0.0
            
        return float(wasserstein_distance(b_clean, t_clean))


class AsynchronousDriftMonitor:
    def __init__(self, feature_name: str, baseline_reference: List[float], 
                 window_capacity: int = 2000, batch_eval_size: int = 500):
        """
        Thread-safe tracking pipeline that leverages a non-blocking queue to intercept 
        live vector samples asynchronously and computes multi-metric data deviation summaries.
        """
        self.feature_name = feature_name
        self.baseline = np.array(baseline_reference, dtype=np.float64)
        self.window_capacity = window_capacity
        self.batch_eval_size = batch_eval_size
        
        # Concurrent synchronization allocations
        self.telemetry_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.sliding_buffer: List[float] = []
        self.lock = threading.Lock()
        
        self.engine = DriftAnalysisEngine()
        self._is_active = True
        
        # Deploy Background Consumer Thread
        self.worker_thread = threading.Thread(
            target=self._process_queue_stream, 
            name=f"DriftDaemon_{self.feature_name}", 
            daemon=True
        )
        self.worker_thread.start()

    def log_inference_sample(self, value: float) -> None:
        """
        Non-blocking ingestion hook. Drops data straight into the queue 
        so recommendation responses maintain lightning-fast execution times.
        """
        try:
            self.telemetry_queue.put_nowait(value)
        except queue.Full:
            # Prevent data collection overflows under high system saturation
            logger.warning(f"Telemetry queue saturated for {self.feature_name}. Frame skipped.")

    def _process_queue_stream(self) -> None:
        """
        Continuous collection loop that manages the ring buffer allocations and evaluates drift triggers.
        """
        while self._is_active:
            try:
                # Gather incoming metrics blocks
                sample = self.telemetry_queue.get(timeout=1.0)
                
                with self.lock:
                    self.sliding_buffer.append(sample)
                    
                    # Evict oldest samples when capacity is exceeded to bound RAM overhead
                    if len(self.sliding_buffer) > self.window_capacity:
                        self.sliding_buffer.pop(0)
                        
                    buffer_len = len(self.sliding_buffer)
                
                # Trigger batch evaluation when the metric accumulation steps hit bounds
                if buffer_len >= self.batch_eval_size and buffer_len % 100 == 0:
                    self._evaluate_distribution_shifts()
                    
                self.telemetry_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as ex:
                logger.error(f"Internal fault inside tracking runtime loop: {str(ex)}")

    def _evaluate_distribution_shifts(self) -> Tuple[float, float]:
        """
        Synchronizes snapshot contexts to run heavy statistical calculations outside locks.
        """
        with self.lock:
            current_snapshot = np.array(self.sliding_buffer, dtype=np.float64)

        # Run decoupled matrix calculation sequences
        start_calc = time.perf_counter()
        psi_score = self.engine.compute_psi(self.baseline, current_snapshot)
        emd_score = self.engine.compute_wasserstein(self.baseline, current_snapshot)
        latency_ms = (time.perf_counter() - start_calc) * 1000

        # Execute Warning Matrix Evaluation Paths
        if psi_score >= 0.25:
            logger.error(
                f"🚨 [CRITICAL MODEL DECAY DETECTED] Feature: '{self.feature_name}' | "
                f"PSI: {psi_score:.4f} (>=0.25) | Earth Mover Distance: {emd_score:.4f} | "
                f"Analysis Compute Overhead: {latency_ms:.2f}ms"
            )
        elif psi_score >= 0.10:
            logger.warning(
                f"⚠️ [MODERATE SYSTEM DRIFT ALERT] Feature: '{self.feature_name}' | "
                f"PSI: {psi_score:.4f} | Earth Mover Distance: {emd_score:.4f} | "
                f"Analysis Compute Overhead: {latency_ms:.2f}ms"
            )
        else:
            logger.info(
                f"✅ [SYSTEM PROFILE STABLE] Feature: '{self.feature_name}' | "
                f"PSI: {psi_score:.4f} | EMD: {emd_score:.4f}"
            )

        return psi_score, emd_score

    def shutdown(self) -> None:
        """Gracefully releases background collection worker dependencies."""
        self._is_active = False
        self.worker_thread.join(timeout=2.0)


# ==========================================
# SIMULATED INTERACTIVE TEST HARNESS PIPELINE
# ==========================================
if __name__ == "__main__":
    print("--- Initializing High-Performance Asynchronous Drift Monitor Pipeline ---")
    
    # 1. Simulate baseline user interaction density parameters (e.g., historical item interaction rate)
    np.random.seed(42)
    historical_training_distribution = np.random.normal(loc=0.75, scale=0.15, size=5000).tolist()
    
    # Instantiate monitor tracker module
    monitor = AsynchronousDriftMonitor(
        feature_name="user_affinity_vector_index",
        baseline_reference=historical_training_distribution,
        window_capacity=1500,
        batch_eval_size=500
    )
    
    # 2. Simulate stable live server requests matching baseline trends
    print("\n[Phase 1] Simulating 600 stable, incoming live API traffic logs...")
    for _ in range(600):
        mock_live_sample = np.random.normal(loc=0.75, scale=0.15)
        monitor.log_inference_sample(mock_live_sample)
    time.sleep(2.0)  # Let worker catch up and process the batch

    # 3. Simulate sudden concept drift (e.g., viral change in content consumption)
    print("\n[Phase 2] Simulating 800 heavily drifted user interaction vectors...")
    for _ in range(800):
        mock_drifted_sample = np.random.normal(loc=0.91, scale=0.08)
        monitor.log_inference_sample(mock_drifted_sample)
    time.sleep(2.0)

    # Cleanup interface
    monitor.shutdown()
    print("\n--- Pipeline Run Finished ---")