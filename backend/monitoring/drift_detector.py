import numpy as np # type: ignore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CharingMind_DriftMonitor")

class DriftMonitor:
    def __init__(self, num_bins: int = 10):
        """
        Advanced ML Monitoring Engine to evaluate Population Stability Index (PSI)
        to prevent silent model decay in recommendation tracking arrays.
        """
        self.num_bins = num_bins

    def calculate_psi(self, baseline: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Population Stability Index between reference training data 
        and real-time live production user inference vectors.
        Formula: PSI = SUM( (Actual% - Expected%) * ln(Actual% / Expected%) )
        """
        # Remove any potential invalid NaN metrics from the arrays
        baseline = baseline[~np.isnan(baseline)]
        target = target[~np.isnan(target)]

        if len(baseline) == 0 or len(target) == 0:
            raise ValueError("Input distribution arrays cannot be empty.")

        # Determine reference quantile bin split points based on training data
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        bins = np.percentile(baseline, percentiles)
        
        # Adjust boundary edges slightly to prevent bin assignment errors
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        # Calculate frequency distribution counts across splits
        base_counts, _ = np.histogram(baseline, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)

        # Convert frequencies to relative percentages (probabilities)
        base_pct = base_counts / len(baseline)
        target_pct = target_counts / len(target)

        # Optimization Step: Apply Laplace smoothing to eliminate 0 probabilities 
        # to guarantee zero-division runtime errors are completely avoided
        base_pct = np.where(base_pct == 0, 1e-4, base_pct)
        target_pct = np.where(target_pct == 0, 1e-4, target_pct)

        # Core Statistical PSI Equation evaluation loop
        psi_value = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))

        # Handle Alert Hierarchy Triggers
        if psi_value >= 0.25:
            logger.error(f"🚨 CRITICAL SYSTEM MODEL DRIFT DETECTED: PSI={psi_value:.4f}. Immediate model update required.")
        elif psi_value >= 0.10:
            logger.warning(f"⚠️ MODERATE MODEL DRIFT WARNING: PSI={psi_value:.4f}. Monitor feature variance closely.")
        else:
            logger.info(f"✅ System stable. Distribution Index: {psi_value:.4f}")

        return float(psi_value)

# Quick Operational Mock Test Verification Segment
if __name__ == "__main__":
    monitor = DriftMonitor()
    
    # Simulate historical baseline user interaction profile metrics
    training_baseline = np.random.normal(loc=0.5, scale=0.1, size=1000)
    
    # Scenario A: Stable system distribution matching baseline parameters
    live_stable_traffic = np.random.normal(loc=0.5, scale=0.1, size=1000)
    print("Running Stable Verification Loop:")
    monitor.calculate_psi(training_baseline, live_stable_traffic)
    
    # Scenario B: Sharp distribution shift due to sudden concept drift
    live_drifted_traffic = np.random.normal(loc=0.62, scale=0.12, size=1000)
    print("\nRunning Shifted Concept Drift Verification Loop:")
    monitor.calculate_psi(training_baseline, live_drifted_traffic)