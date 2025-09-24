import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def confidence_interval(N_samples=40, sample_size=30, true_mean=64, true_std=3.098386, confidence=0.95):
    """
    Demonstrates 95% confidence intervals for the mean.
    Parameters:
    - N_samples: Number of independent samples (here 40)
    - sample_size: Number of observations per sample
    - true_mean: True population mean
    - true_std: True population standard deviation
    - confidence: Confidence level (default 0.95)
    """
    
    # Z value for the two-tailed confidence interval
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    # Store the lower and upper bounds of each CI
    ci_lowers = []
    ci_uppers = []
    
    # Track intervals that do NOT contain the true mean
    misses = 0
    
    # Generate samples and compute CIs
    for i in range(N_samples):
        sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)

        # sample mean and standard error
        sample_mean = np.mean(sample)
        sample_se = np.std(sample, ddof=1) / np.sqrt(sample_size)
        
        # confidence interval bounds
        lower = sample_mean - z * sample_se
        upper = sample_mean + z * sample_se

        # append to CI lists        
        ci_lowers.append(lower)
        ci_uppers.append(upper)
        
        # check if true mean is within CI
        if not (lower <= true_mean <= upper):
            misses += 1
    
    # Plot the CIs
    plt.figure(figsize=(8, 6))
    for i, (low, up) in enumerate(zip(ci_lowers, ci_uppers)):
        # color red if interval misses true mean, else blue
        color = 'red' if not (low <= true_mean <= up) else 'blue'
        plt.plot([low, up], [i, i], color=color, lw=2)
        plt.plot([np.mean([low, up])], [i], 'o', color=color)  # mark sample mean
    
    plt.axvline(true_mean, color='magenta', linestyle='-', label='True Mean', lw=3)
    plt.xlabel("Value")
    plt.ylabel("Sample #")
    plt.title(f"{confidence*100:.0f}% Confidence Intervals for Sample Means\nMissed intervals: {misses}/{N_samples}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # print the number of misses and percentage
    print(f"Out of {N_samples} intervals, {misses} did NOT contain the true mean.")
    print(f"This is roughly {(misses/N_samples)*100:.1f}%, close to the expected {100*(1-confidence):.0f}% for a {confidence*100:.0f}% CI.")

# ================================
# Run main if this script is executed
# ================================
if __name__ == "__main__":
    confidence_interval()
    plt.show()  # do not comment this out
