import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, lognorm, gumbel_r

# ================================
# Define distributions with fixed parameters
# ================================
dists = {
    "Uniform": uniform(loc=-2, scale=4),           # uniform between -2 and 2

    # WRITE_YOUR_CODE HERE TO DEFINE OTHER DISTRIBUTIONS WITH FIXED PARAMETERS
    "Normal":    norm(loc=0, scale=1),              # standard normal
    "Lognormal": lognorm(s=0.5, scale=1),        # median = 1
    "Gumbel-1":  gumbel_r(loc=0, scale=1),             # Gumbel distribution
    # this code block ends here

}

# ================================
# Task 1: Explore distributions (PDFs)
# ================================
def task1_plot_pdfs(N=1000):
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    axes = axes.flatten()
    
    for ax, (name, dist) in zip(axes, dists.items()):
        x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), N)

        # WRITE_YOUR_CODE HERE TO DEFINE y AS THE PDF VALUES
        y = dist.pdf(x)
        # this code block ends here

        ax.plot(x, y, 'r-', lw=2)
        ax.set_title(name)
        ax.grid(True)
    
    plt.tight_layout()

# ================================
# Task 2: Raw sampling (m=1)
# ================================
def task2_histograms(N=1000):
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    axes = axes.flatten()
    
    for ax, (name, dist) in zip(axes, dists.items()):
        samples = dist.rvs(size=N)
        ax.hist(samples, bins=20, density=True, alpha=0.7, color='skyblue')
        
        # WRITE_YOUR_CODE HERE TO OVERLAY PDF ON EACH HISTOGRAM
        x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 1000)
        y = dist.pdf(x)
        ax.plot(x, y, 'r-', lw=2)
        # this code block ends here

        ax.set_title(name)
        ax.grid(True)
    
    plt.tight_layout()

# ================================
# Task 3: Averaging effect (m>1)
# ================================
def task3_averaging(N=1000, m_list=[1, 2, 10, 100]):
    for name, dist in dists.items():

        fig, axes = plt.subplots(2, 2, figsize=(12,8))
        fig.suptitle(f"Averaging effect - {name}")
        
        for ax, m in zip(axes.flat, m_list):
            samples = np.mean(dist.rvs(size=(N,m)), axis=1)

            # WRITE_YOUR_CODE HERE TO FIND SIMULATED MEAN, VAR FROM SAMPLES
            # Simulated mean, var, and std of selected samples
            sim_mean = np.mean(samples)
            sim_var = np.var(samples)

            # WRITE_YOUR_CODE HERE TO FIND SIMULATED MEAN, VAR FROM DISTRIBUTION DEFINITIONS
            # Theoretical sample mean, var, and std
            theo_mu =   dist.mean()
            theo_var =  dist.var()/ m
            # this code block ends here

            ax.hist(samples, bins=20, density=True, alpha=0.7, color='orange')
            ax.set_title(f"m={m}")
            ax.grid(True)
            
            # WRITE_YOUR_CODE HERE TO ADD TITLE WITH THEORETICAL AND SIMULATED VALUES. WHAT WILL GO IN BRACES {}?
            ax.set_title(f"For m={m}, theoretical mean (CLT) ~ N({theo_mu:.3f}, {dist.var():.3f}/{m} = {theo_var:.3f})\n"
            f"Simulated: N={N}, avg={sim_mean:.3f}, Var={sim_var:.3f}"
            )
            
            # this code block ends here

            # WRITE_YOUR_CODE HERE TO OVERLAY THEORETICAL CLT RESULT ON EACH HISTOGRAM
            x = np.linspace(min(samples), max(samples), 1000)                          
            clt_dist = norm(loc=theo_mu, scale=np.sqrt(theo_var))
            y = clt_dist.pdf(x)
            ax.plot(x, y, 'r-', lw=2, label="CLT Normal Approx")
            # this code block ends here

        plt.tight_layout(rect=[0,0,1,0.95])

# ================================
# Task 4: Variance scaling
# ================================
def task4_variance_scaling(N=1000, m_list=[1,2,10,100]):
    for name, dist in dists.items():
        var_original = dist.var()
        print(f"{name}: Original variance = {var_original:.4f}")
        
        for m in m_list:
            samples = np.mean(dist.rvs(size=(N,m)), axis=1)

            # WRITE_YOUR_CODE HERE TO COMPUTE VARIANCE OF THE SAMPLE MEANS
            var_avg = np.var(samples)
            # this code block ends here

            print(f"  m={m}, Variance of mean = {var_avg:.4f}, Expected = {var_original/m:.4f}")
        print("-"*40)

# ================================
# Task 5: Mean vs. standard deviation
# ================================
def task5_mean_std(N=1000, m=10):
    for name, dist in dists.items():
        # draw N samples, each of size m
        data = dist.rvs(size=(N,m))
        means = np.mean(data, axis=1)
        stds  = np.std(data, axis=1)

        # WRITE_YOUR_CODE HERE TO COMPUTE THEORETICAL MEAN, STD OF SAMPLE MEANS
        # theoretical CLT parameters
        theo_mu  =   dist.mean()
        theo_std =   dist.std()/np.sqrt(m)
        # this code block ends here

        fig, axes = plt.subplots(1,2, figsize=(12,4))
        fig.suptitle(f"{name}: Sample Means vs Stds", fontsize=14)

        # --- Means (CLT applies) ---
        axes[0].hist(means, bins=20, density=True, color='purple', alpha=0.7, label="Simulated Means")

        # overlay Normal approx from CLT
        x = np.linspace(min(means), max(means), 200)
        axes[0].plot(x, norm.pdf(x, loc=theo_mu, scale=theo_std), 'r-', lw=2, label="CLT Normal Approx")
        axes[0].set_title("Sample Means (≈ Normal by CLT)")
        axes[0].legend()
        axes[0].grid(True)

        # --- Standard deviations (no CLT) ---
        axes[1].hist(stds, bins=20, density=True, color='brown', alpha=0.7, label="Simulated Stds")
        axes[1].set_title("Sample Standard Deviations (not Normal)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])


# ================================
# Run main if this script is executed
# ================================
if __name__ == "__main__":
    print("Task 1: Plot PDFs")
    task1_plot_pdfs()
    
    # print("Task 2: Raw Sampling Histograms")
    task2_histograms()
    
    # print("Task 3: Averaging Effect")
    task3_averaging()
    
    print("Task 4: Variance Scaling")
    task4_variance_scaling()
    
    print("Task 5: Mean vs Standard Deviation")
    task5_mean_std()

    plt.show() # do not comment this out