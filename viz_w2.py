import numpy as np
import matplotlib.pyplot as plt

def sinkhorn_w2(X, Y, reg=0.05, numItermax=200):
    """Compute regularized W2 distance using Sinkhorn."""
    N = X.shape[0]
    # Pairwise squared euclidean distance
    M = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1)
    
    # Prevent underflow by normalizing M for the kernel
    M_max = np.max(M)
    if M_max == 0:
        return 0.0
    M_norm = M / M_max
    
    K = np.exp(-M_norm / reg)
    a = np.ones(N) / N
    b = np.ones(N) / N
    u = np.ones(N) / N
    
    for _ in range(numItermax):
        v = b / (np.dot(K.T, u) + 1e-12)
        u = a / (np.dot(K, v) + 1e-12)
        
    P = u[:, None] * K * v[None, :]
    return np.sum(P * M)

# Set up the experiment
np.random.seed(42)
d = 32 # dimensionality of our "images"
N = 300 # number of samples to represent the distribution at each timestep

# Base "images" (initial data distributions at t=1)
# We make 4 distinct images to recreate the 3 curves from Plot 1b
x1 = np.random.randn(d) * 3
x2 = np.random.randn(d) * 3 + 2
x3 = np.random.randn(d) * 3 - 2
x4 = np.random.randn(d) * 3 + 5

T_steps = np.linspace(0, 1, 25) # t=0 (noise) to t=1 (data)

def simulate_process(a_func, b_func):
    w2_12, w2_13, w2_14 = [], [], []
    for t in T_steps:
        a = a_func(t)
        b = b_func(t)
        
        # Sample from the noisy distributions
        X1 = a * x1 + b * np.random.randn(N, d)
        X2 = a * x2 + b * np.random.randn(N, d)
        X3 = a * x3 + b * np.random.randn(N, d)
        X4 = a * x4 + b * np.random.randn(N, d)
        
        # Compute empirical regularized W2
        w2_12.append(sinkhorn_w2(X1, X2))
        w2_13.append(sinkhorn_w2(X1, X3))
        w2_14.append(sinkhorn_w2(X1, X4))
        
    return np.array(w2_12), np.array(w2_13), np.array(w2_14)

# 1. Standard Diffusion (Variance Preserving mapped to t in [0, 1])
# At t=1 (data): a=1, b=0
# At t=0 (noise): a=0, b=1
def a_diff(t): return t
def b_diff(t): return np.sqrt(1 - t**2) + 1e-5
diff_res = simulate_process(a_diff, b_diff)

# 2. Flow Matching (Linear Interpolation)
# At t=1 (data): a=1, b=0
# At t=0 (noise): a=0, b=1
def a_fm(t): return t
def b_fm(t): return 1 - t + 1e-5
fm_res = simulate_process(a_fm, b_fm)

# Normalize distances by the max value (which occurs at t=1 data)
diff_res_norm = [r / np.max(r) for r in diff_res]
fm_res_norm = [r / np.max(r) for r in fm_res]

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
labels = [r'$W_2(\mu_t^1, \mu_t^2)$', r'$W_2(\mu_t^1, \mu_t^3)$', r'$W_2(\mu_t^1, \mu_t^4)$']

for r, c, l in zip(diff_res_norm, colors, labels):
    ax1.plot(T_steps, r, color=c, label=l, linewidth=2)
ax1.set_title("Standard Diffusion Process\n(Variance Preserving)", fontsize=14)
ax1.set_xlabel("Time step $t$ ($0$=Noise, $1$=Data)", fontsize=12)
ax1.set_ylabel("Normalized 2-Wasserstein distance", fontsize=12)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)

for r, c, l in zip(fm_res_norm, colors, labels):
    ax2.plot(T_steps, r, color=c, label=l, linewidth=2)
ax2.set_title("Flow Matching Process\n(Linear Interpolation)", fontsize=14)
ax2.set_xlabel("Time step $t$ ($0$=Noise, $1$=Data)", fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('plots/w2_recreation.png', dpi=300)