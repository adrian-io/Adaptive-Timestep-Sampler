import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your specific Run ID, or set to None to auto-detect latest "fm_adaptive_final"
RUN_ID = None 

# Your WandB Entity
ENTITY = "adrian-scholl99-ludwig-maximilian-university-of-munich" 
PROJECT = "adaptive-timestep-sampler"
RUN_NAME_FILTER = "fm_adaptive_final"

def get_run_id(api, entity, project, run_id=None):
    if run_id:
        return run_id
    
    print(f"No RUN_ID provided. Searching for latest run named '{RUN_NAME_FILTER}'...")
    runs = api.runs(f"{entity}/{project}", filters={"display_name": RUN_NAME_FILTER}, order="-created_at")
    
    if len(runs) > 0:
        latest_run = runs[0]
        print(f"Found latest run: {latest_run.name} (ID: {latest_run.id}) created at {latest_run.created_at}")
        return latest_run.id
    else:
        raise ValueError(f"No runs found with name '{RUN_NAME_FILTER}' in {entity}/{project}")

def fetch_wandb_history(entity, project, run_id=None):
    """Fetches the full history of a run from WandB."""
    api = wandb.Api()
    
    # Resolve Run ID
    try:
        actual_run_id = get_run_id(api, entity, project, run_id)
    except Exception as e:
        print(e)
        return None

    run_path = f"{entity}/{project}/{actual_run_id}"
    print(f"Fetching run data: {run_path}...")
    
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Error fetching run. Check your ENTITY and RUN_ID.\n{e}")
        return None

    # Fetch history (metrics logged over time)
    # We increase the samples to ensure we get all points if the run is long
    history = run.history(keys=["policy/alpha_mean", "policy/beta_mean", "epoch", "_step"], samples=100000)
    
    # Clean up dataframe
    df = history.dropna(subset=["policy/alpha_mean", "policy/beta_mean"])
    return df

def plot_mean_evolution(df):
    """Plots the mean of the Beta distribution (alpha / (alpha + beta)) over time."""
    
    # Calculate Beta Distribution Mean: mean = alpha / (alpha + beta)
    df["dist_mean"] = df["policy/alpha_mean"] / (
        df["policy/alpha_mean"] + df["policy/beta_mean"]
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["_step"], df["dist_mean"], label="Beta Distribution Mean", color="blue", alpha=0.8)
    
    plt.title(f"Evolution of Learned Time Preference (Mean of Beta Dist)")
    plt.xlabel("Global Step")
    plt.ylabel("Mean Timestep (normalized [0,1])")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("beta_mean_evolution.png")
    plt.show()
    print("Saved 'beta_mean_evolution.png'")

def plot_beta_distributions(df, epoch_interval=25):
    """Plots the PDF of the beta distribution at specific epoch intervals."""
    
    # Get the last logged step for each available epoch
    # (Grouping by epoch and taking tail ensures we get the final state of that epoch)
    epoch_df = df.groupby("epoch").tail(1).sort_values("epoch")
    
    # Filter for every Nth epoch
    # We also always include the first and last available epoch
    selected_epochs = epoch_df[
        (epoch_df["epoch"] % epoch_interval == 0) | 
        (epoch_df["epoch"] == epoch_df["epoch"].min()) |
        (epoch_df["epoch"] == epoch_df["epoch"].max())
    ].copy()
    
    # Remove duplicates if min/max overlap with interval
    selected_epochs = selected_epochs.drop_duplicates(subset=["epoch"])

    x = np.linspace(0, 1, 500)
    plt.figure(figsize=(12, 8))
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_epochs)))

    for i, (_, row) in enumerate(selected_epochs.iterrows()):
        a = row["policy/alpha_mean"]
        b = row["policy/beta_mean"]
        epoch = int(row["epoch"])
        
        # Calculate PDF
        y = beta.pdf(x, a, b)
        
        plt.plot(x, y, label=f"Epoch {epoch} ($\\alpha$={a:.2f}, $\\beta$={b:.2f})", color=colors[i], lw=2)

    plt.title(f"Evolution of Adaptive Sampler Policy (Every {epoch_interval} Epochs)")
    plt.xlabel("Timestep $t$ (normalized [0,1])")
    plt.ylabel("Probability Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("beta_dist_evolution.png")
    plt.show()
    print("Saved 'beta_dist_evolution.png'")

if __name__ == "__main__":
    # Logic updated to handle None RUN_ID inside fetch_wandb_history
    if ENTITY == "YOUR_WANDB_ENTITY_HERE":
         print("Please edit the script and set your ENTITY first.")
    else:
        df = fetch_wandb_history(ENTITY, PROJECT, RUN_ID)
        
        if df is not None and not df.empty:
            plot_mean_evolution(df)
            plot_beta_distributions(df, epoch_interval=25)
        else:
            print("No data found or processed.")