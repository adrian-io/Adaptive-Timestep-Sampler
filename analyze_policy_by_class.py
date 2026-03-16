import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as scipy_beta
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from ddpm_torch import ActorNetwork

def get_class_samples(root, num_samples_per_class=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    
    class_indices = {i: [] for i in range(10)}
    samples = []
    labels = []
    
    # Iterate through dataset to find indices for each class
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < num_samples_per_class:
            class_indices[label].append(idx)
        
        # Check if we have enough samples for all classes
        if all(len(indices) == num_samples_per_class for indices in class_indices.values()):
            break
            
    # Flatten indices and create subset
    all_indices = [idx for indices in class_indices.values() for idx in indices]
    subset = Subset(dataset, all_indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False) # Single batch
    
    # Get the data
    data, targets = next(iter(loader))
    
    return data, targets, dataset.classes

def plot_beta_pdfs(stats_history, classes, epochs, save_path="policy_evolution_by_class.png"):
    num_classes = len(classes)
    num_checkpoints = len(stats_history)
    
    # Create grid: Rows = Classes, Cols = Checkpoints
    fig, axes = plt.subplots(num_classes, num_checkpoints, figsize=(3 * num_checkpoints, 2 * num_classes), sharex=True, sharey=True)
    
    x = np.linspace(0, 1, 100)
    
    # Global title
    fig.suptitle('Learned Time Sampling Distributions per Class over Training', fontsize=16)

    for col_idx, epoch_stats in enumerate(stats_history):
        epoch = epochs[col_idx]
        
        for row_idx, class_name in enumerate(classes):
            ax = axes[row_idx, col_idx]
            
            # Get average alpha/beta for this class at this epoch
            alpha_val = epoch_stats[row_idx]['alpha']
            beta_val = epoch_stats[row_idx]['beta']
            
            # Plot Beta PDF
            # Handle potential numerical instability or extreme values if untrained
            try:
                if alpha_val <= 0 or beta_val <= 0:
                     pdf = np.zeros_like(x) # Invalid params
                else:
                     pdf = scipy_beta.pdf(x, alpha_val, beta_val)
            except:
                pdf = np.zeros_like(x)

            ax.plot(x, pdf, color='blue', lw=2)
            ax.fill_between(x, pdf, alpha=0.2, color='blue')
            
            # Formatting
            if row_idx == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=10, weight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            # ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--config", default="./configs/cifar10_fm.json", type=str)
    parser.add_argument("--chkpt-dir", required=True, type=str, help="Directory containing checkpoints")
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    root = os.path.expanduser(args.root)

    # 1. Load Data (10 samples per class)
    print("Loading data samples...")
    images, labels, class_names = get_class_samples(root)
    images = images.to(device)
    labels = labels.to(device)

    # 2. Initialize Policy Network
    with open(args.config, "r") as f:
        config = json.load(f)
    
    actor_config = config["ActorNetwork"]
    # Ensure initialize is False so we don't overwrite weights with random init before loading
    actor_config["initialize"] = False 
    policy = ActorNetwork(**actor_config).to(device)
    policy.eval()

    # 3. Find Checkpoints (Every 30th)
    chkpt_files = [f for f in os.listdir(args.chkpt_dir) if f.endswith(".pt")]
    
    # Parse epochs from filenames assuming format "name_epoch.pt" or similar
    # If standard naming from train.py: args.chkpt_name or f"{exp_name}.pt" 
    # train.py saves shadows as intermediate like: chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
    
    epoch_file_map = {}
    for f in chkpt_files:
        # Extract number at the end of filename before extension
        try:
            parts = f.replace(".pt", "").split("_")
            epoch = int(parts[-1])
            epoch_file_map[epoch] = f
        except ValueError:
            continue
            
    sorted_epochs = sorted(epoch_file_map.keys())
    # Filter: every 30th epoch (e.g., 30, 60, 90...)
    # Adjust logic based on how frequently you saved. 
    # If you saved every 10, select indices 2, 5, 8 etc. or simply `e % 30 == 0`
    
    selected_epochs = [e for e in sorted_epochs if e % 30 == 0 and e > 0]
    # If none found via modulo, perhaps just take every 3rd available checkpoint
    if not selected_epochs and len(sorted_epochs) > 0:
         print("Warning: strict 'every 30th' filter returned empty. Using step slicing [::3].")
         selected_epochs = sorted_epochs[::3]

    print(f"Selected epochs for analysis: {selected_epochs}")
    
    stats_history = []

    # 4. Iterate and Evaluate
    with torch.no_grad():
        for epoch in selected_epochs:
            chkpt_path = os.path.join(args.chkpt_dir, epoch_file_map[epoch])
            print(f"Processing {chkpt_path}...")
            
            try:
                checkpoint = torch.load(chkpt_path, map_location=device)
                
                # Check structure of checkpoint
                if "policy" in checkpoint:
                    policy.load_state_dict(checkpoint["policy"])
                else:
                    # Maybe it's a raw state dict? Unlikely given train.py
                    print(f"Skipping {chkpt_path}: 'policy' key not found.")
                    continue

                # Run inference
                alpha, beta = policy(images)
                
                # Squeeze outputs if necessary
                alpha = alpha.squeeze()
                beta = beta.squeeze()
                
                # Group by class
                epoch_stats = {}
                for class_idx in range(10):
                    mask = (labels == class_idx)
                    class_alphas = alpha[mask]
                    class_betas = beta[mask]
                    
                    avg_alpha = class_alphas.mean().item()
                    avg_beta = class_betas.mean().item()
                    
                    epoch_stats[class_idx] = {
                        "alpha": avg_alpha, 
                        "beta": avg_beta
                    }
                
                stats_history.append(epoch_stats)
                
            except Exception as e:
                print(f"Error processing epoch {epoch}: {e}")
                # Remove from epochs list to align with history
                selected_epochs.remove(epoch)

    # 5. Plot
    if stats_history:
        plot_beta_pdfs(stats_history, class_names, selected_epochs)
    else:
        print("No valid statistics collected.")

if __name__ == "__main__":
    main()
