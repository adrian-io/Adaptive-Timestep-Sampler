import subprocess
import time
import re
import os

# --- CONFIGURE RESUME HERE ---
# Leave values as None to start fresh.
# If resuming, provide the WandB ID and the path to the .pt file.
RESUME_CONFIG = {
    "bernoulli90": {
        "wandb_id": None, # e.g., "a1b2c3d4"
        "chkpt_path": None # e.g., "./chkpts/fm_bernoulli90_final/fm_bernoulli90_final_300.pt"
    },
    "bernoulli95": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "bernoulli_inv90": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "bernoulli_inv95": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "adaptive": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "uniform": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "ln": {
        "wandb_id": None,
        "chkpt_path": None
    },
    # Updated keys
    "beta_noise": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "beta_data": {
        "wandb_id": None,
        "chkpt_path": None
    },
    "a2c_ats": {          
        "wandb_id": None,
        "chkpt_path": None
    }
}

def run_experiment(sampler_type):
    exp_name = f"fm_{sampler_type}_final"
    exp_group = f"comparison_{sampler_type}" 
    print(f"==================================================")
    print(f"Starting Experiment: {sampler_type}")
    print(f"==================================================")
    
    cmd = [
        "python", "train.py",
        "--config-path", "configs/cifar10_fm.json",
        "--sampler-type", sampler_type,
        "--exp-name", exp_name,
        "--exp-group", exp_group, 
        "--eval",               
        "--distributed",        
        "--rigid-launch",       
        "--num-gpus", "4"       
    ]

    # --- INJECT RESUME ARGS ---
    config = RESUME_CONFIG.get(sampler_type, {})
    wandb_id = config.get("wandb_id")
    chkpt_path = config.get("chkpt_path")

    if wandb_id:
        print(f"Resuming WandB Run: {wandb_id}")
        cmd.extend(["--wandb-id", wandb_id])
    
    if chkpt_path:
        print(f"Resuming from Checkpoint: {chkpt_path}")
        cmd.extend(["--resume", "--chkpt-path", chkpt_path])
    # --------------------------
    
    start_time = time.time()
    
    # Run process and capture output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    try:
        fid_scores = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip()) # Mirror to console
                
                # Simple regex to catch the FID print we added in Train.py
                match = re.search(r"FID: (\d+\.\d+)", line)
                if match:
                    fid_scores.append(float(match.group(1)))
                    
    except KeyboardInterrupt:
        print("Interrupted! Killing subprocess...")
        process.kill()
        raise
    finally:
        # END OF RUN CLEANUP
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                
    duration = time.time() - start_time
    final_fid = fid_scores[-1] if fid_scores else "N/A"
    
    return duration, final_fid

def main():
    # strategies = ["adaptive", "uniform", "ln", "a2c_ats", "bernoulli90", "bernoulli95", "bernoulli_inv90", "bernoulli_inv95", "beta_noise", "beta_data"]
    strategies = ["a2c_ats", "bernoulli_inv95"]
    # strategies = ["bernoulli90", "bernoulli95", "bernoulli_inv90", "bernoulli_inv95"]
    results = {}
    
    print("Beginning Flow Matching Comparison Sequence...")
    
    for strategy in strategies:
        duration, fid = run_experiment(strategy)
        results[strategy] = {"time": duration, "fid": fid}
        
    print("\n\n")
    print("==================================================")
    print("COMPARISON RESULTS (Flow Matching 300 Epochs)")
    print("==================================================")
    print(f"{'Strategy':<15} | {'Runtime (s)':<15} | {'Final FID':<15}")
    print("-" * 50)
    for strategy, metrics in results.items():
        print(f"{strategy:<15} | {metrics['time']:<15.2f} | {metrics['fid']}")
    print("==================================================")

if __name__ == "__main__":
    main()
