import subprocess
import time
import re

def run_experiment(sampler_type):
    exp_name = f"fm_{sampler_type}_final"
    exp_group = f"comparison_{sampler_type}"  # NEW: Define group name
    print(f"==================================================")
    print(f"Starting Experiment: {sampler_type}")
    print(f"==================================================")
    
    cmd = [
        "python", "train.py",
        "--config-path", "configs/cifar10_fm.json",
        "--sampler-type", sampler_type,
        "--exp-name", exp_name,
        "--exp-group", exp_group,  # NEW: Pass group name
        "--eval",               # Enable evaluation
        "--eval-total-size", "1000",  # <--- ADD THIS LINE
        "--eval-batch-size", "64",   # <--- Optional: Adjust batch size
        "--distributed",        # Enable distributed mode
        "--rigid-launch",       # Use the specific launch mode for this codebase
        "--num-gpus", "2"       # Specify the number of GPUs (match CUDA_VISIBLE_DEVICES count)
    ]
    
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
    strategies = ["adaptive", "uniform", "ln"]
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
