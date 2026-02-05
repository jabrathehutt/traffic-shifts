import subprocess
import time
import psutil
import pandas as pd
import re
import os
import sys

# --- CONFIGURATION: Absolute Paths ---
MODELS = {
    "Lag-Llama": "/root/traffic-shifts/llama/lag-llama/detector22.py",
    "Seasonal-LSTM": "/root/traffic-shifts/lstm_detector.py",
    "Fast-ARIMA": "/root/traffic-shifts/sarima_detector.py",
    "Prophet": "/root/traffic-shifts/prophet_detector.py"
}

OUTPUT_CSV = "model_performance_comparison.csv"

def monitor_resources(proc_pid):
    try:
        p = psutil.Process(proc_pid)
        return p.memory_info().rss / (1024 * 1024) 
    except:
        return 0

def extract_metrics(output):
    metrics = {
        "Precision": 0.0, "Recall": 0.0, "F1": 0.0, 
        "Delay": 0.0, "Latency": 0.0
    }
    
    p = re.search(r"PRECISION:\s*([\d.]+)", output, re.IGNORECASE)
    r = re.search(r"RECALL:\s*([\d.]+)", output, re.IGNORECASE)
    f = re.search(r"F1(?:-SCORE| SCORE)?:\s*([\d.]+)", output, re.IGNORECASE)
    d = re.search(r"(?:AVG DETECTION )?DELAY:\s*([\d.]+)", output, re.IGNORECASE)
    l = re.search(r"INFERENCE LATENCY:\s*([\d.]+)", output, re.IGNORECASE)

    if p: metrics["Precision"] = float(p.group(1))
    if r: metrics["Recall"] = float(r.group(1))
    if f: metrics["F1"] = float(f.group(1))
    if d: metrics["Delay"] = float(d.group(1))
    if l: metrics["Latency"] = float(l.group(1))
    
    return metrics

def run_benchmark():
    results = []

    for name, filepath in MODELS.items():
        if not os.path.exists(filepath):
            print(f"Skipping {name}: {filepath} not found.")
            continue

        print(f"\n{'='*20} RUNNING: {name} {'='*20}")
        script_dir = os.path.dirname(filepath)
        
        start_time = time.time()
        peak_memory = 0
        full_output = []

        process = subprocess.Popen(
            ["python3", "-u", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=script_dir if name == "Lag-Llama" else None
        )

        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    full_output.append(line)
                    
                    current_mem = monitor_resources(process.pid)
                    if current_mem > peak_memory:
                        peak_memory = current_mem
        except KeyboardInterrupt:
            process.kill()
            sys.exit()

        process.wait()
        end_time = time.time()
        
        final_stdout = "".join(full_output)
        metrics = extract_metrics(final_stdout)
        
        results.append({
            "Model": name,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1"],
            "Avg_Delay_Mins": metrics["Delay"],
            "Inf_Latency_Sec": metrics["Latency"],
            "Total_Runtime_Sec": round(end_time - start_time, 2),
            "Peak_Memory_MB": round(peak_memory, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\n{'*'*60}\nBENCHMARK COMPLETE: {OUTPUT_CSV}\n{'*'*60}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
