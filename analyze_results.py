#!/usr/bin/env python

import os
import csv
import numpy as np

def analyze_csv_file(filepath):
    """
    Reads a CSV file with header ["accuracy", "classify_time"] and multiple rows.
    Computes mean and std for each column. Returns (acc_mean, acc_std, time_mean, time_std).
    """
    accuracies = []
    times = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        
        for row in reader:
            if len(row) < 2:
                continue  # skip invalid lines
            
            acc = float(row[0])
            t = float(row[1])
            
            accuracies.append(acc)
            times.append(t)
    
    accuracies = np.array(accuracies)
    times = np.array(times)
    
    acc_mean = accuracies.mean()
    acc_std  = accuracies.std()
    time_mean = times.mean()
    time_std  = times.std()
    
    return acc_mean, acc_std, time_mean, time_std

def main():
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}. Run experiments first.")
        return
    
    # We'll look at all CSV files in ./results
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {results_dir}.")
        return
    
    print("Experiment Analysis (Mean ± Std over runs)\n")
    
    for csv_file in csv_files:
        filepath = os.path.join(results_dir, csv_file)
        acc_mean, acc_std, time_mean, time_std = analyze_csv_file(filepath)
        
        print(f"File: {csv_file}")
        print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f}")
        print(f"  Time    : {time_mean:.2f}s ± {time_std:.2f}s\n")

if __name__ == "__main__":
    main()
