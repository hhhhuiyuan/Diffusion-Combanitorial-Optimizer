#run by $python gdc_plot.py

import re
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the test output file
# file_path = 'eval/mis100/2024-09-22/gdc_subopt/test_output.txt'
# plot_path = 'eval/mis100/2024-09-22/gdc_subopt/solved_cost_vs_guidance.png'
file_path = 'eval/tsp50_large/2024-09-22/gdc_epc49/test_output.txt'
plot_path = 'eval/tsp50_large/2024-09-22/gdc_epc49/solved_cost_vs_guidance.png'
gdc = True

# Define regex patterns to match the required values
seed_pattern = re.compile(r'seed:\s*(\d+)')
solved_cost_pattern = re.compile(r'\s*test/solved_cost_epoch\s+([\d.]+)')
gt_cost_pattern = re.compile(r'\s*test/gt_cost_epoch\s+([\d.]+)')
guidance_pattern = re.compile(r'guidance:\s*(\d+)')

if not gdc:
    # Initialize lists to store the extracted values
    seeds = []
    solved_costs = []
    gt_costs = []

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Check for seed
            seed_match = seed_pattern.search(line)
            if seed_match:
                seeds.append(int(seed_match.group(1)))
                continue
            
            # Check for solved cost
            solved_cost_match = solved_cost_pattern.search(line)
            if solved_cost_match:
                solved_costs.append(float(solved_cost_match.group(1)))
                continue
            
            # Check for gt cost
            gt_cost_match = gt_cost_pattern.search(line)
            if gt_cost_match:
                gt_costs.append(float(gt_cost_match.group(1)))
                continue

    # Print the extracted values
    print("Seeds:", seeds)
    print("Mean Solved Costs:", sum(solved_costs)/len(solved_costs))
    print("GT Costs:", gt_costs[0])

else:
    results_by_guidance = {}
    
    # Variables to keep track of the current guidance and seed
    current_guidance = None
    current_seed = None

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Check for guidance
            guidance_match = guidance_pattern.search(line)
            if guidance_match:
                current_guidance = int(guidance_match.group(1))
                if current_guidance not in results_by_guidance:
                    results_by_guidance[current_guidance] = {}
                continue
            
            # Check for seed
            seed_match = seed_pattern.search(line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                continue
            
            # Check for solved cost
            solved_cost_match = solved_cost_pattern.search(line)
            if solved_cost_match:
                solved_cost = float(solved_cost_match.group(1))
                current_res_slot = results_by_guidance[current_guidance]
                current_res_slot[current_seed] = solved_cost
                continue
            
            # Check for gt cost
            gt_cost_match = gt_cost_pattern.search(line)
            if gt_cost_match:
                gt_cost = float(gt_cost_match.group(1))
                continue

    # Print the grouped results
    print(f"GT Cost is {gt_cost}.")
    
    guidance_levels = sorted(results_by_guidance.keys())
    mean_solved_costs = []
    std_solved_costs = []

    for guidance in guidance_levels:
        costs = list(results_by_guidance[guidance].values())
        mean_solved_costs.append(np.mean(costs))
        std_solved_costs.append(np.std(costs))
    
    # Set figure size for better clarity
    plt.figure(figsize=(6, 4))

    # Improve plot aesthetics
    plt.errorbar(guidance_levels, mean_solved_costs, yerr=std_solved_costs, 
                fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, markersize=2, 
                color='steelblue', ecolor='orange')

    # Set axis labels with better fonts
    plt.xlabel(r'Guidance Strength $\gamma$', fontsize=12, labelpad=10)
    plt.ylabel('Avg. TSP Length', fontsize=12, labelpad=10)

    # Adjust ticks and labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Tighten layout and save the plot with higher resolution
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)