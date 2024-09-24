#run by $python eval/analysis.py

import re

# Define the path to the test output file
file_path = 'eval/tsp100_mid/2024-09-23/gdc_epc47_noedge/test_output.txt'
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
    for guidance, results in results_by_guidance.items():
        print(f"Guidance: {guidance}")
        # for seed, cost in results.items():
        #     print(f" Seed: {seed}, Solved Cost: {cost}")
        all_costs = list(results.values())
        print(f"Avg solved cost: {sum(all_costs)/len(all_costs)}")