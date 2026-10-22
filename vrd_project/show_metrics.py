import os

log_file = "outputs/predcls_20260328_025704/train_20260328_025704.log"

if not os.path.exists(log_file):
    print(f"Error: Could not find {log_file}")
    exit(1)

with open(log_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split the log by the Evaluation Banner
blocks = text.split("==================================================\nEvaluation Results (PREDCLS)\n==================================================")

if len(blocks) < 2:
    print("No evaluation blocks found in the log.")
    exit(1)

# Find the block that corresponds to the highest R@50
best_block_text = ""
best_r50 = 0.0

# The blocks[1:] contain the evaluation logs + the line after it which tells us if it was the best
for block in blocks[1:]:
    # Find the R@50 line via simple string matching
    lines = block.split("\n")
    current_r50 = 0.0
    for line in lines:
        if line.startswith("R@50:"):
            # Extract the percentage
            val = float(line.replace("R@50:", "").replace("%", "").strip())
            current_r50 = val
            break
            
    if current_r50 > best_r50:
        best_r50 = current_r50
        # The relevant part of the block is everything up to the next "========================"
        eval_metrics = block.split("==================================================")[0]
        best_block_text = "\n==================================================\nEvaluation Results (PREDCLS)\n==================================================" + eval_metrics + "=================================================="

print("\n🚀 FOUND THE BEST EPOCH METRICS IN THE LOGS 🚀")
print(best_block_text)
