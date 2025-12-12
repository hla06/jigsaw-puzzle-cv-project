import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Visual comparison
PROJECT_ROOT = Path(".")
DATASET_ROOT = PROJECT_ROOT / "Jigsaw Puzzle Dataset" / "Gravity Falls"
CORRECT_DIR = DATASET_ROOT / "correct"

puzzle_id = 0
scrambled_path = DATASET_ROOT / "puzzle_2x2" / f"{puzzle_id}.jpg"
ground_truth_path = CORRECT_DIR / f"{puzzle_id}.png"

scrambled = cv2.imread(str(scrambled_path))
ground_truth = cv2.imread(str(ground_truth_path))

# Convert BGR to RGB for display
scrambled_rgb = cv2.cvtColor(scrambled, cv2.COLOR_BGR2RGB)
ground_truth_rgb = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

# Create difference image
diff = np.abs(scrambled.astype(float) - ground_truth.astype(float))
diff_display = (diff / diff.max() * 255).astype(np.uint8)
diff_display_rgb = cv2.cvtColor(diff_display, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(scrambled_rgb)
axes[0].set_title(f"Scrambled (puzzle_2x2/{puzzle_id}.jpg)", fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(ground_truth_rgb)
axes[1].set_title(f"Ground Truth (correct/{puzzle_id}.png)", fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(diff_display_rgb)
axes[2].set_title(f"Difference (Mean: {np.mean(diff):.1f})", fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('puzzle_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved comparison to puzzle_comparison.png")
print(f"\nImage details:")
print(f"Scrambled: {scrambled.shape}, dtype={scrambled.dtype}")
print(f"Ground truth: {ground_truth.shape}, dtype={ground_truth.dtype}")
print(f"Mean absolute difference per pixel: {np.mean(diff):.2f}/255")
