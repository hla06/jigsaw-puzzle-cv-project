import cv2
import numpy as np
from pathlib import Path

# Compare scrambled vs ground truth visually
PROJECT_ROOT = Path(".")
DATASET_ROOT = PROJECT_ROOT / "Jigsaw Puzzle Dataset" / "Gravity Falls"
CORRECT_DIR = DATASET_ROOT / "correct"

for puzzle_id in [0, 1, 2]:
    scrambled_path = DATASET_ROOT / "puzzle_2x2" / f"{puzzle_id}.jpg"
    ground_truth_path = CORRECT_DIR / f"{puzzle_id}.png"
    
    scrambled = cv2.imread(str(scrambled_path))
    ground_truth = cv2.imread(str(ground_truth_path))
    
    # Compute difference
    if scrambled.shape == ground_truth.shape:
        diff = np.abs(scrambled.astype(float) - ground_truth.astype(float))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        print(f"Puzzle {puzzle_id}:")
        print(f"  Mean pixel difference: {mean_diff:.2f}")
        print(f"  Max pixel difference: {max_diff:.2f}")
        print(f"  Are they identical? {np.allclose(scrambled, ground_truth, atol=5)}")
    else:
        print(f"Puzzle {puzzle_id}: Shape mismatch!")
