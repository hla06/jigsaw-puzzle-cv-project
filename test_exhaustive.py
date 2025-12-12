# Test exhaustive solver on a single puzzle

from pathlib import Path
import cv2
import numpy as np
from puzzle_utils import create_puzzle_pieces
from edge_matching import build_compatibility_matrix
from exhaustive_solver import exhaustive_solve_2x2, score_2x2_arrangement
from validation import compute_piece_placement_accuracy

# Paths
PROJECT_ROOT = Path(".")
DATASET_ROOT = PROJECT_ROOT / "Jigsaw Puzzle Dataset" / "Gravity Falls"
OUTPUT_ROOT = PROJECT_ROOT / "processed_images"
ENHANCED_DIR = OUTPUT_ROOT / "enhanced"
MASK_DIR = OUTPUT_ROOT / "masks"
CORRECT_DIR = DATASET_ROOT / "correct"

# Test on puzzle 0
puzzle_id = 0
puzzle_folder = "puzzle_2x2"
grid_size = 2

original_path = DATASET_ROOT / puzzle_folder / f"{puzzle_id}.jpg"
enhanced_path = ENHANCED_DIR / f"{puzzle_folder}_{puzzle_id}.jpg"
mask_path = MASK_DIR / f"{puzzle_folder}_{puzzle_id}.jpg"

# Create pieces
pieces = create_puzzle_pieces(
    str(original_path),
    str(enhanced_path),
    str(mask_path),
    grid_size
)

print("Original positions:")
for i, piece in enumerate(pieces):
    print(f"  Piece {i}: original_pos = {piece.original_pos}")

# Build compatibility matrix
compatibility_matrix = build_compatibility_matrix(pieces, strip_width=3)

# Test exhaustive solver
print("\n" + "="*70)
print("Running exhaustive solver...")
print("="*70)
arrangement, best_score = exhaustive_solve_2x2(pieces, compatibility_matrix, verbose=True)

print(f"\nFinal arrangement:\n{arrangement}")
print(f"Best score: {best_score:.4f}")

# Compute accuracy
accuracy = compute_piece_placement_accuracy(arrangement, pieces)
print(f"Piece accuracy: {accuracy:.1f}%")

# Test correct arrangement
correct_arrangement = np.array([[0, 1], [2, 3]])
correct_score = score_2x2_arrangement(correct_arrangement, compatibility_matrix)
print(f"\nCorrect arrangement score: {correct_score:.4f}")
print(f"Correct arrangement:\n{correct_arrangement}")
