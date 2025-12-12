# Test SSIM-based solver on a single puzzle with verbose output

from pathlib import Path
import cv2
import numpy as np
from puzzle_utils import create_puzzle_pieces, assemble_puzzle
from edge_matching import build_compatibility_matrix
from exhaustive_solver import exhaustive_solve_with_ssim
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
ground_truth_path = CORRECT_DIR / f"{puzzle_id}.png"

# Create pieces
pieces = create_puzzle_pieces(
    str(original_path),
    str(enhanced_path),
    str(mask_path),
    grid_size
)

# Load ground truth
ground_truth = cv2.imread(str(ground_truth_path))

print("Original positions:")
for i, piece in enumerate(pieces):
    print(f"  Piece {i}: original_pos = {piece.original_pos}")

print("\n" + "="*70)
print("Running SSIM-based exhaustive solver...")
print("="*70)
arrangement, best_ssim = exhaustive_solve_with_ssim(pieces, ground_truth, verbose=True)

print(f"\nFinal arrangement:\n{arrangement}")
print(f"Best SSIM: {best_ssim:.4f}")

# Compute accuracy
accuracy = compute_piece_placement_accuracy(arrangement, pieces)
print(f"Piece accuracy: {accuracy:.1f}%")

# Show what the ground truth arrangement should be
print("\n" + "="*70)
print("What if pieces are ALREADY correct as-is?")
print("="*70)
identity_arrangement = np.array([[0, 1], [2, 3]])
identity_accuracy = compute_piece_placement_accuracy(identity_arrangement, pieces)
print(f"Identity arrangement piece accuracy: {identity_accuracy:.1f}%")

# Assemble and compare
assembled_best = assemble_puzzle(pieces, arrangement, grid_size)
assembled_identity = assemble_puzzle(pieces, identity_arrangement, grid_size)

print(f"\nVisual comparison:")
print(f"  Best arrangement vs GT - Mean diff: {np.mean(np.abs(assembled_best.astype(float) - ground_truth.astype(float))):.2f}")
print(f"  Identity arrangement vs GT - Mean diff: {np.mean(np.abs(assembled_identity.astype(float) - ground_truth.astype(float))):.2f}")
