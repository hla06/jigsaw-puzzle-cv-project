# Check if puzzle pieces actually know their correct positions

from pathlib import Path
import cv2
import numpy as np
from puzzle_utils import create_puzzle_pieces, assemble_puzzle
from validation import compute_piece_placement_accuracy

# Paths
PROJECT_ROOT = Path(".")
DATASET_ROOT = PROJECT_ROOT / "Jigsaw Puzzle Dataset" / "Gravity Falls"
OUTPUT_ROOT = PROJECT_ROOT / "processed_images"
ENHANCED_DIR = OUTPUT_ROOT / "enhanced"
MASK_DIR = OUTPUT_ROOT / "masks"
CORRECT_DIR = DATASET_ROOT / "correct"

# Test on a few puzzles
for puzzle_id in [0, 1, 2, 7, 8]:
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
    
    print(f"\nPuzzle {puzzle_id}:")
    print("  Piece original_pos values:")
    for i, piece in enumerate(pieces):
        print(f"    Piece {i}: {piece.original_pos}")
    
    # Test: if we arrange pieces by their IDs in order, what's the accuracy?
    identity_arrangement = np.array([[0, 1], [2, 3]])
    accuracy = compute_piece_placement_accuracy(identity_arrangement, pieces)
    print(f"  Identity arrangement accuracy: {accuracy:.1f}%")
    
    # Visualize: assemble using identity arrangement
    assembled = assemble_puzzle(pieces, identity_arrangement, grid_size)
    ground_truth = cv2.imread(str(ground_truth_path))
    
    # Compare dimensions
    print(f"  Assembled shape: {assembled.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    
    # If dimensions match, check if they're similar
    if assembled.shape == ground_truth.shape:
        diff = np.mean(np.abs(assembled.astype(float) - ground_truth.astype(float)))
        print(f"  Mean pixel difference: {diff:.2f}")
