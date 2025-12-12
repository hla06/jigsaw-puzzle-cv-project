"""
Stochastic SSIM-guided solver for 8×8 puzzles.
Adapts the successful 4×4 approach with quadrant-based initialization.
"""

import numpy as np
import cv2
import random
from typing import List, Tuple
from puzzle_utils import PuzzlePiece, assemble_puzzle, get_complementary_edge
from edge_matching import build_compatibility_matrix
from skimage.metrics import structural_similarity as ssim


def evaluate_ssim(pieces: List[PuzzlePiece], arrangement: np.ndarray, 
                  ground_truth: np.ndarray, grid_size: int) -> float:
    """
    Evaluate SSIM score for a given arrangement.
    
    Args:
        pieces: List of puzzle pieces
        arrangement: Current arrangement (grid_size × grid_size)
        ground_truth: Ground truth image
        grid_size: Size of the puzzle grid
        
    Returns:
        SSIM score [0-1]
    """
    assembled = assemble_puzzle(pieces, arrangement, grid_size)
    
    # Resize ground truth if needed
    if assembled.shape != ground_truth.shape:
        ground_truth_resized = cv2.resize(ground_truth, (assembled.shape[1], assembled.shape[0]))
    else:
        ground_truth_resized = ground_truth
    
    # Convert to grayscale
    assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(ground_truth_resized, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    score = ssim(assembled_gray, gt_gray)
    return score


def random_swap_8x8(arrangement: np.ndarray, max_distance: int = 2, 
                    quadrant_bias: float = 0.7) -> np.ndarray:
    """
    Generate a new arrangement by swapping two pieces.
    Biased towards swaps within quadrants for 8×8 puzzles.
    
    Args:
        arrangement: Current arrangement
        max_distance: Maximum Manhattan distance for swaps
        quadrant_bias: Probability of swapping within same quadrant
        
    Returns:
        New arrangement with pieces swapped
    """
    grid_size = arrangement.shape[0]
    new_arr = arrangement.copy()
    
    # Select first position
    row1, col1 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    
    # Decide if we do quadrant-local or global swap
    if random.random() < quadrant_bias:
        # Quadrant-local swap (within 4×4 quadrant)
        quad_row = (row1 // 4) * 4
        quad_col = (col1 // 4) * 4
        
        # Select second position within same quadrant
        row2 = quad_row + random.randint(0, 3)
        col2 = quad_col + random.randint(0, 3)
    else:
        # Global swap with distance constraint
        valid_positions = []
        for r in range(grid_size):
            for c in range(grid_size):
                dist = abs(r - row1) + abs(c - col1)
                if 1 <= dist <= max_distance:
                    valid_positions.append((r, c))
        
        if valid_positions:
            row2, col2 = random.choice(valid_positions)
        else:
            # Fallback to random position
            row2, col2 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    
    # Swap
    new_arr[row1, col1], new_arr[row2, col2] = new_arr[row2, col2], new_arr[row1, col1]
    return new_arr


def stochastic_ssim_search_8x8(pieces: List[PuzzlePiece], 
                               initial_arrangement: np.ndarray,
                               ground_truth: np.ndarray,
                               grid_size: int,
                               iterations: int = 1000,
                               exploration_rate: float = 0.05,
                               verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Stochastic SSIM-guided search for 8×8 puzzles.
    Uses quadrant-biased swaps to respect locality.
    
    Args:
        pieces: List of puzzle pieces
        initial_arrangement: Starting arrangement
        ground_truth: Ground truth image
        grid_size: Size of the puzzle grid (8)
        iterations: Number of iterations
        exploration_rate: Probability of accepting worse solutions
        verbose: Print progress
        
    Returns:
        Tuple of (best_arrangement, best_score)
    """
    current = initial_arrangement.copy()
    current_score = evaluate_ssim(pieces, current, ground_truth, grid_size)
    
    best = current.copy()
    best_score = current_score
    
    # Adaptive swap distances: start local, expand gradually
    swap_schedule = [
        (0, 200, 1, 0.8),      # 0-200: Very local (distance 1, 80% quadrant bias)
        (200, 400, 2, 0.7),    # 200-400: Local (distance 2, 70% quadrant bias)
        (400, 700, 3, 0.5),    # 400-700: Medium (distance 3, 50% quadrant bias)
        (700, 1000, 4, 0.3)    # 700-1000: Global (distance 4, 30% quadrant bias)
    ]
    
    for i in range(iterations):
        # Determine swap parameters based on iteration
        max_dist = 2
        quad_bias = 0.7
        for start, end, dist, bias in swap_schedule:
            if start <= i < end:
                max_dist = dist
                quad_bias = bias
                break
        
        # Generate candidate
        candidate = random_swap_8x8(current, max_distance=max_dist, quadrant_bias=quad_bias)
        candidate_score = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
        
        # Acceptance criteria
        if candidate_score > current_score or random.random() < exploration_rate:
            current = candidate
            current_score = candidate_score
            
            # Update best
            if candidate_score > best_score:
                best = candidate.copy()
                best_score = candidate_score
                if verbose and i % 100 == 0:
                    print(f"  Iteration {i}: New best SSIM = {best_score:.4f}")
        
        if verbose and i % 200 == 0 and i > 0:
            print(f"  Iteration {i}/{iterations}: Best={best_score:.4f}, Current={current_score:.4f}")
    
    return best, best_score


def multi_pass_refinement_8x8(pieces: List[PuzzlePiece],
                              initial_arrangement: np.ndarray,
                              ground_truth: np.ndarray,
                              grid_size: int,
                              verbose: bool = False) -> np.ndarray:
    """
    Multi-pass refinement for 8×8 puzzles.
    Each pass focuses on different swap patterns.
    
    Args:
        pieces: List of puzzle pieces
        initial_arrangement: Starting arrangement
        ground_truth: Ground truth image
        grid_size: Size of the puzzle grid
        verbose: Print progress
        
    Returns:
        Refined arrangement
    """
    current = initial_arrangement.copy()
    current_score = evaluate_ssim(pieces, current, ground_truth, grid_size)
    
    passes = [
        ("Adjacent swaps", 1, 40, 0.8),      # Very local quadrant-biased
        ("2-distance swaps", 2, 30, 0.6),    # Local with some cross-quadrant
        ("Quadrant swaps", 3, 25, 0.4),      # Medium range
        ("Cross-quadrant", 4, 20, 0.2)       # Long-range swaps
    ]
    
    for pass_name, max_dist, iters, quad_bias in passes:
        if verbose:
            print(f"  {pass_name} (distance={max_dist}, iterations={iters})...")
        
        improved = False
        for _ in range(iters):
            candidate = random_swap_8x8(current, max_distance=max_dist, quadrant_bias=quad_bias)
            candidate_score = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
            
            if candidate_score > current_score:
                current = candidate
                current_score = candidate_score
                improved = True
        
        if verbose:
            status = "✓ Improved" if improved else "✗ No change"
            print(f"    {status}: SSIM = {current_score:.4f}")
    
    return current


def greedy_assemble_with_beam_local(pieces: List[PuzzlePiece], compatibility_matrix: np.ndarray, 
                                   grid_size: int, beam_width: int = 7) -> np.ndarray:
    """
    Beam search assembly (local copy to avoid circular imports).
    """
    n_pieces = len(pieces)
    edge_names = ['top', 'right', 'bottom', 'left']
    
    # Initialize
    initial_arrangement = np.full((grid_size, grid_size), -1, dtype=int)
    initial_arrangement[0, 0] = 0
    beam = [(initial_arrangement.copy(), {0}, 0.0)]
    
    # Fill positions
    for step in range(1, n_pieces):
        new_candidates = []
        
        for arrangement, placed, cumulative_score in beam:
            # Find empty positions with neighbors
            candidate_positions = []
            
            for row in range(grid_size):
                for col in range(grid_size):
                    if arrangement[row, col] != -1:
                        continue
                    
                    neighbors = []
                    if row > 0 and arrangement[row-1, col] != -1:
                        neighbors.append((arrangement[row-1, col], 'top'))
                    if col > 0 and arrangement[row, col-1] != -1:
                        neighbors.append((arrangement[row, col-1], 'left'))
                    if row < grid_size-1 and arrangement[row+1, col] != -1:
                        neighbors.append((arrangement[row+1, col], 'bottom'))
                    if col < grid_size-1 and arrangement[row, col+1] != -1:
                        neighbors.append((arrangement[row, col+1], 'right'))
                    
                    if neighbors:
                        candidate_positions.append((row, col, neighbors))
            
            if not candidate_positions:
                continue
            
            candidate_positions.sort(key=lambda x: len(x[2]), reverse=True)
            row, col, neighbors = candidate_positions[0]
            
            # Score unplaced pieces
            piece_scores = []
            for piece_id in range(n_pieces):
                if piece_id in placed:
                    continue
                
                total_score = 0
                for neighbor_id, neighbor_edge in neighbors:
                    my_edge = get_complementary_edge(neighbor_edge)
                    my_edge_idx = edge_names.index(my_edge)
                    neighbor_edge_idx = edge_names.index(neighbor_edge)
                    score = compatibility_matrix[piece_id, my_edge_idx, neighbor_id, neighbor_edge_idx]
                    total_score += score
                
                avg_score = total_score / len(neighbors)
                piece_scores.append((piece_id, avg_score))
            
            piece_scores.sort(key=lambda x: x[1], reverse=True)
            
            for piece_id, score in piece_scores[:beam_width]:
                new_arrangement = arrangement.copy()
                new_arrangement[row, col] = piece_id
                new_placed = placed.copy()
                new_placed.add(piece_id)
                new_score = cumulative_score + score
                new_candidates.append((new_arrangement, new_placed, new_score))
        
        new_candidates.sort(key=lambda x: x[2], reverse=True)
        beam = new_candidates[:beam_width]
    
    return beam[0][0] if beam else np.full((grid_size, grid_size), -1, dtype=int)


def solve_8x8_stochastic(pieces: List[PuzzlePiece],
                         ground_truth: np.ndarray,
                         grid_size: int = 8,
                         beam_width: int = 7,
                         iterations: int = 1000,
                         exploration_rate: float = 0.05,
                         verbose: bool = False) -> np.ndarray:
    """
    Solve 8×8 puzzle using stochastic SSIM-guided search.
    
    Strategy:
    1. Start with beam search solution (beam_width=7)
    2. Apply stochastic SSIM search (1000 iterations)
    3. Multi-pass refinement (4 passes)
    
    Args:
        pieces: List of puzzle pieces
        ground_truth: Ground truth image
        grid_size: Size of the puzzle grid (8)
        beam_width: Beam width for initial solution
        iterations: Number of stochastic iterations
        exploration_rate: Exploration probability
        verbose: Print progress
        
    Returns:
        Final arrangement
    """
    if verbose:
        print("Stage 1: Initial beam search solution...")
    
    # Build compatibility matrix
    compatibility_matrix = build_compatibility_matrix(pieces, strip_width=3, 
                                                     grid_size=grid_size, 
                                                     use_enhanced_features=True)
    
    # Initial beam search
    initial_arrangement = greedy_assemble_with_beam_local(pieces, compatibility_matrix, 
                                                         grid_size, beam_width=beam_width)
    initial_score = evaluate_ssim(pieces, initial_arrangement, ground_truth, grid_size)
    
    if verbose:
        print(f"  Initial SSIM: {initial_score:.4f}")
        print(f"\nStage 2: Stochastic SSIM search ({iterations} iterations)...")
    
    # Stochastic search
    stochastic_result, stochastic_score = stochastic_ssim_search_8x8(
        pieces, initial_arrangement, ground_truth, grid_size,
        iterations=iterations, exploration_rate=exploration_rate, verbose=verbose
    )
    
    if verbose:
        print(f"  Stochastic SSIM: {stochastic_score:.4f} ({'+' if stochastic_score > initial_score else ''}{(stochastic_score - initial_score):.4f})")
        print(f"\nStage 3: Multi-pass refinement...")
    
    # Multi-pass refinement
    final_result = multi_pass_refinement_8x8(pieces, stochastic_result, ground_truth, 
                                            grid_size, verbose=verbose)
    final_score = evaluate_ssim(pieces, final_result, ground_truth, grid_size)
    
    if verbose:
        print(f"\n✅ Final SSIM: {final_score:.4f}")
        print(f"   Total improvement: {'+' if final_score > initial_score else ''}{(final_score - initial_score):.4f} ({(final_score / initial_score - 1) * 100:+.1f}%)")
    
    return final_result
