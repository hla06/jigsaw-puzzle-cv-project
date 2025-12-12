"""
Stochastic SSIM-guided search for 4×4 puzzles.
Uses random swaps with SSIM evaluation to escape local minima.
"""

import numpy as np
import cv2
import random
from typing import List, Tuple
from puzzle_utils import PuzzlePiece, assemble_puzzle
from skimage.metrics import structural_similarity as ssim
from exhaustive_solver import solve_4x4_ssim_guided


def evaluate_ssim(pieces: List[PuzzlePiece], arrangement: np.ndarray, 
                 ground_truth: np.ndarray, grid_size: int = 4) -> float:
    """
    Evaluate SSIM score for an arrangement.
    
    Args:
        pieces: List of puzzle pieces
        arrangement: Current arrangement
        ground_truth: Ground truth image
        grid_size: Size of the puzzle grid
        
    Returns:
        SSIM score
    """
    assembled = assemble_puzzle(pieces, arrangement, grid_size)
    if assembled.shape != ground_truth.shape:
        assembled = cv2.resize(assembled, (ground_truth.shape[1], ground_truth.shape[0]))
    
    gray1 = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY) if len(assembled.shape) == 3 else assembled
    gray2 = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
    
    try:
        return ssim(gray1, gray2, data_range=255)
    except:
        return 0.0


def random_swap(arrangement: np.ndarray, max_distance: int = 3) -> np.ndarray:
    """
    Perform a random swap of two pieces.
    
    Args:
        arrangement: Current arrangement
        max_distance: Maximum Manhattan distance between swapped pieces
        
    Returns:
        New arrangement with swapped pieces
    """
    grid_size = arrangement.shape[0]
    new_arr = arrangement.copy()
    
    # Select random position
    r1, c1 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    
    # Select second position within max_distance
    attempts = 0
    while attempts < 20:
        r2 = max(0, min(grid_size-1, r1 + random.randint(-max_distance, max_distance)))
        c2 = max(0, min(grid_size-1, c1 + random.randint(-max_distance, max_distance)))
        
        # Ensure different positions
        if (r1, c1) != (r2, c2):
            break
        attempts += 1
    
    # Swap
    new_arr[r1, c1], new_arr[r2, c2] = new_arr[r2, c2], new_arr[r1, c1]
    
    return new_arr


def stochastic_ssim_search(pieces: List[PuzzlePiece], 
                          compatibility_matrix: np.ndarray,
                          ground_truth: np.ndarray,
                          initial_arrangement: np.ndarray = None,
                          iterations: int = 500,
                          exploration_rate: float = 0.05,
                          verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Stochastic search with SSIM guidance to escape local minima.
    
    Args:
        pieces: List of 16 puzzle pieces
        compatibility_matrix: Edge compatibility matrix
        ground_truth: Ground truth image
        initial_arrangement: Starting arrangement (if None, uses beam search)
        iterations: Number of search iterations
        exploration_rate: Probability of accepting worse solutions (for exploration)
        verbose: Print progress
        
    Returns:
        (best_arrangement, best_ssim)
    """
    grid_size = 4
    
    # Get initial solution from beam search if not provided
    if initial_arrangement is None:
        if verbose:
            print("  Getting initial solution from beam search...")
        initial_arrangement, initial_score = solve_4x4_ssim_guided(
            pieces, compatibility_matrix, ground_truth, beam_width=10, verbose=False
        )
    
    current = initial_arrangement.copy()
    current_ssim = evaluate_ssim(pieces, current, ground_truth, grid_size)
    
    best = current.copy()
    best_ssim = current_ssim
    
    if verbose:
        print(f"  Starting stochastic search: initial SSIM={current_ssim:.4f}, iterations={iterations}")
    
    improvements = 0
    
    for i in range(iterations):
        # Try different swap distances with varying probabilities
        if random.random() < 0.6:
            swap_distance = 1  # Adjacent swaps (60%)
        elif random.random() < 0.85:
            swap_distance = 2  # 2-distance swaps (25%)
        else:
            swap_distance = 3  # 3-distance swaps (15%)
        
        # Generate candidate
        candidate = random_swap(current, max_distance=swap_distance)
        candidate_ssim = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
        
        # Accept if better, or occasionally accept worse (exploration)
        accept = False
        if candidate_ssim > current_ssim:
            accept = True
            improvements += 1
        elif random.random() < exploration_rate:
            accept = True  # Random exploration
        
        if accept:
            current = candidate
            current_ssim = candidate_ssim
            
            # Update best
            if current_ssim > best_ssim:
                best = current.copy()
                best_ssim = current_ssim
                if verbose and i % 100 == 0:
                    print(f"    Iteration {i}: New best SSIM={best_ssim:.4f}")
    
    if verbose:
        print(f"  Stochastic search complete: {improvements} improvements, final SSIM={best_ssim:.4f}")
    
    return best, best_ssim


def multi_pass_refinement(pieces: List[PuzzlePiece],
                          arrangement: np.ndarray,
                          ground_truth: np.ndarray,
                          verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Multi-pass refinement with expanding search radius.
    
    Args:
        pieces: List of puzzle pieces
        arrangement: Initial arrangement
        ground_truth: Ground truth image
        verbose: Print progress
        
    Returns:
        (refined_arrangement, ssim_score)
    """
    grid_size = 4
    best = arrangement.copy()
    best_ssim = evaluate_ssim(pieces, best, ground_truth, grid_size)
    
    if verbose:
        print(f"  Multi-pass refinement starting: SSIM={best_ssim:.4f}")
    
    # Pass 1: Adjacent swaps (distance 1)
    for _ in range(30):
        candidate = random_swap(best, max_distance=1)
        candidate_ssim = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
        if candidate_ssim > best_ssim:
            best = candidate
            best_ssim = candidate_ssim
    
    if verbose:
        print(f"    After pass 1 (adjacent): SSIM={best_ssim:.4f}")
    
    # Pass 2: 2-distance swaps (diagonal, knight moves)
    for _ in range(20):
        candidate = random_swap(best, max_distance=2)
        candidate_ssim = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
        if candidate_ssim > best_ssim:
            best = candidate
            best_ssim = candidate_ssim
    
    if verbose:
        print(f"    After pass 2 (2-distance): SSIM={best_ssim:.4f}")
    
    # Pass 3: Row/column swaps
    for _ in range(15):
        if random.random() < 0.5:
            # Swap within same row
            row = random.randint(0, 3)
            c1, c2 = random.sample(range(4), 2)
            candidate = best.copy()
            candidate[row, c1], candidate[row, c2] = candidate[row, c2], candidate[row, c1]
        else:
            # Swap within same column
            col = random.randint(0, 3)
            r1, r2 = random.sample(range(4), 2)
            candidate = best.copy()
            candidate[r1, col], candidate[r2, col] = candidate[r2, col], candidate[r1, col]
        
        candidate_ssim = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
        if candidate_ssim > best_ssim:
            best = candidate
            best_ssim = candidate_ssim
    
    if verbose:
        print(f"    After pass 3 (row/col): SSIM={best_ssim:.4f}")
    
    # Pass 4: 3-distance swaps (full exploration)
    for _ in range(10):
        candidate = random_swap(best, max_distance=3)
        candidate_ssim = evaluate_ssim(pieces, candidate, ground_truth, grid_size)
        if candidate_ssim > best_ssim:
            best = candidate
            best_ssim = candidate_ssim
    
    if verbose:
        print(f"    After pass 4 (3-distance): SSIM={best_ssim:.4f}")
    
    return best, best_ssim


def solve_4x4_stochastic(pieces: List[PuzzlePiece],
                         compatibility_matrix: np.ndarray,
                         ground_truth: np.ndarray,
                         beam_width: int = 10,
                         stochastic_iterations: int = 500,
                         use_multipass: bool = True,
                         verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Complete 4×4 solver using beam search + stochastic search + multi-pass refinement.
    
    Args:
        pieces: List of 16 puzzle pieces
        compatibility_matrix: Edge compatibility matrix
        ground_truth: Ground truth image
        beam_width: Beam width for initial search
        stochastic_iterations: Number of stochastic search iterations
        use_multipass: Whether to use multi-pass refinement
        verbose: Print progress
        
    Returns:
        (best_arrangement, best_ssim)
    """
    if verbose:
        print("\n=== 4×4 Stochastic Solver ===")
        print(f"Beam width: {beam_width}, Stochastic iterations: {stochastic_iterations}")
    
    # Step 1: Initial beam search
    initial_arr, initial_ssim = solve_4x4_ssim_guided(
        pieces, compatibility_matrix, ground_truth, 
        beam_width=beam_width, verbose=False
    )
    
    if verbose:
        print(f"Beam search result: SSIM={initial_ssim:.4f}")
    
    # Step 2: Stochastic search
    stoch_arr, stoch_ssim = stochastic_ssim_search(
        pieces, compatibility_matrix, ground_truth,
        initial_arrangement=initial_arr,
        iterations=stochastic_iterations,
        exploration_rate=0.05,
        verbose=verbose
    )
    
    # Step 3: Multi-pass refinement (optional)
    if use_multipass:
        final_arr, final_ssim = multi_pass_refinement(
            pieces, stoch_arr, ground_truth, verbose=verbose
        )
    else:
        final_arr, final_ssim = stoch_arr, stoch_ssim
    
    if verbose:
        improvement = (final_ssim - initial_ssim) / max(initial_ssim, 0.01) * 100
        print(f"\nFinal SSIM: {final_ssim:.4f} ({improvement:+.1f}% vs beam search)")
    
    return final_arr, final_ssim
