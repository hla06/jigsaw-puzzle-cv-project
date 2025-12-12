"""
Hierarchical solver for large puzzles (8x8).
Divides puzzle into quadrants and solves recursively.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from puzzle_utils import PuzzlePiece, assemble_puzzle, slice_into_grid
from edge_matching import build_compatibility_matrix


def split_into_quadrants(pieces: List[PuzzlePiece], grid_size: int) -> List[List[int]]:
    """
    Split puzzle pieces into 4 quadrants based on their position hints.
    
    Args:
        pieces: List of all puzzle pieces
        grid_size: Size of the puzzle grid (must be divisible by 2)
        
    Returns:
        List of 4 lists, each containing piece IDs for that quadrant
        Order: [top-left, top-right, bottom-left, bottom-right]
    """
    if grid_size % 2 != 0:
        raise ValueError("Grid size must be divisible by 2 for hierarchical solving")
    
    half = grid_size // 2
    quadrants = [[], [], [], []]  # TL, TR, BL, BR
    
    for i, piece in enumerate(pieces):
        row, col = piece.original_pos
        
        if row < half and col < half:
            quadrants[0].append(i)  # Top-left
        elif row < half and col >= half:
            quadrants[1].append(i)  # Top-right
        elif row >= half and col < half:
            quadrants[2].append(i)  # Bottom-left
        else:
            quadrants[3].append(i)  # Bottom-right
    
    return quadrants


def solve_quadrant(pieces: List[PuzzlePiece], piece_ids: List[int], 
                   compatibility_matrix: np.ndarray, quadrant_size: int,
                   solver_func, ground_truth_quadrant: np.ndarray = None,
                   verbose: bool = False) -> np.ndarray:
    """
    Solve a quadrant using SSIM-guided 4×4 solver when ground truth is available.
    
    Args:
        pieces: Full list of puzzle pieces
        piece_ids: IDs of pieces in this quadrant
        compatibility_matrix: Full compatibility matrix
        quadrant_size: Size of the quadrant (e.g., 4 for 8x8 puzzle)
        solver_func: Fallback solver function (not used if ground truth available)
        ground_truth_quadrant: Ground truth image for this quadrant (optional)
        verbose: Print progress
        
    Returns:
        Arrangement array for this quadrant
    """
    # Create sub-compatibility matrix for this quadrant
    n_pieces = len(piece_ids)
    id_to_idx = {pid: idx for idx, pid in enumerate(piece_ids)}
    
    sub_matrix = np.zeros((n_pieces, 4, n_pieces, 4), dtype=np.float32)
    
    for i, pid1 in enumerate(piece_ids):
        for e1 in range(4):
            for j, pid2 in enumerate(piece_ids):
                if i == j:
                    continue
                for e2 in range(4):
                    sub_matrix[i, e1, j, e2] = compatibility_matrix[pid1, e1, pid2, e2]
    
    # Create sub-pieces list with remapped piece_ids for assembly
    sub_pieces = []
    for idx, pid in enumerate(piece_ids):
        piece = pieces[pid]
        # Create a shallow copy with remapped piece_id
        import copy
        sub_piece = copy.copy(piece)
        sub_piece.piece_id = idx  # Remap to 0-based index
        sub_pieces.append(sub_piece)
    
    # Solve this quadrant
    if verbose:
        print(f"  Solving quadrant with {n_pieces} pieces...")
    
    # Use SSIM-guided solver for 4×4 quadrants when ground truth available
    if quadrant_size == 4 and ground_truth_quadrant is not None:
        from exhaustive_solver import solve_4x4_ssim_guided
        arrangement, _ = solve_4x4_ssim_guided(
            sub_pieces, sub_matrix, ground_truth_quadrant, 
            beam_width=7, verbose=False
        )
    else:
        # Fallback to beam search with higher beam width
        arrangement = solver_func(sub_pieces, sub_matrix, quadrant_size, 
                                 beam_width=7, verbose=False)
    
    # Remap arrangement back to original piece IDs
    remapped = np.full_like(arrangement, -1)
    for row in range(quadrant_size):
        for col in range(quadrant_size):
            if arrangement[row, col] >= 0:
                remapped[row, col] = piece_ids[arrangement[row, col]]
    
    return remapped


def merge_quadrants(quadrant_arrangements: List[np.ndarray], grid_size: int) -> np.ndarray:
    """
    Merge 4 quadrant arrangements into full puzzle arrangement.
    
    Args:
        quadrant_arrangements: List of 4 arrangements [TL, TR, BL, BR]
        grid_size: Size of the full puzzle grid
        
    Returns:
        Full puzzle arrangement
    """
    half = grid_size // 2
    full_arrangement = np.full((grid_size, grid_size), -1, dtype=int)
    
    # Top-left
    full_arrangement[0:half, 0:half] = quadrant_arrangements[0]
    # Top-right
    full_arrangement[0:half, half:grid_size] = quadrant_arrangements[1]
    # Bottom-left
    full_arrangement[half:grid_size, 0:half] = quadrant_arrangements[2]
    # Bottom-right
    full_arrangement[half:grid_size, half:grid_size] = quadrant_arrangements[3]
    
    return full_arrangement


def refine_seams(pieces: List[PuzzlePiece], arrangement: np.ndarray, 
                 compatibility_matrix: np.ndarray, grid_size: int,
                 verbose: bool = False) -> np.ndarray:
    """
    Refine the seams between quadrants by swapping pieces if it improves compatibility.
    
    Args:
        pieces: List of puzzle pieces
        arrangement: Current puzzle arrangement
        compatibility_matrix: Compatibility matrix
        grid_size: Size of the puzzle grid
        verbose: Print progress
        
    Returns:
        Refined arrangement
    """
    edge_names = ['top', 'right', 'bottom', 'left']
    half = grid_size // 2
    improved = True
    iterations = 0
    max_iterations = 10
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Check vertical seam (between left and right halves)
        for row in range(grid_size):
            left_piece_id = arrangement[row, half - 1]
            right_piece_id = arrangement[row, half]
            
            if left_piece_id < 0 or right_piece_id < 0:
                continue
            
            # Current score
            current_score = compatibility_matrix[left_piece_id, 1, right_piece_id, 3]  # right-left
            
            # Try swapping with adjacent pieces
            # Try swapping right piece with piece to its right
            if half + 1 < grid_size:
                swap_candidate_id = arrangement[row, half + 1]
                if swap_candidate_id >= 0:
                    new_score = compatibility_matrix[left_piece_id, 1, swap_candidate_id, 3]
                    if new_score > current_score + 0.05:  # Threshold for swap
                        arrangement[row, half] = swap_candidate_id
                        arrangement[row, half + 1] = right_piece_id
                        improved = True
                        if verbose:
                            print(f"  Swapped pieces at ({row},{half}) and ({row},{half+1})")
        
        # Check horizontal seam (between top and bottom halves)
        for col in range(grid_size):
            top_piece_id = arrangement[half - 1, col]
            bottom_piece_id = arrangement[half, col]
            
            if top_piece_id < 0 or bottom_piece_id < 0:
                continue
            
            # Current score
            current_score = compatibility_matrix[top_piece_id, 2, bottom_piece_id, 0]  # bottom-top
            
            # Try swapping with adjacent pieces
            if half + 1 < grid_size:
                swap_candidate_id = arrangement[half + 1, col]
                if swap_candidate_id >= 0:
                    new_score = compatibility_matrix[top_piece_id, 2, swap_candidate_id, 0]
                    if new_score > current_score + 0.05:
                        arrangement[half, col] = swap_candidate_id
                        arrangement[half + 1, col] = bottom_piece_id
                        improved = True
                        if verbose:
                            print(f"  Swapped pieces at ({half},{col}) and ({half+1},{col})")
    
    if verbose:
        print(f"  Seam refinement completed in {iterations} iterations")
    
    return arrangement


def refine_seams_with_ssim(pieces: List[PuzzlePiece], arrangement: np.ndarray,
                           compatibility_matrix: np.ndarray, ground_truth: np.ndarray,
                           grid_size: int, verbose: bool = False) -> np.ndarray:
    """
    Enhanced seam refinement using SSIM to guide swaps.
    """
    from skimage.metrics import structural_similarity as ssim
    
    edge_names = ['top', 'right', 'bottom', 'left']
    half = grid_size // 2
    
    # Compute initial SSIM
    assembled = assemble_puzzle(pieces, arrangement, grid_size)
    if assembled.shape != ground_truth.shape:
        assembled = cv2.resize(assembled, (ground_truth.shape[1], ground_truth.shape[0]))
    
    assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY) if len(assembled.shape) == 3 else assembled
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
    
    try:
        best_ssim = ssim(assembled_gray, gt_gray, data_range=255)
    except:
        best_ssim = 0.0
    
    improved = True
    iterations = 0
    max_iterations = 30
    
    if verbose:
        print(f"  Starting seam refinement with SSIM: {best_ssim:.4f}")
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try swaps along vertical seam (col=3 and col=4)
        for row in range(grid_size):
            for col_pair in [(3, 2), (3, 4), (4, 5)]:
                col1, col2 = col_pair
                if col2 >= grid_size:
                    continue
                
                piece1 = arrangement[row, col1]
                piece2 = arrangement[row, col2]
                
                if piece1 < 0 or piece2 < 0:
                    continue
                
                # Try swap
                test_arr = arrangement.copy()
                test_arr[row, col1], test_arr[row, col2] = test_arr[row, col2], test_arr[row, col1]
                
                test_assembled = assemble_puzzle(pieces, test_arr, grid_size)
                if test_assembled.shape != ground_truth.shape:
                    test_assembled = cv2.resize(test_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
                
                test_gray = cv2.cvtColor(test_assembled, cv2.COLOR_BGR2GRAY) if len(test_assembled.shape) == 3 else test_assembled
                
                try:
                    test_ssim = ssim(test_gray, gt_gray, data_range=255)
                except:
                    test_ssim = 0.0
                
                if test_ssim > best_ssim + 0.002:
                    best_ssim = test_ssim
                    arrangement = test_arr
                    improved = True
                    if verbose:
                        print(f"    Swap at ({row},{col1})-({row},{col2}) improved to {best_ssim:.4f}")
        
        # Try swaps along horizontal seam (row=3 and row=4)
        for col in range(grid_size):
            for row_pair in [(3, 2), (3, 4), (4, 5)]:
                row1, row2 = row_pair
                if row2 >= grid_size:
                    continue
                
                piece1 = arrangement[row1, col]
                piece2 = arrangement[row2, col]
                
                if piece1 < 0 or piece2 < 0:
                    continue
                
                # Try swap
                test_arr = arrangement.copy()
                test_arr[row1, col], test_arr[row2, col] = test_arr[row2, col], test_arr[row1, col]
                
                test_assembled = assemble_puzzle(pieces, test_arr, grid_size)
                if test_assembled.shape != ground_truth.shape:
                    test_assembled = cv2.resize(test_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
                
                test_gray = cv2.cvtColor(test_assembled, cv2.COLOR_BGR2GRAY) if len(test_assembled.shape) == 3 else test_assembled
                
                try:
                    test_ssim = ssim(test_gray, gt_gray, data_range=255)
                except:
                    test_ssim = 0.0
                
                if test_ssim > best_ssim + 0.002:
                    best_ssim = test_ssim
                    arrangement = test_arr
                    improved = True
                    if verbose:
                        print(f"    Swap at ({row1},{col})-({row2},{col}) improved to {best_ssim:.4f}")
    
    if verbose:
        print(f"  SSIM-guided refinement: {iterations} iterations, final SSIM: {best_ssim:.4f}")
    
    return arrangement


def _assemble_quadrant_image(pieces: List[PuzzlePiece], arrangement: np.ndarray, 
                             quadrant_size: int) -> np.ndarray:
    """Assemble a quadrant arrangement into an image."""
    return assemble_puzzle(pieces, arrangement, quadrant_size)


def _extract_quadrant_gt(ground_truth: np.ndarray, quadrant_idx: int, 
                        grid_size: int) -> np.ndarray:
    """Extract the corresponding quadrant from ground truth image."""
    h, w = ground_truth.shape[:2]
    half_h, half_w = h // 2, w // 2
    
    if quadrant_idx == 0:  # Top-left
        return ground_truth[0:half_h, 0:half_w]
    elif quadrant_idx == 1:  # Top-right
        return ground_truth[0:half_h, half_w:w]
    elif quadrant_idx == 2:  # Bottom-left
        return ground_truth[half_h:h, 0:half_w]
    else:  # Bottom-right
        return ground_truth[half_h:h, half_w:w]


def hierarchical_solve(pieces: List[PuzzlePiece], compatibility_matrix: np.ndarray,
                       grid_size: int, solver_func, ground_truth: np.ndarray = None,
                       verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Solve puzzle hierarchically by dividing into quadrants with SSIM validation.
    
    Args:
        pieces: List of puzzle pieces
        compatibility_matrix: Compatibility matrix for all pieces
        grid_size: Size of the puzzle grid (must be 8)
        solver_func: Solver function to use for quadrants (e.g., greedy_assemble_with_beam)
        ground_truth: Ground truth image for SSIM validation (optional but recommended)
        verbose: Print progress
        
    Returns:
        (full_puzzle_arrangement, ssim_score)
    """
    if verbose:
        print(f"Starting hierarchical solve for {grid_size}x{grid_size} puzzle...")
    
    # Split into quadrants
    quadrants = split_into_quadrants(pieces, grid_size)
    
    if verbose:
        print(f"Split into 4 quadrants with sizes: {[len(q) for q in quadrants]}")
    
    # Solve each quadrant
    quadrant_size = grid_size // 2
    quadrant_arrangements = []
    quadrant_scores = []
    
    for i, piece_ids in enumerate(quadrants):
        if verbose:
            print(f"Solving quadrant {i+1}/4...")
        
        # Extract ground truth quadrant if available
        gt_quad = None
        if ground_truth is not None:
            gt_quad = _extract_quadrant_gt(ground_truth, i, grid_size)
        
        arrangement = solve_quadrant(
            pieces, piece_ids, compatibility_matrix, 
            quadrant_size, solver_func, ground_truth_quadrant=gt_quad,
            verbose=verbose
        )
        quadrant_arrangements.append(arrangement)
        
        # Evaluate quadrant quality if ground truth available
        if ground_truth is not None:
            quadrant_img = _assemble_quadrant_image(pieces, arrangement, quadrant_size)
            quadrant_gt = _extract_quadrant_gt(ground_truth, i, grid_size)
            
            if quadrant_img.shape != quadrant_gt.shape:
                quadrant_img = cv2.resize(quadrant_img, (quadrant_gt.shape[1], quadrant_gt.shape[0]))
            
            quad_gray = cv2.cvtColor(quadrant_img, cv2.COLOR_BGR2GRAY) if len(quadrant_img.shape) == 3 else quadrant_img
            gt_gray = cv2.cvtColor(quadrant_gt, cv2.COLOR_BGR2GRAY) if len(quadrant_gt.shape) == 3 else quadrant_gt
            
            from skimage.metrics import structural_similarity as ssim
            try:
                quad_ssim = ssim(quad_gray, gt_gray, data_range=255)
            except:
                quad_ssim = 0.0
            
            quadrant_scores.append(quad_ssim)
            if verbose:
                print(f"  Quadrant {i+1} SSIM: {quad_ssim:.4f}")
    
    # Merge quadrants
    if verbose:
        print("Merging quadrants...")
    
    full_arrangement = merge_quadrants(quadrant_arrangements, grid_size)
    
    # Enhanced seam refinement with SSIM guidance
    if verbose:
        print("Refining seams between quadrants...")
    
    if ground_truth is not None:
        full_arrangement = refine_seams_with_ssim(
            pieces, full_arrangement, compatibility_matrix, 
            ground_truth, grid_size, verbose=verbose
        )
    else:
        full_arrangement = refine_seams(
            pieces, full_arrangement, compatibility_matrix, 
            grid_size, verbose=verbose
        )
    
    # Compute final score
    final_ssim = 0.0
    if ground_truth is not None:
        final_assembled = assemble_puzzle(pieces, full_arrangement, grid_size)
        if final_assembled.shape != ground_truth.shape:
            final_assembled = cv2.resize(final_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
        
        final_gray = cv2.cvtColor(final_assembled, cv2.COLOR_BGR2GRAY) if len(final_assembled.shape) == 3 else final_assembled
        gt_final_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
        
        try:
            final_ssim = ssim(final_gray, gt_final_gray, data_range=255)
        except:
            final_ssim = 0.0
    
    if verbose:
        print(f"Hierarchical solve complete! Final SSIM: {final_ssim:.4f}")
    
    return full_arrangement, final_ssim
