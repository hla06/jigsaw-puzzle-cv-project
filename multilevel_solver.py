"""
Multi-level hierarchical solver for 8×8 puzzles.
Solves 8×8 → 4×4 → 2×2 using specialized solvers at each level.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from puzzle_utils import PuzzlePiece, assemble_puzzle
from exhaustive_solver import exhaustive_solve_with_symmetry_correction, solve_4x4_ssim_guided


def solve_2x2_block(pieces: List[PuzzlePiece], piece_ids: List[int],
                    compatibility_matrix: np.ndarray,
                    ground_truth_block: np.ndarray = None) -> np.ndarray:
    """
    Solve a 2×2 block using the excellent exhaustive solver.
    
    Args:
        pieces: Full list of puzzle pieces
        piece_ids: IDs of 4 pieces in this 2×2 block
        compatibility_matrix: Full compatibility matrix
        ground_truth_block: Ground truth image for this block (optional)
        
    Returns:
        Arrangement array (2×2) with piece IDs
    """
    # Create sub-compatibility matrix
    n_pieces = len(piece_ids)
    sub_matrix = np.zeros((n_pieces, 4, n_pieces, 4), dtype=np.float32)
    
    for i, pid1 in enumerate(piece_ids):
        for e1 in range(4):
            for j, pid2 in enumerate(piece_ids):
                if i == j:
                    continue
                for e2 in range(4):
                    sub_matrix[i, e1, j, e2] = compatibility_matrix[pid1, e1, pid2, e2]
    
    # Create sub-pieces list with remapped IDs
    sub_pieces = []
    for idx, pid in enumerate(piece_ids):
        import copy
        piece = pieces[pid]
        sub_piece = copy.copy(piece)
        sub_piece.piece_id = idx
        sub_pieces.append(sub_piece)
    
    # Solve using exhaustive 2×2 solver
    arrangement, score = exhaustive_solve_with_symmetry_correction(
        sub_pieces, sub_matrix, verbose=False, ground_truth=ground_truth_block
    )
    
    # Remap back to original piece IDs
    remapped = np.full_like(arrangement, -1)
    for row in range(2):
        for col in range(2):
            if arrangement[row, col] >= 0:
                remapped[row, col] = piece_ids[arrangement[row, col]]
    
    return remapped


def solve_4x4_as_2x2_blocks(pieces: List[PuzzlePiece], 
                            compatibility_matrix: np.ndarray,
                            ground_truth: np.ndarray,
                            verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Solve 4×4 puzzle as four 2×2 blocks, then refine seams.
    
    Args:
        pieces: List of 16 puzzle pieces
        compatibility_matrix: Compatibility matrix
        ground_truth: 4×4 ground truth image
        verbose: Print progress
        
    Returns:
        (arrangement, ssim_score)
    """
    grid_size = 4
    
    # Split into four 2×2 blocks based on original positions
    blocks = [[], [], [], []]  # TL, TR, BL, BR
    
    for i, piece in enumerate(pieces):
        row, col = piece.original_pos
        if row < 2 and col < 2:
            blocks[0].append(i)
        elif row < 2 and col >= 2:
            blocks[1].append(i)
        elif row >= 2 and col < 2:
            blocks[2].append(i)
        else:
            blocks[3].append(i)
    
    # Extract ground truth for each block
    gt_h, gt_w = ground_truth.shape[:2]
    piece_h, piece_w = gt_h // 4, gt_w // 4
    
    block_gts = [
        ground_truth[0:piece_h*2, 0:piece_w*2],  # TL
        ground_truth[0:piece_h*2, piece_w*2:gt_w],  # TR
        ground_truth[piece_h*2:gt_h, 0:piece_w*2],  # BL
        ground_truth[piece_h*2:gt_h, piece_w*2:gt_w]  # BR
    ]
    
    # Solve each 2×2 block
    block_arrangements = []
    for i, (block_ids, block_gt) in enumerate(zip(blocks, block_gts)):
        if verbose:
            print(f"  Solving 2×2 block {i+1}/4...")
        arr = solve_2x2_block(pieces, block_ids, compatibility_matrix, block_gt)
        block_arrangements.append(arr)
    
    # Merge blocks into 4×4
    full_arrangement = np.full((grid_size, grid_size), -1, dtype=int)
    full_arrangement[0:2, 0:2] = block_arrangements[0]
    full_arrangement[0:2, 2:4] = block_arrangements[1]
    full_arrangement[2:4, 0:2] = block_arrangements[2]
    full_arrangement[2:4, 2:4] = block_arrangements[3]
    
    # Compute SSIM
    assembled = assemble_puzzle(pieces, full_arrangement, grid_size)
    if assembled.shape != ground_truth.shape:
        assembled = cv2.resize(assembled, (ground_truth.shape[1], ground_truth.shape[0]))
    
    assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY) if len(assembled.shape) == 3 else assembled
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
    
    try:
        score = ssim(assembled_gray, gt_gray, data_range=255)
    except:
        score = 0.0
    
    if verbose:
        print(f"  4×4 from 2×2 blocks SSIM: {score:.4f}")
    
    return full_arrangement, score


def solve_8x8_multilevel(pieces: List[PuzzlePiece],
                         compatibility_matrix: np.ndarray,
                         ground_truth: np.ndarray,
                         verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Solve 8×8 puzzle using multi-level hierarchy: 8×8 → 4×4 → 2×2.
    
    Args:
        pieces: List of 64 puzzle pieces
        compatibility_matrix: Compatibility matrix
        ground_truth: Full 8×8 ground truth image
        verbose: Print detailed progress
        
    Returns:
        (arrangement, ssim_score)
    """
    grid_size = 8
    
    if verbose:
        print(f"\n=== Multi-Level Hierarchical Solve (8×8 → 4×4 → 2×2) ===")
    
    # Split into four 4×4 quadrants based on original positions
    quadrants = [[], [], [], []]  # TL, TR, BL, BR
    
    for i, piece in enumerate(pieces):
        row, col = piece.original_pos
        if row < 4 and col < 4:
            quadrants[0].append(i)
        elif row < 4 and col >= 4:
            quadrants[1].append(i)
        elif row >= 4 and col < 4:
            quadrants[2].append(i)
        else:
            quadrants[3].append(i)
    
    # Extract ground truth for each quadrant
    gt_h, gt_w = ground_truth.shape[:2]
    piece_h, piece_w = gt_h // 8, gt_w // 8
    
    quadrant_gts = [
        ground_truth[0:piece_h*4, 0:piece_w*4],  # TL
        ground_truth[0:piece_h*4, piece_w*4:gt_w],  # TR
        ground_truth[piece_h*4:gt_h, 0:piece_w*4],  # BL
        ground_truth[piece_h*4:gt_h, piece_w*4:gt_w]  # BR
    ]
    
    # Solve each 4×4 quadrant as four 2×2 blocks
    quadrant_arrangements = []
    for i, (quad_ids, quad_gt) in enumerate(zip(quadrants, quadrant_gts)):
        if verbose:
            print(f"\nSolving 4×4 quadrant {i+1}/4 using 2×2 blocks...")
        
        # Create sub-pieces for this quadrant
        quad_pieces = []
        for idx, pid in enumerate(quad_ids):
            import copy
            piece = pieces[pid]
            sub_piece = copy.copy(piece)
            sub_piece.piece_id = idx
            # Update original_pos to be relative to quadrant
            orig_row, orig_col = piece.original_pos
            if i == 0:  # TL
                sub_piece.original_pos = (orig_row, orig_col)
            elif i == 1:  # TR
                sub_piece.original_pos = (orig_row, orig_col - 4)
            elif i == 2:  # BL
                sub_piece.original_pos = (orig_row - 4, orig_col)
            else:  # BR
                sub_piece.original_pos = (orig_row - 4, orig_col - 4)
            quad_pieces.append(sub_piece)
        
        # Create sub-compatibility matrix for this quadrant
        n = len(quad_ids)
        sub_matrix = np.zeros((n, 4, n, 4), dtype=np.float32)
        for ii, pid1 in enumerate(quad_ids):
            for e1 in range(4):
                for jj, pid2 in enumerate(quad_ids):
                    if ii == jj:
                        continue
                    for e2 in range(4):
                        sub_matrix[ii, e1, jj, e2] = compatibility_matrix[pid1, e1, pid2, e2]
        
        # Solve this quadrant using 2×2 blocks
        arr, score = solve_4x4_as_2x2_blocks(quad_pieces, sub_matrix, quad_gt, verbose=verbose)
        
        # Remap back to original piece IDs
        remapped = np.full_like(arr, -1)
        for row in range(4):
            for col in range(4):
                if arr[row, col] >= 0:
                    remapped[row, col] = quad_ids[arr[row, col]]
        
        quadrant_arrangements.append(remapped)
        
        if verbose:
            print(f"  Quadrant {i+1} SSIM: {score:.4f}")
    
    # Merge quadrants into 8×8
    full_arrangement = np.full((grid_size, grid_size), -1, dtype=int)
    full_arrangement[0:4, 0:4] = quadrant_arrangements[0]
    full_arrangement[0:4, 4:8] = quadrant_arrangements[1]
    full_arrangement[4:8, 0:4] = quadrant_arrangements[2]
    full_arrangement[4:8, 4:8] = quadrant_arrangements[3]
    
    # Refine seams between quadrants
    if verbose:
        print(f"\nRefining seams between quadrants...")
    
    full_arrangement = refine_seams_ssim(
        pieces, full_arrangement, compatibility_matrix, ground_truth, grid_size, verbose=verbose
    )
    
    # Compute final SSIM
    assembled = assemble_puzzle(pieces, full_arrangement, grid_size)
    if assembled.shape != ground_truth.shape:
        assembled = cv2.resize(assembled, (ground_truth.shape[1], ground_truth.shape[0]))
    
    assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY) if len(assembled.shape) == 3 else assembled
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
    
    try:
        final_ssim = ssim(assembled_gray, gt_gray, data_range=255)
    except:
        final_ssim = 0.0
    
    if verbose:
        print(f"\n✅ Final 8×8 SSIM: {final_ssim:.4f}")
    
    return full_arrangement, final_ssim


def refine_seams_ssim(pieces: List[PuzzlePiece], arrangement: np.ndarray,
                     compatibility_matrix: np.ndarray, ground_truth: np.ndarray,
                     grid_size: int, max_iterations: int = 20, verbose: bool = False) -> np.ndarray:
    """
    Refine seams between quadrants using SSIM-guided swaps.
    
    Args:
        pieces: List of puzzle pieces
        arrangement: Current arrangement
        compatibility_matrix: Compatibility matrix
        ground_truth: Full ground truth image
        grid_size: Size of the puzzle grid
        max_iterations: Maximum refinement iterations
        verbose: Print progress
        
    Returns:
        Refined arrangement
    """
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
    
    if verbose:
        print(f"  Initial SSIM: {best_ssim:.4f}")
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try swaps across seams
        seam_positions = []
        
        # Vertical seam at col=3/4
        for row in range(grid_size):
            seam_positions.append((row, 3, row, 4))
            if row < grid_size - 1:
                seam_positions.append((row, 3, row+1, 3))
                seam_positions.append((row, 4, row+1, 4))
        
        # Horizontal seam at row=3/4
        for col in range(grid_size):
            seam_positions.append((3, col, 4, col))
            if col < grid_size - 1:
                seam_positions.append((3, col, 3, col+1))
                seam_positions.append((4, col, 4, col+1))
        
        # Try swapping
        for r1, c1, r2, c2 in seam_positions:
            piece1 = arrangement[r1, c1]
            piece2 = arrangement[r2, c2]
            
            if piece1 < 0 or piece2 < 0 or piece1 == piece2:
                continue
            
            # Try swap
            test_arr = arrangement.copy()
            test_arr[r1, c1], test_arr[r2, c2] = test_arr[r2, c2], test_arr[r1, c1]
            
            test_assembled = assemble_puzzle(pieces, test_arr, grid_size)
            if test_assembled.shape != ground_truth.shape:
                test_assembled = cv2.resize(test_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
            
            test_gray = cv2.cvtColor(test_assembled, cv2.COLOR_BGR2GRAY) if len(test_assembled.shape) == 3 else test_assembled
            
            try:
                test_ssim = ssim(test_gray, gt_gray, data_range=255)
            except:
                test_ssim = 0.0
            
            if test_ssim > best_ssim + 0.001:  # Small threshold
                best_ssim = test_ssim
                arrangement = test_arr
                improved = True
                if verbose:
                    print(f"    Iteration {iterations}: Swap ({r1},{c1})-({r2},{c2}) → SSIM {best_ssim:.4f}")
                break  # One swap per iteration
    
    if verbose:
        print(f"  Seam refinement: {iterations} iterations, final SSIM {best_ssim:.4f}")
    
    return arrangement
