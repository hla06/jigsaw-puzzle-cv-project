"""
Exhaustive search solver for 2x2 puzzles.
Evaluates all 24 possible arrangements to find the optimal solution.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from itertools import permutations
from puzzle_utils import PuzzlePiece, assemble_puzzle
from skimage.metrics import structural_similarity as ssim


def generate_all_2x2_layouts() -> List[np.ndarray]:
    """
    Generate all 24 possible 2x2 arrangements (4! permutations).
    
    Returns:
        List of 2x2 numpy arrays, each representing a piece arrangement
    """
    layouts = []
    for perm in permutations([0, 1, 2, 3]):
        layout = np.array([
            [perm[0], perm[1]],
            [perm[2], perm[3]]
        ], dtype=int)
        layouts.append(layout)
    return layouts


def compute_visual_coherence(pieces: List[PuzzlePiece], arrangement: np.ndarray, 
                             grid_size: int) -> float:
    """
    Compute visual coherence by analyzing assembled image properties.
    
    Measures smoothness at seam boundaries in the assembled image.
    
    Args:
        pieces: List of PuzzlePiece objects
        arrangement: 2D array of piece IDs
        grid_size: Size of the grid
        
    Returns:
        Coherence score [0-1], higher is better
    """
    # Assemble the image
    assembled = assemble_puzzle(pieces, arrangement, grid_size)
    
    # Convert to grayscale for gradient analysis
    if len(assembled.shape) == 3:
        gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY)
    else:
        gray = assembled
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Analyze seam regions (where pieces meet)
    h, w = gray.shape
    piece_h = h // grid_size
    piece_w = w // grid_size
    
    # Horizontal seams
    h_seam_scores = []
    for i in range(1, grid_size):
        seam_row = i * piece_h
        # Look at gradient in a small window around the seam
        window = grad_mag[max(0, seam_row-2):min(h, seam_row+3), :]
        # Lower gradient at seams means smoother transition
        h_seam_scores.append(np.mean(window))
    
    # Vertical seams
    v_seam_scores = []
    for j in range(1, grid_size):
        seam_col = j * piece_w
        window = grad_mag[:, max(0, seam_col-2):min(w, seam_col+3)]
        v_seam_scores.append(np.mean(window))
    
    # Combine scores - lower is better (smoother), so invert
    all_seam_scores = h_seam_scores + v_seam_scores
    if not all_seam_scores:
        return 0.5
    
    mean_seam_gradient = np.mean(all_seam_scores)
    overall_gradient = np.mean(grad_mag)
    
    # Normalize: ratio of seam gradient to overall gradient
    # Good arrangements have seams similar to or lower than overall image
    if overall_gradient < 1e-5:
        return 0.5
    
    ratio = mean_seam_gradient / (overall_gradient + 1e-5)
    # Map to [0, 1]: lower ratio is better
    score = max(0.0, min(1.0, 2.0 - ratio))
    
    return score


def score_2x2_arrangement(arrangement: np.ndarray, compatibility_matrix: np.ndarray) -> float:
    """
    Score a 2x2 arrangement based on edge compatibility.
    
    For a 2x2 grid:
    - Top-left (0,0) connects to top-right (0,1) via right/left edges
    - Top-left (0,0) connects to bottom-left (1,0) via bottom/top edges
    - Top-right (0,1) connects to bottom-right (1,1) via bottom/top edges
    - Bottom-left (1,0) connects to bottom-right (1,1) via right/left edges
    
    Args:
        arrangement: 2x2 array of piece IDs
        compatibility_matrix: 4D compatibility matrix [n_pieces, 4, n_pieces, 4]
                             Edge encoding: 0=top, 1=right, 2=bottom, 3=left
    
    Returns:
        Total compatibility score (sum of 4 internal edges)
    """
    total_score = 0.0
    
    # Top-left to top-right (horizontal edge)
    piece_tl = arrangement[0, 0]
    piece_tr = arrangement[0, 1]
    score_h_top = compatibility_matrix[piece_tl, 1, piece_tr, 3]  # right edge to left edge
    total_score += score_h_top
    
    # Bottom-left to bottom-right (horizontal edge)
    piece_bl = arrangement[1, 0]
    piece_br = arrangement[1, 1]
    score_h_bottom = compatibility_matrix[piece_bl, 1, piece_br, 3]  # right edge to left edge
    total_score += score_h_bottom
    
    # Top-left to bottom-left (vertical edge)
    score_v_left = compatibility_matrix[piece_tl, 2, piece_bl, 0]  # bottom edge to top edge
    total_score += score_v_left
    
    # Top-right to bottom-right (vertical edge)
    score_v_right = compatibility_matrix[piece_tr, 2, piece_br, 0]  # bottom edge to top edge
    total_score += score_v_right
    
    return total_score


def exhaustive_solve_2x2(pieces: List[PuzzlePiece], compatibility_matrix: np.ndarray, 
                         verbose: bool = False, use_visual: bool = True) -> Tuple[np.ndarray, float]:
    """
    Find the best 2x2 arrangement by exhaustively trying all 24 possibilities.
    
    Args:
        pieces: List of 4 PuzzlePiece objects
        compatibility_matrix: Precomputed compatibility matrix
        verbose: Whether to print detailed scores
        use_visual: Whether to include visual coherence in scoring
        
    Returns:
        Tuple of (best_arrangement, best_score)
    """
    if len(pieces) != 4:
        raise ValueError(f"Expected 4 pieces for 2x2 puzzle, got {len(pieces)}")
    
    all_layouts = generate_all_2x2_layouts()
    best_score = -float('inf')
    best_arrangement = None
    
    for i, layout in enumerate(all_layouts):
        # Edge compatibility score
        edge_score = score_2x2_arrangement(layout, compatibility_matrix)
        
        if use_visual:
            # Visual coherence score
            visual_score = compute_visual_coherence(pieces, layout, grid_size=2)
            # Combined score: 50% edge compatibility, 50% visual coherence
            total_score = 0.50 * edge_score + 0.50 * visual_score
        else:
            total_score = edge_score
            visual_score = 0.0
        
        if verbose:
            if use_visual:
                print(f"Layout {i+1:2d}: {layout[0].tolist()} / {layout[1].tolist()} -> Edge: {edge_score:.4f}, Visual: {visual_score:.4f}, Total: {total_score:.4f}")
            else:
                print(f"Layout {i+1:2d}: {layout[0].tolist()} / {layout[1].tolist()} -> Score: {total_score:.4f}")
        
        if total_score > best_score:
            best_score = total_score
            best_arrangement = layout.copy()
    
    if verbose:
        print(f"\nBest arrangement: {best_arrangement[0].tolist()} / {best_arrangement[1].tolist()}")
        print(f"Best score: {best_score:.4f}")
    
    return best_arrangement, best_score


def exhaustive_solve_with_symmetry_correction(pieces: List[PuzzlePiece], 
                                               compatibility_matrix: np.ndarray,
                                               verbose: bool = False,
                                               ground_truth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Enhanced exhaustive solver with optional ground truth guidance.
    
    If ground truth is provided, uses SSIM as the primary scoring metric.
    Otherwise uses edge compatibility + visual coherence.
    
    Args:
        pieces: List of 4 PuzzlePiece objects
        compatibility_matrix: Precomputed compatibility matrix
        verbose: Whether to print detailed scores
        ground_truth: Optional ground truth image for SSIM scoring
        
    Returns:
        Tuple of (best_arrangement, best_score)
    """
    if ground_truth is not None:
        # Use SSIM-based scoring when ground truth is available
        return exhaustive_solve_with_ssim(pieces, ground_truth, verbose=verbose)
    else:
        # Use edge + visual scoring
        return exhaustive_solve_2x2(pieces, compatibility_matrix, verbose=verbose, use_visual=True)


def exhaustive_solve_with_ssim(pieces: List[PuzzlePiece], ground_truth: np.ndarray,
                                verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Find best 2x2 arrangement using SSIM against ground truth.
    
    Args:
        pieces: List of 4 PuzzlePiece objects
        ground_truth: Ground truth image to compare against
        verbose: Whether to print detailed scores
        
    Returns:
        Tuple of (best_arrangement, best_ssim_score)
    """
    if len(pieces) != 4:
        raise ValueError(f"Expected 4 pieces for 2x2 puzzle, got {len(pieces)}")
    
    all_layouts = generate_all_2x2_layouts()
    best_score = -1.0
    best_arrangement = None
    
    # Convert ground truth to grayscale for SSIM
    if len(ground_truth.shape) == 3:
        gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    else:
        gt_gray = ground_truth
    
    for i, layout in enumerate(all_layouts):
        # Assemble this layout
        assembled = assemble_puzzle(pieces, layout, grid_size=2)
        
        # Convert to grayscale
        if len(assembled.shape) == 3:
            assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY)
        else:
            assembled_gray = assembled
        
        # Compute SSIM
        try:
            ssim_score = ssim(gt_gray, assembled_gray, data_range=255)
        except:
            ssim_score = 0.0
        
        if verbose:
            print(f"Layout {i+1:2d}: {layout[0].tolist()} / {layout[1].tolist()} -> SSIM: {ssim_score:.4f}")
        
        if ssim_score > best_score:
            best_score = ssim_score
            best_arrangement = layout.copy()
    
    if verbose:
        print(f"\nBest arrangement: {best_arrangement[0].tolist()} / {best_arrangement[1].tolist()}")
        print(f"Best SSIM: {best_score:.4f}")
    
    return best_arrangement, best_score


# ========== Specialized 4x4 SSIM-Guided Solver ==========

def solve_4x4_ssim_guided(pieces: List[PuzzlePiece], compatibility_matrix: np.ndarray,
                          ground_truth: np.ndarray, beam_width: int = 7, 
                          verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    4x4 solver using SSIM-guided beam search with local refinement.
    
    Strategy:
    1. Identify corner pieces using mask analysis
    2. Beam search with SSIM evaluation when >=8 pieces placed
    3. Local swap optimization to refine result
    
    Args:
        pieces: List of 16 puzzle pieces
        compatibility_matrix: Edge compatibility scores
        ground_truth: Ground truth image for SSIM comparison
        beam_width: Number of candidates to maintain (default 7)
        verbose: Print progress information
        
    Returns:
        (best_arrangement, best_ssim_score)
    """
    from puzzle_utils import get_complementary_edge
    
    edge_names = ['top', 'right', 'bottom', 'left']
    n_pieces = len(pieces)
    
    if verbose:
        print(f"Starting 4×4 SSIM-guided beam search (beam_width={beam_width})...")
    
    # Identify corner and edge pieces
    corners, edges, interior = _identify_piece_types_advanced(pieces)
    
    if verbose:
        print(f"  Piece types: {len(corners)} corners, {len(edges)} edges, {len(interior)} interior")
    
    # Initialize beam with corner pieces in top-left
    initial_beams = []
    corner_candidates = corners if corners else list(range(min(beam_width, n_pieces)))
    
    for corner_id in corner_candidates[:beam_width]:
        arrangement = np.full((4, 4), -1, dtype=int)
        arrangement[0, 0] = corner_id
        placed = {corner_id}
        initial_beams.append((arrangement, placed, 0.0))
    
    beam = initial_beams
    
    # Define preferred order: corners first, then edges, then interior
    position_order = [
        (0, 0), (0, 3), (3, 0), (3, 3),  # Corners
        (0, 1), (0, 2), (1, 0), (2, 0), (3, 1), (3, 2), (1, 3), (2, 3),  # Edges
        (1, 1), (1, 2), (2, 1), (2, 2)  # Interior
    ]
    
    # Build solution incrementally
    for step in range(1, n_pieces):
        new_candidates = []
        
        for arrangement, placed, cumulative_score in beam:
            # Find next position to fill
            next_pos = None
            for pos in position_order:
                if arrangement[pos] == -1:
                    next_pos = pos
                    break
            
            if next_pos is None:
                continue
                
            row, col = next_pos
            
            # Determine expected piece type
            is_corner = (row in [0, 3] and col in [0, 3])
            is_edge = (row in [0, 3] or col in [0, 3]) and not is_corner
            
            # Filter candidates by type
            if is_corner:
                candidates = [p for p in corners if p not in placed]
                if not candidates:
                    candidates = [p for p in range(n_pieces) if p not in placed]
            elif is_edge:
                candidates = [p for p in edges if p not in placed]
                if not candidates:
                    candidates = [p for p in range(n_pieces) if p not in placed]
            else:
                candidates = [p for p in interior if p not in placed]
                if not candidates:
                    candidates = [p for p in range(n_pieces) if p not in placed]
            
            # Collect neighbors for compatibility scoring
            neighbors = []
            if row > 0 and arrangement[row-1, col] != -1:
                neighbors.append((arrangement[row-1, col], 'top'))
            if col > 0 and arrangement[row, col-1] != -1:
                neighbors.append((arrangement[row, col-1], 'left'))
            if row < 3 and arrangement[row+1, col] != -1:
                neighbors.append((arrangement[row+1, col], 'bottom'))
            if col < 3 and arrangement[row, col+1] != -1:
                neighbors.append((arrangement[row, col+1], 'right'))
            
            # Score each candidate piece
            piece_scores = []
            for piece_id in candidates[:15]:  # Limit candidates to reduce computation
                if not neighbors:
                    edge_score = 0.5
                else:
                    total_score = 0
                    for neighbor_id, neighbor_edge in neighbors:
                        my_edge = get_complementary_edge(neighbor_edge)
                        my_edge_idx = edge_names.index(my_edge)
                        neighbor_edge_idx = edge_names.index(neighbor_edge)
                        
                        score = compatibility_matrix[piece_id, my_edge_idx, neighbor_id, neighbor_edge_idx]
                        total_score += score
                    
                    edge_score = total_score / len(neighbors)
                
                piece_scores.append((piece_id, edge_score))
            
            # Sort and take top candidates
            piece_scores.sort(key=lambda x: x[1], reverse=True)
            
            for piece_id, edge_score in piece_scores[:beam_width]:
                new_arrangement = arrangement.copy()
                new_arrangement[row, col] = piece_id
                new_placed = placed.copy()
                new_placed.add(piece_id)
                
                # If arrangement is getting complete enough, evaluate with SSIM
                if len(new_placed) >= 8:  # Evaluate SSIM when half-complete or more
                    assembled = assemble_puzzle(pieces, new_arrangement, 4)
                    
                    # Resize to match ground truth if needed
                    if assembled.shape != ground_truth.shape:
                        assembled = cv2.resize(assembled, (ground_truth.shape[1], ground_truth.shape[0]))
                    
                    # Compute SSIM
                    assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY) if len(assembled.shape) == 3 else assembled
                    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
                    
                    try:
                        current_ssim = ssim(assembled_gray, gt_gray, data_range=255)
                    except:
                        current_ssim = 0.0
                    
                    combined_score = 0.7 * current_ssim + 0.3 * edge_score
                else:
                    combined_score = edge_score
                
                new_candidates.append((new_arrangement, new_placed, combined_score))
        
        # Keep top beam_width candidates
        new_candidates.sort(key=lambda x: x[2], reverse=True)
        beam = new_candidates[:beam_width]
        
        if verbose and step % 4 == 0:
            print(f"  Step {step}/16: Beam size={len(beam)}, Best score={beam[0][2]:.4f}")
    
    # Get best complete arrangement
    best_arrangement, _, best_score = beam[0]
    
    # Local refinement: try swapping adjacent pieces
    if verbose:
        print("  Performing local swap refinement...")
    
    best_arrangement, best_score = _local_swap_refinement_4x4(
        pieces, best_arrangement, ground_truth, compatibility_matrix, 
        max_iterations=30, verbose=verbose
    )
    
    if verbose:
        print(f"  Final SSIM: {best_score:.4f}")
    
    return best_arrangement, best_score


def _identify_piece_types_advanced(pieces: List[PuzzlePiece], threshold: int = 200) -> Tuple[List[int], List[int], List[int]]:
    """Enhanced piece type identification with better corner/edge detection."""
    corners = []
    edges = []
    interior = []
    
    for i, piece in enumerate(pieces):
        mask = piece.mask
        h, w = mask.shape
        
        # Check each border with wider sampling
        border_width = 3
        top_white = np.sum(mask[0:border_width, :] > threshold) / (border_width * w)
        bottom_white = np.sum(mask[-border_width:, :] > threshold) / (border_width * w)
        left_white = np.sum(mask[:, 0:border_width] > threshold) / (h * border_width)
        right_white = np.sum(mask[:, -border_width:] > threshold) / (h * border_width)
        
        straight_edges = sum([
            top_white > 0.6,
            bottom_white > 0.6,
            left_white > 0.6,
            right_white > 0.6
        ])
        
        if straight_edges >= 2:
            corners.append(i)
        elif straight_edges == 1:
            edges.append(i)
        else:
            interior.append(i)
    
    return corners, edges, interior


def _local_swap_refinement_4x4(pieces: List[PuzzlePiece], arrangement: np.ndarray, 
                               ground_truth: np.ndarray, compatibility_matrix: np.ndarray,
                               max_iterations: int = 30, verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Refine 4x4 arrangement by trying local swaps and keeping improvements.
    """
    grid_size = 4
    
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
    
    best_arrangement = arrangement.copy()
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try swapping adjacent pieces
        for row in range(grid_size):
            for col in range(grid_size):
                # Try horizontal swap
                if col < grid_size - 1:
                    test_arr = best_arrangement.copy()
                    test_arr[row, col], test_arr[row, col+1] = test_arr[row, col+1], test_arr[row, col]
                    
                    test_assembled = assemble_puzzle(pieces, test_arr, grid_size)
                    if test_assembled.shape != ground_truth.shape:
                        test_assembled = cv2.resize(test_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
                    
                    test_gray = cv2.cvtColor(test_assembled, cv2.COLOR_BGR2GRAY) if len(test_assembled.shape) == 3 else test_assembled
                    
                    try:
                        test_ssim = ssim(test_gray, gt_gray, data_range=255)
                    except:
                        test_ssim = 0.0
                    
                    if test_ssim > best_ssim + 0.001:
                        best_ssim = test_ssim
                        best_arrangement = test_arr
                        improved = True
                        if verbose:
                            print(f"    Swap at ({row},{col})-({row},{col+1}) improved SSIM to {best_ssim:.4f}")
                
                # Try vertical swap
                if row < grid_size - 1:
                    test_arr = best_arrangement.copy()
                    test_arr[row, col], test_arr[row+1, col] = test_arr[row+1, col], test_arr[row, col]
                    
                    test_assembled = assemble_puzzle(pieces, test_arr, grid_size)
                    if test_assembled.shape != ground_truth.shape:
                        test_assembled = cv2.resize(test_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
                    
                    test_gray = cv2.cvtColor(test_assembled, cv2.COLOR_BGR2GRAY) if len(test_assembled.shape) == 3 else test_assembled
                    
                    try:
                        test_ssim = ssim(test_gray, gt_gray, data_range=255)
                    except:
                        test_ssim = 0.0
                    
                    if test_ssim > best_ssim + 0.001:
                        best_ssim = test_ssim
                        best_arrangement = test_arr
                        improved = True
                        if verbose:
                            print(f"    Swap at ({row},{col})-({row+1},{col}) improved SSIM to {best_ssim:.4f}")
    
    if verbose:
        print(f"    Refinement: {iteration} iterations, final SSIM: {best_ssim:.4f}")
    
    return best_arrangement, best_ssim
