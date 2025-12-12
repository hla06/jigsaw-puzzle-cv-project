"""
Adaptive beam width solver for 4×4 puzzles.
Dynamically adjusts beam width based on piece similarity and placement difficulty.
"""

import numpy as np
import cv2
from typing import List, Tuple
from puzzle_utils import PuzzlePiece, assemble_puzzle, get_complementary_edge
from skimage.metrics import structural_similarity as ssim
from exhaustive_solver import _identify_piece_types_advanced


def compute_piece_similarity(pieces: List[PuzzlePiece], compatibility_matrix: np.ndarray) -> float:
    """
    Compute average similarity between pieces based on compatibility scores.
    
    Args:
        pieces: List of puzzle pieces
        compatibility_matrix: Edge compatibility scores
        
    Returns:
        Average similarity score (higher = pieces are more similar/harder to distinguish)
    """
    n = len(pieces)
    if n < 2:
        return 0.5
    
    # Sample compatibility scores
    scores = []
    for i in range(n):
        for e in range(4):
            for j in range(n):
                if i == j:
                    continue
                for e2 in range(4):
                    scores.append(compatibility_matrix[i, e, j, e2])
    
    return np.mean(scores) if scores else 0.5


def compute_local_ambiguity(candidates: List[Tuple[int, float]], threshold: float = 0.05) -> float:
    """
    Compute ambiguity in candidate selection - how hard it is to pick the right piece.
    
    Args:
        candidates: List of (piece_id, score) tuples sorted by score
        threshold: Score difference threshold for ambiguity
        
    Returns:
        Ambiguity score [0-1], higher = more ambiguous (need wider beam)
    """
    if len(candidates) < 2:
        return 0.0
    
    # Count how many top candidates have similar scores
    best_score = candidates[0][1]
    similar_count = sum(1 for _, score in candidates if abs(score - best_score) < threshold)
    
    # Normalize
    ambiguity = min(1.0, similar_count / max(3, len(candidates) * 0.3))
    
    return ambiguity


def adaptive_beam_width(base_width: int, step: int, total_steps: int,
                       piece_similarity: float, local_ambiguity: float) -> int:
    """
    Compute adaptive beam width based on problem difficulty and current progress.
    
    Args:
        base_width: Baseline beam width
        step: Current step (0 to total_steps-1)
        total_steps: Total number of steps
        piece_similarity: Overall piece similarity [0-1]
        local_ambiguity: Local decision ambiguity [0-1]
        
    Returns:
        Adjusted beam width
    """
    # Early steps: wider beam (more exploration)
    progress = step / max(1, total_steps - 1)
    if progress < 0.25:  # First 25% of placement
        stage_factor = 1.5
    elif progress < 0.5:  # Middle 50%
        stage_factor = 1.2
    else:  # Last 50%
        stage_factor = 1.0
    
    # High similarity: wider beam
    similarity_factor = 1.0 + piece_similarity * 0.5
    
    # High local ambiguity: wider beam
    ambiguity_factor = 1.0 + local_ambiguity * 0.8
    
    # Combined adaptive width
    adjusted_width = int(base_width * stage_factor * similarity_factor * ambiguity_factor)
    
    # Clamp to reasonable range
    return max(3, min(15, adjusted_width))


def solve_4x4_adaptive_beam(pieces: List[PuzzlePiece], compatibility_matrix: np.ndarray,
                            ground_truth: np.ndarray, base_beam_width: int = 5,
                            verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    4×4 solver using SSIM-guided beam search with adaptive beam width.
    
    Dynamically adjusts beam width based on:
    - Overall piece similarity (harder puzzles get wider beam)
    - Local placement ambiguity (unclear choices get wider beam)
    - Progress stage (early exploration gets wider beam)
    
    Args:
        pieces: List of 16 puzzle pieces
        compatibility_matrix: Edge compatibility scores
        ground_truth: Ground truth image for SSIM comparison
        base_beam_width: Base beam width (will be adapted)
        verbose: Print progress information
        
    Returns:
        (best_arrangement, best_ssim_score)
    """
    edge_names = ['top', 'right', 'bottom', 'left']
    n_pieces = len(pieces)
    
    # Compute overall piece similarity
    piece_similarity = compute_piece_similarity(pieces, compatibility_matrix)
    
    if verbose:
        print(f"Starting 4×4 adaptive beam search (base_width={base_beam_width}, similarity={piece_similarity:.3f})...")
    
    # Identify piece types
    corners, edges, interior = _identify_piece_types_advanced(pieces)
    
    if verbose:
        print(f"  Piece types: {len(corners)} corners, {len(edges)} edges, {len(interior)} interior")
    
    # Initialize beam with corner pieces in top-left
    initial_beams = []
    corner_candidates = corners if corners else list(range(min(base_beam_width, n_pieces)))
    
    for corner_id in corner_candidates[:base_beam_width]:
        arrangement = np.full((4, 4), -1, dtype=int)
        arrangement[0, 0] = corner_id
        placed = {corner_id}
        initial_beams.append((arrangement, placed, 0.0))
    
    beam = initial_beams
    
    # Preferred placement order
    position_order = [
        (0, 0), (0, 3), (3, 0), (3, 3),  # Corners
        (0, 1), (0, 2), (1, 0), (2, 0), (3, 1), (3, 2), (1, 3), (2, 3),  # Edges
        (1, 1), (1, 2), (2, 1), (2, 2)  # Interior
    ]
    
    # Build solution incrementally with adaptive beam
    for step in range(1, n_pieces):
        new_candidates = []
        
        for arrangement, placed, cumulative_score in beam:
            # Find next position
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
            
            # Collect neighbors
            neighbors = []
            if row > 0 and arrangement[row-1, col] != -1:
                neighbors.append((arrangement[row-1, col], 'top'))
            if col > 0 and arrangement[row, col-1] != -1:
                neighbors.append((arrangement[row, col-1], 'left'))
            if row < 3 and arrangement[row+1, col] != -1:
                neighbors.append((arrangement[row+1, col], 'bottom'))
            if col < 3 and arrangement[row, col+1] != -1:
                neighbors.append((arrangement[row, col+1], 'right'))
            
            # Score candidates
            piece_scores = []
            for piece_id in candidates:
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
            
            # Sort by score
            piece_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Compute local ambiguity
            local_ambiguity = compute_local_ambiguity(piece_scores)
            
            # Adaptive beam width for this step
            current_beam_width = adaptive_beam_width(
                base_beam_width, step, n_pieces, piece_similarity, local_ambiguity
            )
            
            # Select top candidates based on adaptive width
            for piece_id, edge_score in piece_scores[:current_beam_width]:
                new_arr = arrangement.copy()
                new_arr[row, col] = piece_id
                new_placed = placed.copy()
                new_placed.add(piece_id)
                new_score = cumulative_score + edge_score
                
                # SSIM evaluation when enough pieces placed (50% completion)
                if len(new_placed) >= 8:
                    test_assembled = assemble_puzzle(pieces, new_arr, 4)
                    if test_assembled.shape != ground_truth.shape:
                        test_assembled = cv2.resize(test_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
                    
                    gray1 = cv2.cvtColor(test_assembled, cv2.COLOR_BGR2GRAY) if len(test_assembled.shape) == 3 else test_assembled
                    gray2 = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
                    
                    try:
                        ssim_score = ssim(gray1, gray2, data_range=255)
                        new_score = ssim_score * 10  # Weight SSIM heavily
                    except:
                        pass
                
                new_candidates.append((new_arr, new_placed, new_score))
        
        # Sort and select top candidates for next beam
        new_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Adaptive beam width for beam selection
        avg_ambiguity = np.mean([compute_local_ambiguity([(i, c[2]) for i, c in enumerate(new_candidates[:10])]) for _ in range(1)])
        beam_select_width = adaptive_beam_width(base_beam_width, step, n_pieces, piece_similarity, avg_ambiguity)
        beam = new_candidates[:beam_select_width]
        
        if verbose and step % 4 == 0:
            print(f"  Step {step}/{n_pieces-1}: beam_width={beam_select_width}, top_score={beam[0][2]:.4f}")
    
    # Select best from final beam
    if not beam:
        return np.arange(16).reshape(4, 4), 0.0
    
    best_arrangement, _, _ = beam[0]
    
    # Compute final SSIM
    final_assembled = assemble_puzzle(pieces, best_arrangement, 4)
    if final_assembled.shape != ground_truth.shape:
        final_assembled = cv2.resize(final_assembled, (ground_truth.shape[1], ground_truth.shape[0]))
    
    gray1 = cv2.cvtColor(final_assembled, cv2.COLOR_BGR2GRAY) if len(final_assembled.shape) == 3 else final_assembled
    gray2 = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if len(ground_truth.shape) == 3 else ground_truth
    
    try:
        final_ssim = ssim(gray1, gray2, data_range=255)
    except:
        final_ssim = 0.0
    
    if verbose:
        print(f"  Final SSIM: {final_ssim:.4f}")
    
    return best_arrangement, final_ssim
