"""
Overlap-based hierarchical solver for 8×8 puzzles.
Uses overlapping regions instead of fixed quadrant boundaries.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from puzzle_utils import PuzzlePiece, assemble_puzzle


def extract_overlapping_regions(pieces: List[PuzzlePiece], grid_size: int, 
                                overlap: int = 1) -> List[Dict]:
    """
    Extract overlapping 4×4 regions from 8×8 grid.
    
    Args:
        pieces: List of all puzzle pieces
        grid_size: Size of the puzzle grid (8)
        overlap: Number of pieces to overlap (1 = one piece overlap)
        
    Returns:
        List of regions, each containing:
        - 'piece_ids': List of piece IDs in this region
        - 'positions': List of (row, col) positions within the region
        - 'global_offset': (row_offset, col_offset) in the full grid
    """
    if grid_size != 8:
        raise ValueError("This solver is designed for 8×8 puzzles")
    
    region_size = 4
    step = region_size - overlap
    regions = []
    
    # Create overlapping regions
    for row_start in range(0, grid_size - region_size + 1, step):
        for col_start in range(0, grid_size - region_size + 1, step):
            region = {
                'piece_ids': [],
                'positions': [],
                'global_offset': (row_start, col_start)
            }
            
            # Collect pieces in this region
            for local_row in range(region_size):
                for local_col in range(region_size):
                    global_row = row_start + local_row
                    global_col = col_start + local_col
                    
                    # Find piece at this position
                    for i, piece in enumerate(pieces):
                        if piece.original_pos == (global_row, global_col):
                            region['piece_ids'].append(i)
                            region['positions'].append((local_row, local_col))
                            break
            
            regions.append(region)
    
    return regions


def solve_region_ssim_guided(pieces: List[PuzzlePiece], piece_ids: List[int],
                             compatibility_matrix: np.ndarray, 
                             ground_truth_region: np.ndarray,
                             beam_width: int = 7, verbose: bool = False) -> np.ndarray:
    """
    Solve a 4×4 region using SSIM-guided beam search.
    
    Args:
        pieces: Full list of puzzle pieces
        piece_ids: IDs of pieces in this region
        compatibility_matrix: Full compatibility matrix
        ground_truth_region: Ground truth image for this region
        beam_width: Beam search width
        verbose: Print progress
        
    Returns:
        Arrangement array (4×4) with piece IDs
    """
    from exhaustive_solver import solve_4x4_ssim_guided
    
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
    
    # Solve using SSIM-guided solver
    arrangement, score = solve_4x4_ssim_guided(
        sub_pieces, sub_matrix, ground_truth_region, 
        beam_width=beam_width, verbose=verbose
    )
    
    # Remap back to original piece IDs
    remapped = np.full_like(arrangement, -1)
    for row in range(4):
        for col in range(4):
            if arrangement[row, col] >= 0:
                remapped[row, col] = piece_ids[arrangement[row, col]]
    
    return remapped, score


def merge_overlapping_regions(region_solutions: List[Dict], grid_size: int,
                              pieces: List[PuzzlePiece], ground_truth: np.ndarray,
                              verbose: bool = False) -> np.ndarray:
    """
    Merge overlapping region solutions using SSIM voting in overlap areas.
    
    Args:
        region_solutions: List of solved regions with arrangements and scores
        grid_size: Size of the full puzzle grid (8)
        pieces: List of puzzle pieces
        ground_truth: Full ground truth image
        verbose: Print progress
        
    Returns:
        Full 8×8 arrangement
    """
    # Track all candidates for each position with their SSIM scores
    position_candidates = {}
    
    for region_sol in region_solutions:
        arrangement = region_sol['arrangement']
        offset = region_sol['global_offset']
        region_score = region_sol['score']
        
        # Add each piece position as a candidate
        for local_row in range(4):
            for local_col in range(4):
                piece_id = arrangement[local_row, local_col]
                if piece_id < 0:
                    continue
                
                global_row = offset[0] + local_row
                global_col = offset[1] + local_col
                pos = (global_row, global_col)
                
                if pos not in position_candidates:
                    position_candidates[pos] = []
                
                # Score is region score weighted by distance from center
                center_dist = abs(local_row - 1.5) + abs(local_col - 1.5)
                weight = 1.0 / (1.0 + 0.2 * center_dist)  # Center pieces weighted more
                weighted_score = region_score * weight
                
                position_candidates[pos].append({
                    'piece_id': piece_id,
                    'score': weighted_score,
                    'votes': 1
                })
    
    # Select best candidate for each position based on score AND votes
    full_arrangement = np.full((grid_size, grid_size), -1, dtype=int)
    used_pieces = set()
    
    for pos, candidates in position_candidates.items():
        # Group by piece_id and aggregate scores and votes
        piece_data = {}
        for cand in candidates:
            pid = cand['piece_id']
            if pid not in piece_data:
                piece_data[pid] = {'score': 0.0, 'votes': 0}
            piece_data[pid]['score'] += cand['score']
            piece_data[pid]['votes'] += cand['votes']
        
        # Select piece with highest combined score (prioritize votes then score)
        # This prevents same piece being used multiple times
        best_piece = None
        best_metric = -1.0
        
        for pid, data in piece_data.items():
            if pid in used_pieces:
                continue  # Skip already used pieces
            # Combined metric: votes matter more, then score
            metric = data['votes'] * 10 + data['score']
            if metric > best_metric:
                best_metric = metric
                best_piece = pid
        
        if best_piece is not None:
            full_arrangement[pos[0], pos[1]] = best_piece
            used_pieces.add(best_piece)
    
    if verbose:
        filled = np.sum(full_arrangement >= 0)
        print(f"  Merged regions: {filled}/{grid_size*grid_size} positions filled")
        print(f"  Used {len(used_pieces)} unique pieces")
    
    return full_arrangement


def resolve_conflicts(arrangement: np.ndarray, pieces: List[PuzzlePiece],
                     compatibility_matrix: np.ndarray, ground_truth: np.ndarray,
                     grid_size: int, verbose: bool = False) -> np.ndarray:
    """
    Resolve conflicts where same piece appears multiple times and fill empty positions.
    
    Args:
        arrangement: Current arrangement (may have duplicates)
        pieces: List of puzzle pieces
        compatibility_matrix: Compatibility matrix
        ground_truth: Full ground truth image
        grid_size: Size of the puzzle grid
        verbose: Print progress
        
    Returns:
        Conflict-free arrangement with all positions filled
    """
    # This function should not be needed with improved merge
    # Just fill any remaining empty positions
    
    used = set(arrangement.flatten())
    used.discard(-1)  # Remove the "empty" marker
    unused = [i for i in range(len(pieces)) if i not in used]
    
    empty_positions = [(r, c) for r in range(grid_size) for c in range(grid_size) 
                       if arrangement[r, c] < 0]
    
    if verbose:
        print(f"  Filling {len(empty_positions)} empty positions with {len(unused)} unused pieces")
    
    if len(unused) < len(empty_positions):
        if verbose:
            print(f"  Warning: Not enough unused pieces ({len(unused)}) for empty positions ({len(empty_positions)})")
        # Use remaining pieces even if they create some duplication
        unused = list(range(len(pieces)))
    
    # Greedy fill based on compatibility with neighbors
    for row, col in empty_positions:
        if not unused:
            break
            
        best_piece = None
        best_score = -float('inf')
        
        for piece_id in unused[:]:  # Work with copy to allow removal
            if piece_id < 0:
                continue
            
            # Compute compatibility with neighbors
            score = 0.0
            count = 0
            
            for dr, dc, edge in [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbor_id = arrangement[nr, nc]
                    if neighbor_id >= 0:
                        opposite_edge = (edge + 2) % 4
                        score += compatibility_matrix[piece_id, edge, neighbor_id, opposite_edge]
                        count += 1
            
            avg_score = score / count if count > 0 else 0.0
            
            if avg_score > best_score:
                best_score = avg_score
                best_piece = piece_id
        
        if best_piece is not None:
            arrangement[row, col] = best_piece
            if best_piece in unused:
                unused.remove(best_piece)
    
    if verbose:
        remaining_empty = np.sum(arrangement < 0)
        if remaining_empty > 0:
            print(f"  Warning: {remaining_empty} positions still empty after filling")
    
    return arrangement


def solve_8x8_with_overlap(pieces: List[PuzzlePiece], 
                           compatibility_matrix: np.ndarray,
                           ground_truth: np.ndarray,
                           overlap: int = 1,
                           beam_width: int = 7,
                           verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Solve 8×8 puzzle using overlapping regions.
    
    Args:
        pieces: List of puzzle pieces
        compatibility_matrix: Compatibility matrix
        ground_truth: Full ground truth image
        overlap: Number of pieces to overlap between regions
        beam_width: Beam search width for 4×4 regions
        verbose: Print detailed progress
        
    Returns:
        (arrangement, ssim_score)
    """
    grid_size = 8
    
    if verbose:
        print(f"\n=== Solving 8×8 with {overlap}-piece overlap ===")
    
    # Extract overlapping regions
    regions = extract_overlapping_regions(pieces, grid_size, overlap)
    
    if verbose:
        print(f"Created {len(regions)} overlapping 4×4 regions")
    
    # Solve each region
    region_solutions = []
    piece_h, piece_w = pieces[0].original.shape[:2]
    region_img_size = (piece_h * 4, piece_w * 4)
    
    for i, region in enumerate(regions):
        offset = region['global_offset']
        
        # Extract ground truth for this region
        gt_h, gt_w = ground_truth.shape[:2]
        gt_piece_h = gt_h // grid_size
        gt_piece_w = gt_w // grid_size
        
        row_start = offset[0] * gt_piece_h
        row_end = (offset[0] + 4) * gt_piece_h
        col_start = offset[1] * gt_piece_w
        col_end = (offset[1] + 4) * gt_piece_w
        
        gt_region = ground_truth[row_start:row_end, col_start:col_end]
        
        # Solve this region
        if verbose:
            print(f"  Solving region {i+1}/{len(regions)} at offset {offset}")
        
        arrangement, score = solve_region_ssim_guided(
            pieces, region['piece_ids'], compatibility_matrix,
            gt_region, beam_width=beam_width, verbose=False
        )
        
        region_solutions.append({
            'arrangement': arrangement,
            'global_offset': offset,
            'score': score
        })
        
        if verbose:
            print(f"    Region SSIM: {score:.4f}")
    
    # Merge overlapping regions
    if verbose:
        print("\nMerging overlapping regions...")
    
    full_arrangement = merge_overlapping_regions(
        region_solutions, grid_size, pieces, ground_truth, verbose=verbose
    )
    
    # Resolve conflicts
    if verbose:
        print("\nResolving conflicts...")
    
    full_arrangement = resolve_conflicts(
        full_arrangement, pieces, compatibility_matrix, ground_truth,
        grid_size, verbose=verbose
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
        print(f"\n✅ Final SSIM: {final_ssim:.4f}")
    
    return full_arrangement, final_ssim
