"""
Edge matching and compatibility scoring for puzzle pieces.
Implements multi-source scoring with color prioritization.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List
from puzzle_utils import PuzzlePiece, extract_edge_strip, get_complementary_edge


def compute_color_histogram_similarity(strip1: np.ndarray, strip2: np.ndarray, bins: int = 32) -> float:
    """
    Compute histogram correlation between two color edge strips.
    
    Args:
        strip1: First edge strip (BGR color)
        strip2: Second edge strip (BGR color)
        bins: Number of histogram bins per channel
        
    Returns:
        Similarity score [0-1], where 1 is perfect match
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    # Ensure strips are in the correct format for calcHist
    # calcHist expects images as uint8
    if strip1.dtype != np.uint8:
        strip1 = strip1.astype(np.uint8)
    if strip2.dtype != np.uint8:
        strip2 = strip2.astype(np.uint8)
    
    # Compute histograms for each channel with higher resolution
    similarities = []
    for channel in range(3):  # B, G, R
        hist1 = cv2.calcHist([strip1], [channel], None, [bins], [0, 256])
        hist2 = cv2.calcHist([strip2], [channel], None, [bins], [0, 256])
        
        # Normalize
        hist1 = hist1.flatten() / (hist1.sum() + 1e-7)
        hist2 = hist2.flatten() / (hist2.sum() + 1e-7)
        
        # Compute correlation
        correlation = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
        similarities.append(correlation)
    
    # Average across channels, map from [-1, 1] to [0, 1]
    avg_similarity = np.mean(similarities)
    return (avg_similarity + 1) / 2


def compute_color_difference(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Compute pixel-wise color difference between two edge strips.
    
    Args:
        strip1: First edge strip (BGR color)
        strip2: Second edge strip (BGR color)
        
    Returns:
        Similarity score [0-1], where 1 means very similar colors
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    # Compute pixel-wise difference along the boundary
    # For horizontal edges, compare the boundary rows
    # For vertical edges, compare the boundary columns
    if strip1.shape[0] < strip1.shape[1]:  # Horizontal edge
        boundary1 = strip1[-1, :, :] if strip1.shape[0] > 1 else strip1[0, :, :]
        boundary2 = strip2[0, :, :] if strip2.shape[0] > 1 else strip2[0, :, :]
    else:  # Vertical edge
        boundary1 = strip1[:, -1, :] if strip1.shape[1] > 1 else strip1[:, 0, :]
        boundary2 = strip2[:, 0, :] if strip2.shape[1] > 1 else strip2[:, 0, :]
    
    # Compute mean absolute difference
    pixel_diff = np.mean(np.abs(boundary1.astype(np.float32) - boundary2.astype(np.float32)))
    
    # Normalize to [0, 1], where 1 is best match
    similarity = 1.0 - min(pixel_diff / 255.0, 1.0)
    
    return similarity


def compute_gradient_continuity(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Compute gradient continuity between two edge strips from enhanced images.
    
    Args:
        strip1: First edge strip (grayscale)
        strip2: Second edge strip (grayscale)
        
    Returns:
        Similarity score [0-1], where 1 means smooth gradient transition
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    # Compute gradients in both X and Y directions
    grad1_x = cv2.Sobel(strip1.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad1_y = cv2.Sobel(strip1.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad2_x = cv2.Sobel(strip2.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad2_y = cv2.Sobel(strip2.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
    # Extract boundary gradients
    if strip1.shape[0] < strip1.shape[1]:  # Horizontal edge
        boundary1 = mag1[-1, :] if mag1.shape[0] > 1 else mag1[0, :]
        boundary2 = mag2[0, :] if mag2.shape[0] > 1 else mag2[0, :]
    else:  # Vertical edge
        boundary1 = mag1[:, -1] if mag1.shape[1] > 1 else mag1[:, 0]
        boundary2 = mag2[:, 0] if mag2.shape[1] > 1 else mag2[:, 0]
    
    # Compute normalized cross-correlation
    if np.std(boundary1) < 1e-5 or np.std(boundary2) < 1e-5:
        return 0.5
    
    correlation = np.corrcoef(boundary1.flatten(), boundary2.flatten())[0, 1]
    
    # Map from [-1, 1] to [0, 1]
    return max((correlation + 1) / 2, 0.0)


def check_mask_boundary_alignment(strip1: np.ndarray, strip2: np.ndarray, threshold: int = 200) -> float:
    """
    Check if mask boundaries align (both should have white pixels at edges).
    
    Args:
        strip1: First edge strip from mask (grayscale)
        strip2: Second edge strip from mask (grayscale)
        threshold: Threshold for white pixel detection
        
    Returns:
        Alignment score [0-1], where 1 means good alignment
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    # Count white pixels in each strip
    white1 = np.sum(strip1 > threshold)
    white2 = np.sum(strip2 > threshold)
    
    total_pixels = strip1.size
    
    # If both strips have significant white pixels, they likely align
    white_ratio1 = white1 / total_pixels
    white_ratio2 = white2 / total_pixels
    
    # Both should have high white pixel ratio for good alignment
    min_ratio = min(white_ratio1, white_ratio2)
    
    return min_ratio


def compute_texture_similarity(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Compute texture similarity using Local Binary Pattern-like features.
    
    Args:
        strip1: First edge strip (grayscale)
        strip2: Second edge strip (grayscale)
        
    Returns:
        Similarity score [0-1], where 1 means similar texture
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    # Compute standard deviation in local windows as texture measure
    def compute_texture_energy(strip):
        if strip.size < 9:
            return np.std(strip)
        kernel_size = min(3, min(strip.shape[0], strip.shape[1]))
        if kernel_size < 2:
            return np.std(strip)
        blurred = cv2.GaussianBlur(strip.astype(np.float32), (kernel_size, kernel_size), 0)
        texture = np.abs(strip.astype(np.float32) - blurred)
        return np.mean(texture)
    
    energy1 = compute_texture_energy(strip1)
    energy2 = compute_texture_energy(strip2)
    
    # Compute similarity based on energy difference
    max_energy = max(energy1, energy2, 1.0)
    diff = abs(energy1 - energy2)
    similarity = 1.0 - min(diff / max_energy, 1.0)
    
    return similarity


def compute_color_similarity_lab(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Compute perceptual color similarity using LAB color space.
    
    Args:
        strip1: First edge strip (BGR color)
        strip2: Second edge strip (BGR color)
        
    Returns:
        Similarity score [0-1], where 1 means very similar perceptual colors
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    # Convert to LAB color space for perceptual comparison
    lab1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Extract boundary pixels
    if strip1.shape[0] < strip1.shape[1]:  # Horizontal edge
        boundary1 = lab1[-1, :, :] if lab1.shape[0] > 1 else lab1[0, :, :]
        boundary2 = lab2[0, :, :] if lab2.shape[0] > 1 else lab2[0, :, :]
    else:  # Vertical edge
        boundary1 = lab1[:, -1, :] if lab1.shape[1] > 1 else lab1[:, 0, :]
        boundary2 = lab2[:, 0, :] if lab2.shape[1] > 1 else lab2[:, 0, :]
    
    # Compute Delta E (CIE76 simplified)
    delta_e = np.sqrt(np.sum((boundary1 - boundary2) ** 2, axis=-1))
    mean_delta_e = np.mean(delta_e)
    
    # Normalize: typical max Delta E is around 100, map to [0, 1]
    similarity = 1.0 - min(mean_delta_e / 100.0, 1.0)
    
    return similarity


def compute_multiscale_gradient(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Compute gradient continuity at multiple scales.
    
    Args:
        strip1: First edge strip (grayscale)
        strip2: Second edge strip (grayscale)
        
    Returns:
        Similarity score [0-1], where 1 means smooth gradient transition
    """
    if strip1.shape != strip2.shape:
        return 0.0
    
    scores = []
    
    # Try different kernel sizes for multi-scale analysis
    for ksize in [3, 5]:
        if min(strip1.shape[0], strip1.shape[1]) < ksize:
            continue
            
        # Compute gradients
        grad1_x = cv2.Sobel(strip1.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ksize)
        grad1_y = cv2.Sobel(strip1.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ksize)
        grad2_x = cv2.Sobel(strip2.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ksize)
        grad2_y = cv2.Sobel(strip2.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ksize)
        
        # Compute gradient magnitude
        mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
        mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Extract boundary gradients
        if strip1.shape[0] < strip1.shape[1]:  # Horizontal edge
            boundary1 = mag1[-1, :] if mag1.shape[0] > 1 else mag1[0, :]
            boundary2 = mag2[0, :] if mag2.shape[0] > 1 else mag2[0, :]
        else:  # Vertical edge
            boundary1 = mag1[:, -1] if mag1.shape[1] > 1 else mag1[:, 0]
            boundary2 = mag2[:, 0] if mag2.shape[1] > 1 else mag2[:, 0]
        
        # Compute normalized cross-correlation
        if np.std(boundary1) < 1e-5 or np.std(boundary2) < 1e-5:
            scores.append(0.5)
        else:
            correlation = np.corrcoef(boundary1.flatten(), boundary2.flatten())[0, 1]
            scores.append(max((correlation + 1) / 2, 0.0))
    
    # Return average across scales
    return np.mean(scores) if scores else 0.5


def get_adaptive_strip_width(grid_size: int) -> int:
    """
    Calculate adaptive strip width based on grid size.
    Larger grids need wider strips for better matching.
    
    Args:
        grid_size: Size of the puzzle grid (2, 4, or 8)
        
    Returns:
        Strip width in pixels
    """
    width_map = {2: 3, 4: 5, 8: 7}
    return width_map.get(grid_size, 3)


def compute_edge_compatibility(piece1: PuzzlePiece, edge1: str, 
                                piece2: PuzzlePiece, edge2: str,
                                strip_width: int = 3,
                                weights: Dict[str, float] = None,
                                use_enhanced_features: bool = False) -> float:
    """
    Compute overall compatibility score between two edges.
    
    Args:
        piece1: First puzzle piece
        edge1: Edge of first piece ('top', 'right', 'bottom', 'left')
        piece2: Second puzzle piece
        edge2: Edge of second piece (should be complementary to edge1)
        strip_width: Width of edge strip in pixels
        weights: Dictionary of weights for different metrics
                 Default: {'color_hist': 0.25, 'color_mean': 0.25, 'color_lab': 0.15, 
                          'gradient': 0.15, 'gradient_multi': 0.1, 'texture': 0.05, 'mask': 0.05}
        use_enhanced_features: If True, use LAB color space and multi-scale gradients (slower but more accurate)
        
    Returns:
        Compatibility score [0-1], where higher is better
    """
    if weights is None:
        if use_enhanced_features:
            weights = {
                'color_hist': 0.25,
                'color_mean': 0.25,
                'color_lab': 0.15,
                'gradient': 0.15,
                'gradient_multi': 0.10,
                'texture': 0.05,
                'mask': 0.05
            }
        else:
            weights = {
                'color_hist': 0.30,
                'color_mean': 0.35,
                'gradient': 0.20,
                'texture': 0.10,
                'mask': 0.05
            }
    
    # Extract edge strips
    strip1_original = extract_edge_strip(piece1.original, edge1, strip_width)
    strip2_original = extract_edge_strip(piece2.original, edge2, strip_width)
    
    strip1_enhanced = extract_edge_strip(piece1.enhanced, edge1, strip_width)
    strip2_enhanced = extract_edge_strip(piece2.enhanced, edge2, strip_width)
    
    strip1_mask = extract_edge_strip(piece1.mask, edge1, strip_width)
    strip2_mask = extract_edge_strip(piece2.mask, edge2, strip_width)
    
    # Compute individual scores
    color_hist_score = compute_color_histogram_similarity(strip1_original, strip2_original)
    color_mean_score = compute_color_difference(strip1_original, strip2_original)
    gradient_score = compute_gradient_continuity(strip1_enhanced, strip2_enhanced)
    texture_score = compute_texture_similarity(strip1_enhanced, strip2_enhanced)
    mask_score = check_mask_boundary_alignment(strip1_mask, strip2_mask)
    
    # Base weighted combination
    total_score = (
        weights['color_hist'] * color_hist_score +
        weights['color_mean'] * color_mean_score +
        weights['gradient'] * gradient_score +
        weights['texture'] * texture_score +
        weights['mask'] * mask_score
    )
    
    # Add enhanced features if requested (for larger grids)
    if use_enhanced_features:
        color_lab_score = compute_color_similarity_lab(strip1_original, strip2_original)
        gradient_multi_score = compute_multiscale_gradient(strip1_enhanced, strip2_enhanced)
        
        total_score += (
            weights['color_lab'] * color_lab_score +
            weights['gradient_multi'] * gradient_multi_score
        )
    
    return total_score


def build_compatibility_matrix(pieces: List[PuzzlePiece], strip_width: int = 3, 
                              grid_size: int = 2, use_enhanced_features: bool = False) -> np.ndarray:
    """
    Build a compatibility matrix for all piece-edge pairs.
    
    Args:
        pieces: List of PuzzlePiece objects
        strip_width: Width of edge strip in pixels (will be overridden by adaptive width if grid_size provided)
        grid_size: Size of the puzzle grid (2, 4, or 8) - used for adaptive strip width
        use_enhanced_features: If True, use LAB color space and multi-scale gradients (recommended for 4x4+)
        
    Returns:
        4D array of shape (n_pieces, 4, n_pieces, 4) where
        matrix[i, e1, j, e2] is the compatibility between piece i's edge e1 and piece j's edge e2
        Edge encoding: 0=top, 1=right, 2=bottom, 3=left
    """
    n_pieces = len(pieces)
    edge_names = ['top', 'right', 'bottom', 'left']
    
    # Use adaptive strip width for larger grids
    if grid_size > 2:
        strip_width = get_adaptive_strip_width(grid_size)
        use_enhanced_features = True  # Always use enhanced features for 4x4+
    
    # Initialize matrix
    matrix = np.zeros((n_pieces, 4, n_pieces, 4), dtype=np.float32)
    
    # Compute all pairwise edge compatibilities
    for i, piece1 in enumerate(pieces):
        for e1_idx, edge1 in enumerate(edge_names):
            for j, piece2 in enumerate(pieces):
                if i == j:  # Skip same piece
                    continue
                
                for e2_idx, edge2 in enumerate(edge_names):
                    # Only compute if edges are complementary
                    if get_complementary_edge(edge1) == edge2:
                        score = compute_edge_compatibility(
                            piece1, edge1, piece2, edge2, 
                            strip_width, use_enhanced_features=use_enhanced_features
                        )
                        matrix[i, e1_idx, j, e2_idx] = score
    
    return matrix


def build_position_aware_compatibility_matrix(pieces: List[PuzzlePiece], 
                                             arrangement: np.ndarray,
                                             strip_width: int = 3,
                                             grid_size: int = 8,
                                             position_weight: float = 0.2) -> np.ndarray:
    """
    Build compatibility matrix with position-aware weighting.
    Boosts scores for pieces near their original positions.
    
    Args:
        pieces: List of PuzzlePiece objects
        arrangement: Current arrangement (or initial arrangement)
        strip_width: Width of edge strip in pixels
        grid_size: Size of the puzzle grid (typically 8)
        position_weight: Weight for position bonus (0.2 = 20% boost)
        
    Returns:
        4D position-aware compatibility matrix
    """
    # Start with base compatibility matrix
    base_matrix = build_compatibility_matrix(pieces, strip_width, grid_size, use_enhanced_features=True)
    
    n_pieces = len(pieces)
    edge_names = ['top', 'right', 'bottom', 'left']
    
    # Apply position-aware weighting
    for i, piece1 in enumerate(pieces):
        for e1_idx in range(4):
            for j, piece2 in enumerate(pieces):
                if i == j:
                    continue
                
                for e2_idx in range(4):
                    base_score = base_matrix[i, e1_idx, j, e2_idx]
                    
                    if base_score == 0:
                        continue
                    
                    # Find current positions in arrangement
                    pos1 = np.argwhere(arrangement == i)
                    pos2 = np.argwhere(arrangement == j)
                    
                    if len(pos1) > 0 and len(pos2) > 0:
                        current_pos1 = tuple(pos1[0])
                        current_pos2 = tuple(pos2[0])
                        
                        # Manhattan distance from original position
                        dist1 = abs(current_pos1[0] - piece1.original_pos[0]) + abs(current_pos1[1] - piece1.original_pos[1])
                        dist2 = abs(current_pos2[0] - piece2.original_pos[0]) + abs(current_pos2[1] - piece2.original_pos[1])
                        
                        # Pieces closer to original position get higher weight
                        avg_distance = (dist1 + dist2) / 2.0
                        max_distance = grid_size * 2  # Maximum possible distance
                        
                        # Position bonus: closer to original = higher bonus
                        position_bonus = position_weight * (1.0 - avg_distance / max_distance)
                        
                        # Apply bonus
                        base_matrix[i, e1_idx, j, e2_idx] = base_score * (1.0 + position_bonus)
    
    return base_matrix


def get_top_matches(compatibility_matrix: np.ndarray, k: int = 5) -> List[Tuple[int, int, int, int, float]]:
    """
    Get top k matching edge pairs from compatibility matrix.
    
    Args:
        compatibility_matrix: 4D compatibility matrix from build_compatibility_matrix
        k: Number of top matches to return
        
    Returns:
        List of tuples (piece1_id, edge1_idx, piece2_id, edge2_idx, score)
    """
    n_pieces = compatibility_matrix.shape[0]
    
    # Flatten and get top k
    matches = []
    for i in range(n_pieces):
        for e1 in range(4):
            for j in range(n_pieces):
                for e2 in range(4):
                    score = compatibility_matrix[i, e1, j, e2]
                    if score > 0:
                        matches.append((i, e1, j, e2, score))
    
    # Sort by score
    matches.sort(key=lambda x: x[4], reverse=True)
    
    return matches[:k]
