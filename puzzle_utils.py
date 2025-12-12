"""
Utility functions for jigsaw puzzle solving (Phase 2)
Provides grid slicing, edge extraction, and piece management.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np


class PuzzlePiece:
    """
    Represents a single puzzle piece with its original, enhanced, and mask data.
    """
    def __init__(self, original: np.ndarray, enhanced: np.ndarray, mask: np.ndarray, 
                 piece_id: int, original_pos: Tuple[int, int]):
        """
        Initialize a puzzle piece.
        
        Args:
            original: Original color image tile (BGR)
            enhanced: Enhanced/thresholded tile from Phase 1
            mask: Mask tile from Phase 1
            piece_id: Unique identifier for this piece
            original_pos: Original (row, col) position in the grid
        """
        self.original = original
        self.enhanced = enhanced
        self.mask = mask
        self.piece_id = piece_id
        self.original_pos = original_pos
        self.current_pos = None  # Will be set during assembly
        
    def get_height(self):
        return self.original.shape[0]
    
    def get_width(self):
        return self.original.shape[1]


def slice_into_grid(image: np.ndarray, grid_size: int) -> List[np.ndarray]:
    """
    Slice an image into a grid of equal-sized tiles.
    
    Args:
        image: Input image (can be grayscale or color)
        grid_size: Size of the grid (e.g., 2 for 2x2, 4 for 4x4)
        
    Returns:
        List of image tiles in row-major order
    """
    h, w = image.shape[:2]
    tile_h = h // grid_size
    tile_w = w // grid_size
    
    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * tile_h
            y_end = (row + 1) * tile_h
            x_start = col * tile_w
            x_end = (col + 1) * tile_w
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)
    
    return tiles


def create_puzzle_pieces(original_path: str, enhanced_path: str, mask_path: str, 
                         grid_size: int) -> List[PuzzlePiece]:
    """
    Load images and create PuzzlePiece objects for all tiles.
    
    Args:
        original_path: Path to original scrambled puzzle image
        enhanced_path: Path to Phase 1 enhanced image
        mask_path: Path to Phase 1 mask image
        grid_size: Size of the grid (2, 4, or 8)
        
    Returns:
        List of PuzzlePiece objects
    """
    # Load images
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or enhanced is None or mask is None:
        raise ValueError(f"Failed to load images: {original_path}, {enhanced_path}, {mask_path}")
    
    # Slice into grids
    original_tiles = slice_into_grid(original, grid_size)
    enhanced_tiles = slice_into_grid(enhanced, grid_size)
    mask_tiles = slice_into_grid(mask, grid_size)
    
    # Create PuzzlePiece objects
    pieces = []
    piece_id = 0
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            piece = PuzzlePiece(
                original=original_tiles[idx],
                enhanced=enhanced_tiles[idx],
                mask=mask_tiles[idx],
                piece_id=piece_id,
                original_pos=(row, col)
            )
            pieces.append(piece)
            piece_id += 1
    
    return pieces


def extract_edge_strip(image: np.ndarray, edge: str, strip_width: int = 3) -> np.ndarray:
    """
    Extract a strip of pixels from one edge of an image.
    
    Args:
        image: Input image (grayscale or color)
        edge: Edge to extract ('top', 'right', 'bottom', 'left')
        strip_width: Width of the strip in pixels (default: 3)
        
    Returns:
        Extracted edge strip as numpy array
    """
    h, w = image.shape[:2]
    
    if edge == 'top':
        return image[:strip_width, :].copy()
    elif edge == 'bottom':
        return image[-strip_width:, :].copy()
    elif edge == 'left':
        return image[:, :strip_width].copy()
    elif edge == 'right':
        return image[:, -strip_width:].copy()
    else:
        raise ValueError(f"Invalid edge: {edge}. Must be 'top', 'right', 'bottom', or 'left'")


def get_all_edge_strips(piece: PuzzlePiece, strip_width: int = 3) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract all edge strips from a puzzle piece (original, enhanced, and mask).
    
    Args:
        piece: PuzzlePiece object
        strip_width: Width of the strip in pixels (default: 3)
        
    Returns:
        Dictionary with structure: {edge: {source: strip}}
        Example: {'top': {'original': array, 'enhanced': array, 'mask': array}, ...}
    """
    edges = ['top', 'right', 'bottom', 'left']
    edge_data = {}
    
    for edge in edges:
        edge_data[edge] = {
            'original': extract_edge_strip(piece.original, edge, strip_width),
            'enhanced': extract_edge_strip(piece.enhanced, edge, strip_width),
            'mask': extract_edge_strip(piece.mask, edge, strip_width)
        }
    
    return edge_data


def get_complementary_edge(edge: str) -> str:
    """
    Get the complementary edge that should match with the given edge.
    
    Args:
        edge: Edge name ('top', 'right', 'bottom', 'left')
        
    Returns:
        Complementary edge name
    """
    complement = {
        'top': 'bottom',
        'bottom': 'top',
        'left': 'right',
        'right': 'left'
    }
    return complement[edge]


def assemble_puzzle(pieces: List[PuzzlePiece], arrangement: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Assemble puzzle pieces into a complete image based on arrangement.
    
    Args:
        pieces: List of PuzzlePiece objects
        arrangement: 2D array of piece IDs showing the arrangement (grid_size x grid_size)
        grid_size: Size of the grid
        
    Returns:
        Assembled image
    """
    # Create a mapping from piece_id to piece
    piece_map = {p.piece_id: p for p in pieces}
    
    # Get dimensions from first piece
    tile_h = pieces[0].get_height()
    tile_w = pieces[0].get_width()
    
    # Create output image
    output_h = tile_h * grid_size
    output_w = tile_w * grid_size
    assembled = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    # Place pieces according to arrangement
    for row in range(grid_size):
        for col in range(grid_size):
            piece_id = arrangement[row, col]
            if piece_id == -1:  # Empty slot
                continue
            
            piece = piece_map[piece_id]
            y_start = row * tile_h
            y_end = (row + 1) * tile_h
            x_start = col * tile_w
            x_end = (col + 1) * tile_w
            
            assembled[y_start:y_end, x_start:x_end] = piece.original
    
    return assembled
