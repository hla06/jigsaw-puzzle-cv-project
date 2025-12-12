"""
Validation and visualization utilities for puzzle assembly evaluation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from puzzle_utils import PuzzlePiece


def compute_accuracy_metrics(assembled: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive accuracy metrics between assembled and ground truth images.
    
    Args:
        assembled: Assembled puzzle image (BGR)
        ground_truth: Ground truth image (BGR)
        
    Returns:
        Dictionary containing SSIM, MSE, and PSNR scores
    """
    # Ensure same size
    if assembled.shape != ground_truth.shape:
        ground_truth = cv2.resize(ground_truth, (assembled.shape[1], assembled.shape[0]))
    
    # Convert to grayscale for SSIM
    assembled_gray = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    
    # Structural Similarity Index
    ssim_score = ssim(assembled_gray, gt_gray)
    
    # Mean Squared Error
    mse = np.mean((assembled.astype(float) - ground_truth.astype(float)) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    return {
        'ssim': ssim_score,
        'mse': mse,
        'psnr': psnr
    }


def compute_piece_placement_accuracy(arrangement: np.ndarray, pieces: List[PuzzlePiece]) -> float:
    """
    Compute percentage of pieces placed in their correct positions.
    
    Args:
        arrangement: 2D array of piece IDs showing final arrangement
        pieces: List of PuzzlePiece objects
        
    Returns:
        Accuracy percentage (0-100)
    """
    grid_size = arrangement.shape[0]
    correct_count = 0
    
    for row in range(grid_size):
        for col in range(grid_size):
            piece_id = arrangement[row, col]
            if piece_id == -1:
                continue
            
            piece = pieces[piece_id]
            if piece.original_pos == (row, col):
                correct_count += 1
    
    total_pieces = len(pieces)
    return 100.0 * correct_count / total_pieces


def visualize_comparison(scrambled: np.ndarray, assembled: np.ndarray, 
                        ground_truth: np.ndarray, save_path: Path = None):
    """
    Create side-by-side comparison visualization.
    
    Args:
        scrambled: Original scrambled puzzle
        assembled: Assembled result
        ground_truth: Correct solution
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(scrambled, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Scrambled Input", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(assembled, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Assembled Output", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def visualize_piece_correctness(assembled: np.ndarray, arrangement: np.ndarray, 
                                pieces: List[PuzzlePiece], grid_size: int,
                                save_path: Path = None):
    """
    Visualize which pieces are correctly placed with colored overlays.
    
    Args:
        assembled: Assembled puzzle image
        arrangement: 2D array of piece IDs
        pieces: List of PuzzlePiece objects
        grid_size: Size of the grid
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display assembled image
    ax.imshow(cv2.cvtColor(assembled, cv2.COLOR_BGR2RGB))
    
    # Calculate tile dimensions
    tile_h = assembled.shape[0] // grid_size
    tile_w = assembled.shape[1] // grid_size
    
    # Overlay grid with correctness indicators
    for row in range(grid_size):
        for col in range(grid_size):
            piece_id = arrangement[row, col]
            if piece_id == -1:
                continue
            
            piece = pieces[piece_id]
            is_correct = (piece.original_pos == (row, col))
            
            # Draw bounding box
            x = col * tile_w
            y = row * tile_h
            
            color = 'lime' if is_correct else 'red'
            linewidth = 5
            
            rect = plt.Rectangle((x, y), tile_w, tile_h,
                                fill=False, edgecolor=color, linewidth=linewidth)
            ax.add_patch(rect)
            
            # Add label
            status_symbol = "✓" if is_correct else "✗"
            label_text = f"{status_symbol}\nPiece {piece_id}"
            
            ax.text(x + tile_w//2, y + tile_h//2,
                   label_text,
                   color='white', fontsize=16, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=color, alpha=0.8, edgecolor='white', linewidth=2))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lime', edgecolor='white', label='Correct Position'),
        Patch(facecolor='red', edgecolor='white', label='Incorrect Position')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title("Piece Placement Correctness", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved correctness visualization to: {save_path}")
    
    plt.show()


def visualize_compatibility_heatmap(compatibility_matrix: np.ndarray, 
                                   grid_size: int, save_path: Path = None):
    """
    Visualize compatibility matrix as a heatmap.
    
    Args:
        compatibility_matrix: 4D compatibility array
        grid_size: Size of puzzle grid
        save_path: Optional path to save the figure
    """
    n_pieces = compatibility_matrix.shape[0]
    
    # Create simplified matrix (max score across all edge pairs)
    simple_matrix = np.zeros((n_pieces, n_pieces))
    
    for i in range(n_pieces):
        for j in range(n_pieces):
            if i != j:
                simple_matrix[i, j] = compatibility_matrix[i, :, j, :].max()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(simple_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Compatibility Score', fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(n_pieces))
    ax.set_yticks(range(n_pieces))
    ax.set_xlabel('Piece ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Piece ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Piece Compatibility Matrix ({grid_size}x{grid_size} Puzzle)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(n_pieces):
        for j in range(n_pieces):
            if i != j:
                text_color = 'black' if simple_matrix[i, j] < 0.6 else 'white'
                ax.text(j, i, f"{simple_matrix[i, j]:.2f}",
                       ha="center", va="center", color=text_color, 
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved compatibility heatmap to: {save_path}")
    
    plt.show()


def print_evaluation_report(puzzle_name: str, metrics: Dict[str, float], 
                           piece_accuracy: float, elapsed_time: float):
    """
    Print a formatted evaluation report.
    
    Args:
        puzzle_name: Name/ID of the puzzle
        metrics: Dictionary with SSIM, MSE, PSNR
        piece_accuracy: Percentage of correctly placed pieces
        elapsed_time: Time taken for assembly
    """
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT: {puzzle_name}")
    print("=" * 70)
    print(f"\n  Image Quality Metrics:")
    print(f"    • Structural Similarity (SSIM):     {metrics['ssim']:.4f}")
    print(f"    • Mean Squared Error (MSE):         {metrics['mse']:.2f}")
    print(f"    • Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB")
    print(f"\n  Piece Placement:")
    print(f"    • Accuracy:                         {piece_accuracy:.1f}%")
    print(f"\n  Performance:")
    print(f"    • Assembly Time:                    {elapsed_time:.2f} seconds")
    print("\n" + "=" * 70 + "\n")


def create_summary_figure(results: List[Dict], save_path: Path = None):
    """
    Create a summary figure showing results across multiple puzzles.
    
    Args:
        results: List of result dictionaries with keys: 'puzzle_id', 'ssim', 'piece_accuracy'
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    puzzle_ids = [r['puzzle_id'] for r in results]
    ssim_scores = [r['ssim'] for r in results]
    accuracies = [r['piece_accuracy'] for r in results]
    
    # SSIM scores
    axes[0].bar(puzzle_ids, ssim_scores, color='steelblue', alpha=0.8)
    axes[0].axhline(y=np.mean(ssim_scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(ssim_scores):.3f}')
    axes[0].set_xlabel('Puzzle ID', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('SSIM Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Structural Similarity (SSIM)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Piece accuracy
    axes[1].bar(puzzle_ids, accuracies, color='forestgreen', alpha=0.8)
    axes[1].axhline(y=np.mean(accuracies), color='red', linestyle='--',
                    label=f'Mean: {np.mean(accuracies):.1f}%')
    axes[1].set_xlabel('Puzzle ID', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Piece Placement Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary figure to: {save_path}")
    
    plt.show()
