"""
Visualization utilities for SAR processing pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


def plot_chirp(t: np.ndarray, chirp: np.ndarray, 
               save_path: Optional[str] = None) -> None:
    """Plot chirp signal."""
    plt.figure(figsize=(10, 3))
    plt.plot(t * 1e6, np.real(chirp), label="Real", linewidth=1.5)
    plt.plot(t * 1e6, np.imag(chirp), label="Imaginary", linewidth=1.5, alpha=0.8)
    plt.title("LFM Chirp Signal", fontsize=12, fontweight='bold')
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_sar_image(data: np.ndarray, title: str, 
                   cmap: str = 'gray', save_path: Optional[str] = None) -> None:
    """Plot SAR image in dB scale."""
    plt.figure(figsize=(8, 5))
    img_db = 20 * np.log10(np.abs(data) + 1e-9)
    plt.imshow(img_db, aspect='auto', cmap=cmap)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Azimuth (Pulse Index)")
    plt.ylabel("Range Bin")
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_metrics(metrics: Dict, save_path: Optional[str] = None) -> None:
    """Plot quality metrics summary."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    metric_names = ['PSLR\n(Range)', 'PSLR\n(Azimuth)', 
                   'ISLR\n(Range)', 'ISLR\n(Azimuth)', 'ENL']
    values = [metrics['pslr_range'], metrics['pslr_azimuth'],
             metrics['islr_range'], metrics['islr_azimuth'], metrics['enl']]
    
    colors = ['#2E86AB', '#A23B72', '#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Value (dB or dimensionless)', fontweight='bold')
    ax.set_title('SAR Image Quality Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.show()
