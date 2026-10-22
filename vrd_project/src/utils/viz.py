"""
Visualization Utilities for VRD
================================

Provides functions for drawing scene graphs, visualizing predictions,
and saving results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb


def generate_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    Generate n distinct colors.

    Args:
        n: Number of colors to generate

    Returns:
        List of RGB tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        value = 0.9
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(tuple(rgb))
    return colors


def draw_boxes_with_labels(
    ax,
    boxes: np.ndarray,
    labels: List[str],
    colors: Optional[List] = None,
    linewidth: float = 2.0,
    fontsize: int = 10
) -> None:
    """
    Draw bounding boxes with labels on a matplotlib axis.

    Args:
        ax: Matplotlib axis
        boxes: [N, 4] array of boxes [x1, y1, x2, y2]
        labels: List of label strings
        colors: Optional list of colors
        linewidth: Line width for boxes
        fontsize: Font size for labels
    """
    n_boxes = len(boxes)
    if colors is None:
        colors = generate_colors(n_boxes)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]

        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Draw label
        ax.text(
            x1, y1 - 5,
            f"[{i}] {label}",
            fontsize=fontsize,
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
        )


def visualize_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    relations: List[Tuple[int, int, str, float]],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    max_relations: int = 15
) -> Optional[plt.Figure]:
    """
    Visualize VRD predictions on an image.

    Args:
        image: Image as numpy array (H, W, 3)
        boxes: [N, 4] object boxes
        labels: Object label strings
        relations: List of (subj_idx, obj_idx, predicate, score) tuples
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
        max_relations: Maximum relations to display

    Returns:
        Figure object if show=False, else None
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Image with boxes
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title("Objects", fontsize=12)
    ax1.axis('off')

    colors = generate_colors(len(boxes))
    draw_boxes_with_labels(ax1, boxes, labels, colors)

    # Right: Relations list
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title("Predicted Relations", fontsize=12)

    # Sort relations by score
    sorted_relations = sorted(relations, key=lambda x: x[3], reverse=True)
    display_relations = sorted_relations[:max_relations]

    # Create text
    text_lines = []
    for i, (subj_idx, obj_idx, pred, score) in enumerate(display_relations):
        subj_label = labels[subj_idx] if subj_idx < len(labels) else f"obj_{subj_idx}"
        obj_label = labels[obj_idx] if obj_idx < len(labels) else f"obj_{obj_idx}"
        line = f"{i+1}. [{subj_idx}]{subj_label} --{pred}--> [{obj_idx}]{obj_label} ({score:.3f})"
        text_lines.append(line)

    if not text_lines:
        text_lines = ["No relations predicted"]

    text = '\n'.join(text_lines)
    ax2.text(
        0.05, 0.95, text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
        return None

    return fig


def draw_scene_graph(
    objects: List[Dict],
    relations: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[plt.Figure]:
    """
    Draw a scene graph visualization (graph layout).

    Args:
        objects: List of object dicts with 'id' and 'label'
        relations: List of relation dicts with 'subj_id', 'obj_id', 'predicate'
        save_path: Optional path to save
        show: Whether to display
        figsize: Figure size

    Returns:
        Figure if show=False
    """
    try:
        import networkx as nx
    except ImportError:
        print("Warning: networkx not installed. Cannot draw graph layout.")
        print("Install with: pip install networkx")
        return None

    # Create graph
    G = nx.DiGraph()

    # Add nodes
    for obj in objects:
        G.add_node(obj['id'], label=obj['label'])

    # Add edges
    for rel in relations:
        G.add_edge(
            rel['subj_id'],
            rel['obj_id'],
            predicate=rel['predicate'],
            score=rel.get('score', 1.0)
        )

    # Draw
    fig, ax = plt.subplots(figsize=figsize)

    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except Exception:
        pos = nx.circular_layout(G)

    # Draw nodes
    node_labels = {obj['id']: f"{obj['id']}: {obj['label']}" for obj in objects}
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color='lightblue',
        node_size=2000,
        alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8)

    # Draw edges
    edge_labels = {
        (rel['subj_id'], rel['obj_id']): rel['predicate']
        for rel in relations
    }
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        alpha=0.7,
        connectionstyle="arc3,rad=0.1"
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=7,
        font_color='red'
    )

    ax.set_title("Scene Graph", fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scene graph to {save_path}")

    if show:
        plt.show()
        return None

    return fig


def save_scene_graph_json(
    objects: List[Dict],
    relations: List[Dict],
    save_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save scene graph to JSON file.

    Args:
        objects: List of object dicts with 'id', 'label', 'box'
        relations: List of relation dicts with 'subj_id', 'obj_id', 'predicate', 'score'
        save_path: Path to save JSON
        metadata: Optional metadata to include
    """
    scene_graph = {
        'objects': objects,
        'relations': relations
    }

    if metadata:
        scene_graph['metadata'] = metadata

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(scene_graph, f, indent=2)

    print(f"Saved scene graph JSON to {save_path}")


def load_scene_graph_json(load_path: str) -> Dict:
    """
    Load scene graph from JSON file.

    Args:
        load_path: Path to JSON file

    Returns:
        Scene graph dictionary
    """
    with open(load_path, 'r') as f:
        return json.load(f)


def plot_training_curves(
    csv_path: str,
    metrics: List[str] = ['loss', 'R@50', 'mR@50'],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 4)
) -> Optional[plt.Figure]:
    """
    Plot training curves from CSV log.

    Args:
        csv_path: Path to CSV file
        metrics: List of metrics to plot
        save_path: Optional path to save figure
        show: Whether to display
        figsize: Figure size

    Returns:
        Figure if show=False
    """
    import csv

    # Load CSV
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)

    # Filter to requested metrics that exist
    available_metrics = [m for m in metrics if m in data]

    if not available_metrics:
        print(f"Warning: None of {metrics} found in CSV. Available: {list(data.keys())}")
        return None

    n_plots = len(available_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        values = data[metric]
        epochs = range(1, len(values) + 1)

        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    if show:
        plt.show()
        return None

    return fig


def print_relation_summary(
    predictions: List[Tuple[int, int, str, float]],
    labels: List[str],
    top_k: int = 10
) -> str:
    """
    Create a printable summary of predicted relations.

    Args:
        predictions: List of (subj_idx, obj_idx, predicate, score)
        labels: Object label strings
        top_k: Number of top predictions to show

    Returns:
        Formatted string
    """
    lines = ["\n" + "="*60, "Top Predicted Relations", "="*60]

    sorted_preds = sorted(predictions, key=lambda x: x[3], reverse=True)

    for i, (subj_idx, obj_idx, pred, score) in enumerate(sorted_preds[:top_k]):
        subj = labels[subj_idx] if subj_idx < len(labels) else f"obj_{subj_idx}"
        obj = labels[obj_idx] if obj_idx < len(labels) else f"obj_{obj_idx}"
        lines.append(f"  {i+1:2d}. {subj} --[{pred}]--> {obj}  (score: {score:.4f})")

    lines.append("="*60 + "\n")

    return '\n'.join(lines)
