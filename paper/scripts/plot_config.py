# Plot-style portion copied from /home/strongich/PythonProj/SELF/lever/LEVER/config.py
# Snapshot: 2026-10-19. No external LEVER runtime dependency.
"""
Publication-quality plotting configuration for the project.

This module provides matplotlib settings for paper-ready plots including:
- High-DPI vector and raster output
- Colorblind-friendly palette (Okabe-Ito scheme)
- Clean, minimal styling optimized for single-column figures
- Type 42 fonts for Illustrator compatibility
"""

import matplotlib as mpl

# Publication-ready matplotlib configuration
PAPER_STYLE = {
    # Figure dimensions optimized for single-column papers
    "figure.figsize": (6.0, 3.6),  # Single-column width
    "figure.dpi": 150,  # Display resolution
    # High-quality output settings
    "savefig.dpi": 300,  # Publication DPI
    "savefig.bbox": "tight",  # Tight bounding box
    "savefig.pad_inches": 0.1,  # Minimal padding
    "savefig.format": "pdf",  # Default vector format
    # Clean, minimal styling
    "axes.grid": True,  # Enable grid
    "grid.alpha": 0.25,  # Subtle grid
    "axes.spines.top": False,  # Remove top spine
    "axes.spines.right": False,  # Remove right spine
    "axes.linewidth": 0.8,  # Thin axes
    "axes.edgecolor": "#333333",  # Dark gray axes
    # Typography optimized for papers
    "axes.labelsize": 11,  # Axis label size
    "axes.titlesize": 12,  # Title size
    "xtick.labelsize": 10,  # X-tick size
    "ytick.labelsize": 10,  # Y-tick size
    "legend.fontsize": 9,  # Legend size
    "legend.frameon": False,  # No legend frame
    "font.family": "sans-serif",  # Font family
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    # Line and marker styling
    "lines.linewidth": 2.0,  # Line thickness
    "lines.markersize": 4,  # Marker size
    "lines.markeredgewidth": 0.5,  # Marker edge thickness
    # Vector output compatibility
    "pdf.fonttype": 42,  # Type 42 fonts (editable in Illustrator)
    "ps.fonttype": 42,  # Type 42 fonts for PostScript
    "svg.fonttype": "none",  # Embed fonts in SVG
}

# Colorblind-friendly palette (Okabe-Ito scheme)
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion (red-orange)
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
    "#000000",  # Black
]

# Color cycle for automatic plot coloring
COLOR_CYCLE = mpl.cycler(color=COLORBLIND_COLORS)

# Complete style dictionary with color cycle
COMPLETE_STYLE = {
    **PAPER_STYLE,
    "axes.prop_cycle": COLOR_CYCLE,
}


def apply_paper_style(font_scale: float = 1.0):
    """Apply the paper-ready matplotlib style."""
    style = dict(COMPLETE_STYLE)
    if font_scale != 1.0:
        for key in (
            "axes.labelsize",
            "axes.titlesize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
        ):
            style[key] = style[key] * font_scale
    mpl.rcParams.update(style)


def get_color(index: int) -> str:
    """
    Get a colorblind-friendly color by index.

    Args:
        index: Color index (cycles through available colors)

    Returns:
        Hex color string
    """
    return COLORBLIND_COLORS[index % len(COLORBLIND_COLORS)]


