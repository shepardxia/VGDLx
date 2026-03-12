def in_bounds(cell_pos, height, width):
    """Check which sprites have cell positions within the grid. Returns [max_n] bool.

    Args:
        cell_pos: [max_n, 2] int32 cell coordinates (from pixel_pos // block_size)
    """
    return (
        (cell_pos[:, 0] >= 0) & (cell_pos[:, 0] < height) &
        (cell_pos[:, 1] >= 0) & (cell_pos[:, 1] < width)
    )


def detect_eos(positions, alive, height, width, block_size):
    """Returns [max_n] bool — which alive sprites are out of bounds.

    Args:
        positions: [max_n, 2] int32 pixel coordinates
        height, width: grid dimensions in cells
        block_size: pixels per cell
    """
    h_px = height * block_size
    w_px = width * block_size
    oob = ((positions[:, 0] < 0) | (positions[:, 0] >= h_px) |
           (positions[:, 1] < 0) | (positions[:, 1] >= w_px))
    return oob & alive
