"""
   Series of useful basic functions.
   (Written by Ranjay Krishna)
"""


def union(bbox1, bbox2):
    """Creates the union of the two bboxes.

    Args:
        bbox1: y0, y1, x0, x1 format.
        bbox2: y0, y1, x0, x1 format.

    Returns:
        The union of the arguments.
    """
    y0 = min(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x0 = min(bbox1[2], bbox2[2])
    x1 = max(bbox1[3], bbox2[3])
    return [y0, y1, x0, x1]
