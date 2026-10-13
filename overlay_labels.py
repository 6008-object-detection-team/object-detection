"""Opaque, bounded label panels shared by detection and correction views."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
PADDING = 3
FOREGROUND = (255, 255, 255)


def place_label(x, y, width, height, bounds):
    """Return (left, top, width, height) within exclusive image bounds.

    Prefer above the box. At the top edge, move inside instead of collapsing
    the background to zero height. Shift left at the right edge. The layout
    changes only the display label, never the underlying detection box.
    """
    left_edge, top_edge, right_edge, bottom_edge = bounds
    available_width = max(0, right_edge - left_edge)
    available_height = max(0, bottom_edge - top_edge)
    width = min(max(0, width), available_width)
    height = min(max(0, height), available_height)
    left = min(max(x, left_edge), right_edge - width)
    preferred_top = y - height if y - height >= top_edge else y
    top = min(max(preferred_top, top_edge), bottom_edge - height)
    return left, top, width, height


def fit_text(text, max_width):
    """Fit an OpenCV label with ellipsis, without allocating a wider image."""
    def width(value):
        return cv2.getTextSize(value, FONT, FONT_SCALE, THICKNESS)[0][0]

    if max_width <= 0:
        return ""
    if width(text) <= max_width:
        return text
    if width("...") > max_width:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if width(text[:middle] + "...") <= max_width:
            low = middle
        else:
            high = middle - 1
    return text[:low] + "..."


def draw_label(image, text, x, y, accent):
    """Draw white text on a box-colored panel; return its exclusive rectangle."""
    height, width = image.shape[:2]
    if not height or not width:
        return None
    text = fit_text(str(text), width - 2 * PADDING)
    (text_width, text_height), baseline = cv2.getTextSize(text or "Ag", FONT, FONT_SCALE, THICKNESS)
    left, top, panel_width, panel_height = map(int, place_label(
        int(x), int(y), text_width + 2 * PADDING, text_height + baseline + 2 * PADDING,
        (0, 0, width, height)))
    right, bottom = left + panel_width, top + panel_height
    # Drawing text on the panel ROI also clips glyphs on exceptionally tiny inputs.
    panel = image[top:bottom, left:right]
    panel[:] = accent
    cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), accent, 1)
    if text:
        # A dark outline keeps white text readable on light source colors too.
        cv2.putText(panel, text, (PADDING, PADDING + text_height), FONT,
                    FONT_SCALE, (24, 24, 24), THICKNESS + 2, cv2.LINE_AA)
        cv2.putText(panel, text, (PADDING, PADDING + text_height), FONT,
                    FONT_SCALE, FOREGROUND, THICKNESS, cv2.LINE_AA)
    return left, top, right, bottom
