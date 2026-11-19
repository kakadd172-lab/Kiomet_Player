
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"


def load_screenshot(path: Path):
    """Load an image and convert it to grayscale."""

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compare_screenshots(first_path: Path, second_path: Path):
    """Compare two screenshots and estimate visual changes."""

    first = load_screenshot(first_path)
    second = load_screenshot(second_path)

    if first.shape != second.shape:
        second = cv2.resize(
            second,
            (first.shape[1], first.shape[0])
        )

    difference = cv2.absdiff(first, second)

    # Ignore tiny pixel-level variations.
    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    changed_pixels = np.count_nonzero(threshold)
    total_pixels = threshold.size

    changed_percentage = (
        changed_pixels / total_pixels
    ) * 100

    average_difference = float(np.mean(difference))

    return {
        "changed_percentage": round(changed_percentage, 2),
        "average_difference": round(average_difference, 2),
        "significant_change": changed_percentage > 2.0
    }


def main():
    screenshots = sorted(
        SCREENSHOT_DIR.glob("screen_*.png")
    )

    if len(screenshots) < 2:
        print("[Vision] At least two screenshots are required.")
        return

    first = screenshots[-2]
    second = screenshots[-1]

    print("[Vision] Comparing screenshots...")
    print(f"First:  {first.name}")
    print(f"Second: {second.name}")

    result = compare_screenshots(first, second)

    print("\n--- Visual Analysis ---")
    print(f"Changed pixels: {result['changed_percentage']}%")
    print(f"Average difference: {result['average_difference']}")
    print(f"Significant change: {result['significant_change']}")


if __name__ == "__main__":
    main()
