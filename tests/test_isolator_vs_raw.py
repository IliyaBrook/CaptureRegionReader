"""Test: does text_isolator damage the image for Tesseract?

Compares OCR results with and without text isolation to see if
the binarization/cropping step is causing garbling.
"""
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from capture_region_reader.text_isolator import isolate_text
from capture_region_reader.ocr_worker import _upscale


def create_text_image_rgb(text: str, width: int = 1200, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_size: int = 32) -> np.ndarray:
    """Create text image as RGB numpy array (simulating a screen capture)."""
    img = Image.new("RGB", (width, 120), color=bg_color)
    draw = ImageDraw.Draw(img)
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((20, 30), text, fill=text_color, font=font)
    return np.array(img)


def test_pipeline(text: str, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Compare raw vs isolated OCR on the same text."""
    print(f"\n{'='*80}")
    print(f"INPUT: {text}")
    print(f"BG={bg_color}, TEXT={text_color}")
    print(f"{'='*80}")

    rgb = create_text_image_rgb(text, bg_color=bg_color, text_color=text_color)

    # Method 1: Run Tesseract directly on the RGB image (no isolation)
    pil_raw = Image.fromarray(rgb)
    raw_result = pytesseract.image_to_string(
        pil_raw, lang="eng+rus", config="--psm 6 --oem 1"
    ).strip()
    print(f"  RAW (no isolation):  {raw_result}")

    # Method 2: Run through text_isolator + upscale (our pipeline)
    isolated = isolate_text(rgb)
    if isolated is not None:
        pil_isolated = _upscale(Image.fromarray(isolated))
        iso_result = pytesseract.image_to_string(
            pil_isolated, lang="eng+rus", config="--psm 6 --oem 1"
        ).strip()
        print(f"  ISOLATED (pipeline): {iso_result}")

        # Save for visual inspection
        pil_isolated.save(f"/tmp/crr_isolated_test.png")
    else:
        print(f"  ISOLATED: text_isolator returned None (no text found)")

    # Method 3: Just upscale the raw image (no binarization)
    pil_upscaled = _upscale(pil_raw)
    up_result = pytesseract.image_to_string(
        pil_upscaled, lang="eng+rus", config="--psm 6 --oem 1"
    ).strip()
    print(f"  UPSCALED (no bin):   {up_result}")


if __name__ == "__main__":
    # Test 1: White text on black background (typical subtitle)
    test_pipeline(
        "Сегодня мы будем использовать новый framework для разработки",
        bg_color=(0, 0, 0), text_color=(255, 255, 255)
    )

    # Test 2: Same but with Disney Plus
    test_pipeline(
        "Доступны на Disney Plus каждую ночь",
        bg_color=(0, 0, 0), text_color=(255, 255, 255)
    )

    # Test 3: White text on dark gray (semi-dark background)
    test_pipeline(
        "Завтра нам нужно провести важный meeting",
        bg_color=(30, 30, 30), text_color=(220, 220, 220)
    )

    # Test 4: Complex mixed text
    test_pipeline(
        "Вчера я купил отличный laptop с высокой performance",
        bg_color=(0, 0, 0), text_color=(255, 255, 255)
    )
