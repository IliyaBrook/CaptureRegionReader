"""Test Tesseract directly with bilingual text to isolate the garbling problem.

Creates a clean white image with known bilingual text, runs Tesseract on it
with different lang configurations, and shows what comes back.
"""
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import sys


def create_text_image(text: str, width: int = 1200, font_size: int = 32) -> Image.Image:
    """Create a clean black-text-on-white image for OCR testing."""
    # Create image with plenty of space
    img = Image.new("RGB", (width, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use a good font, fall back to default
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        font = ImageFont.load_default()

    # Draw text
    draw.text((20, 30), text, fill=(0, 0, 0), font=font)
    return img


def test_single_line(text: str, langs: list[str]):
    """Test a single line of text with multiple lang configs."""
    img = create_text_image(text)
    print(f"\n{'='*80}")
    print(f"INPUT:  {text}")
    print(f"{'='*80}")

    for lang in langs:
        try:
            result = pytesseract.image_to_string(
                img, lang=lang, config="--psm 6 --oem 1"
            ).strip()
            # Show garbling
            garbled = any(
                c in result for c in "\\|{}$"
            )
            status = "GARBLED!" if garbled else "OK"
            print(f"  lang={lang:10s} → {result}  [{status}]")
        except Exception as e:
            print(f"  lang={lang:10s} → ERROR: {e}")


def main():
    print("Tesseract version:", pytesseract.get_tesseract_version())
    print()

    test_lines = [
        # Pure Russian
        "Сегодня мы будем использовать новый подход для разработки",
        # Pure English
        "Today we will use a new framework for development",
        # Mixed: Russian with English words
        "Сегодня мы будем использовать новый framework для разработки",
        "Вчера я купил отличный laptop с высокой performance",
        "Завтра нам нужно провести важный meeting чтобы обсудить progress",
        # Brand names
        "Доступны на Disney Plus каждую ночь",
        "Скачай приложение Netflix или YouTube",
    ]

    langs = ["eng+rus", "rus", "eng", "rus+eng"]

    for text in test_lines:
        test_single_line(text, langs)

    # Also test: what if we run eng+rus and eng separately and compare?
    print(f"\n{'='*80}")
    print("DUAL-PASS COMPARISON (eng+rus vs eng-only)")
    print(f"{'='*80}")

    mixed_lines = [
        "Сегодня мы будем использовать новый framework для разработки",
        "Доступны на Disney Plus каждую ночь",
    ]
    for text in mixed_lines:
        img = create_text_image(text)
        result_bilingual = pytesseract.image_to_string(
            img, lang="eng+rus", config="--psm 6 --oem 1"
        ).strip()
        result_eng = pytesseract.image_to_string(
            img, lang="eng", config="--psm 6 --oem 1"
        ).strip()
        result_rus = pytesseract.image_to_string(
            img, lang="rus", config="--psm 6 --oem 1"
        ).strip()
        print(f"\n  INPUT:    {text}")
        print(f"  eng+rus:  {result_bilingual}")
        print(f"  eng:      {result_eng}")
        print(f"  rus:      {result_rus}")


if __name__ == "__main__":
    main()
