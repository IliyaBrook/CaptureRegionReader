.PHONY: setup run dev clean check-deps

setup: check-deps
	uv sync

check-deps:
	@which tesseract > /dev/null 2>&1 || (echo "ERROR: tesseract not found. Install with: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus" && exit 1)
	@which ffplay > /dev/null 2>&1 || (echo "ERROR: ffplay not found. Install with: sudo apt install ffmpeg" && exit 1)
	@echo "All system dependencies OK"

run: check-deps
	uv run capture-region-reader

dev: check-deps
	uv run python -m capture_region_reader

clean:
	rm -rf .venv __pycache__ dist build *.egg-info src/capture_region_reader/__pycache__
