.PHONY: setup run dev debug clean check-deps check-venv

setup: check-deps check-venv
	uv sync

check-deps:
	@which tesseract > /dev/null 2>&1 || (echo "ERROR: tesseract not found. Install with: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus" && exit 1)
	@which ffplay > /dev/null 2>&1 || (echo "ERROR: ffplay not found. Install with: sudo apt install ffmpeg" && exit 1)
	@which ffmpeg > /dev/null 2>&1 || (echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg" && exit 1)
	@echo "All system dependencies OK"

check-venv:
	@if [ -d .venv ]; then \
		if ! uv run python -c "import cv2; cv2.cvtColor" > /dev/null 2>&1; then \
			echo "WARNING: Broken venv detected (cv2 missing or corrupt). Recreating..."; \
			rm -rf .venv; \
		elif ! uv run python -c "import PyQt6" > /dev/null 2>&1; then \
			echo "WARNING: Broken venv detected (PyQt6 missing). Recreating..."; \
			rm -rf .venv; \
		elif ! uv run python -c "import numpy" > /dev/null 2>&1; then \
			echo "WARNING: Broken venv detected (numpy missing). Recreating..."; \
			rm -rf .venv; \
		elif ! uv run python -c "import torchaudio" > /dev/null 2>&1; then \
			echo "WARNING: Broken venv detected (torchaudio missing). Recreating..."; \
			rm -rf .venv; \
		elif ! uv run python -c "import torch; import transformers.pytorch_utils as p; p.isin_mps_friendly=torch.isin if not hasattr(p,'isin_mps_friendly') else p.isin_mps_friendly; from TTS.api import TTS" > /dev/null 2>&1; then \
			echo "WARNING: Broken venv detected (coqui-tts missing). Recreating..."; \
			rm -rf .venv; \
		fi; \
	fi

run: check-deps
	uv run capture-region-reader

dev: check-deps
	uv run python -m capture_region_reader

debug: check-deps
	CRR_DEBUG=1 uv run capture-region-reader

clean:
	rm -rf .venv __pycache__ dist build *.egg-info src/capture_region_reader/__pycache__
