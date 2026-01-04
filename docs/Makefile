# Makefile for MkDocs documentation

.PHONY: help clean images docs serve

help:
	@echo "Available targets:"
	@echo "  make images  - Generate example images and assets"
	@echo "  make docs    - Build documentation with MkDocs"
	@echo "  make serve   - Serve documentation locally (with live reload)"
	@echo "  make clean   - Clean generated files"

images:
	@echo "Generating example images and assets..."
	python3 generate_examples.py

docs: images
	@echo "Building documentation with MkDocs..."
	mkdocs build

serve: images
	@echo "Serving documentation at http://127.0.0.1:8000"
	@echo "Press Ctrl+C to stop"
	mkdocs serve

clean:
	@echo "Cleaning generated files..."
	rm -rf site/
	rm -rf api/
	rm -rf assets/images/
	rm -f assets/manifest.json
