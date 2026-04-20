# Examples

This folder contains example usage of auto-raster2svg.

## Quick Start

```bash
# Single image
python -m src.cli examples/input.png examples/output.svg

# Batch folder
python -m src.cli ./examples/input/ ./examples/output/ --workers 4

# Stroke-only (outline) mode
python -m src.cli examples/input.png examples/output.svg --stroke

# Industrial mode with quality retry
python -m src.cli examples/input.png examples/output.svg --industrial
```

## Expected Output

Each output SVG contains one `<g>` layer per colour region, ordered from
largest area (bottom) to smallest area (top). The layers are directly editable
in Adobe Illustrator — just drag the SVG file into AI.
