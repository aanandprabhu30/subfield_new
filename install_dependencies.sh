#!/bin/bash
# Install dependencies for ultra-fast abstract classifier

echo "Installing required Python packages for ultra-fast processing..."

# Core dependencies
pip install aiohttp asyncio

# Performance optimizations (optional but recommended)
echo "Installing optional performance packages..."
pip install uvloop orjson psutil

echo "Installation complete!"
echo "Note: uvloop, orjson, and psutil are optional but highly recommended for maximum performance."