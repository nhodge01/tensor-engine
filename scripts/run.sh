#!/bin/bash

echo "=== ZERO-COPY DUAL-GPU PIPELINE ==="
echo "🚀 4M+ RPS DuckLake → 8 Tokenizers → Dual RTX 4090s"
echo ""

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Run the beast!
echo "LAUNCHING THE BEAST!"
echo "===================="
./build/embed_test