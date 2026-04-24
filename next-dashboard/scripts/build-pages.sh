#!/bin/bash
set -e

# 1. Build the app using OpenNext
echo "Running OpenNext build..."
npx @opennextjs/cloudflare build

# 2. Prepare the output directory for Cloudflare Pages (Advanced Mode)
# We use .open-next/assets as the base because it already contains the static files
echo "Preparing assets directory..."
OUT_DIR=".open-next/assets"

# 3. Copy the worker dependencies into the assets directory
# These are required by worker.js via relative imports
echo "Copying worker dependencies..."
cp -r .open-next/server-functions "$OUT_DIR/"
cp -r .open-next/cloudflare "$OUT_DIR/"
cp -r .open-next/middleware "$OUT_DIR/"
cp -r .open-next/.build "$OUT_DIR/"

# 4. Copy and rename the worker entry point
echo "Creating _worker.js..."
cp .open-next/worker.js "$OUT_DIR/_worker.js"

echo "Build complete. Ready for Cloudflare Pages deployment from $OUT_DIR"
