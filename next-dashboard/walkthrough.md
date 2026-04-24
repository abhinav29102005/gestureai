# Cloudflare Deployment Fixes Walkthrough

I have resolved the critical blockers preventing the Cloudflare Pages deployment. The main issues were the 25MB asset limit per file and dependency resolution errors in the production build environment.

## Changes Made

### 1. Data Optimization & Splitting
- **Issue**: Several JSON results files exceeded the 25MB Cloudflare Pages asset limit.
- **Solution**: Created a `scripts/split_data.py` script that automatically detects large files and splits them into smaller `_p0`, `_p1` chunks.
- **Results**: All 1.2GB of data is now compliant with Cloudflare's per-file size limits.
- **Manifest**: Generated a `public/data/results/manifest.json` to allow the API to list subjects without expensive directory scans.

### 2. API Route Refactoring
- **`app/api/subjects/route.ts`**: Updated to use the new `manifest.json`. This is faster and works reliably in serverless environments.
- **`app/api/subject/[id]/route.ts`**: Implemented a "chunk stitching" logic. If a subject's data is split across multiple files, the API now automatically finds and concatenates them before serving the downsampled result to the frontend.

### 3. Dependency Management
- **`package.json`**: Verified that all build-essential packages (`@opennextjs/cloudflare`, `wrangler`, `@tailwindcss/postcss`) are in the `dependencies` section. This ensures they are installed during Cloudflare's `npm clean-install` (production) phase.

### 4. Path Configuration
- Moved data files to `public/data/results/` to ensure they are treated as static assets by Cloudflare.
- Updated `RESULTS_DIR` in all API routes to use the correct absolute paths via `process.cwd()`.

## Next Steps for USER

> [!IMPORTANT]
> You must push the latest commit to GitHub to trigger the Cloudflare build.

1. **Push Changes**:
   ```bash
   git push origin main
   ```

2. **Cloudflare Settings**:
   Ensure your Cloudflare Pages project is configured with:
   - **Framework Preset**: `None` (Manual configuration is better for OpenNext)
   - **Build command**: `npm run pages:build`
   - **Output directory**: `.open-next/assets`
   - **Node.js Compatibility Flag**: `nodejs_compat` (Must be enabled in the Cloudflare dashboard under *Settings -> Functions -> Compatibility Flags*).

## Verification Plan

### Automated Tests
- The API routes have been tested to handle both single-file and multi-chunk subjects.
- The `split_data.py` script successfully processed the entire 1.2GB dataset.

### Manual Verification
- Once deployed, navigate to a subject with a large dataset (e.g., Sub 1629) to ensure the timeline and stats load correctly.
