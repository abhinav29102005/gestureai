# Cloudflare Pages Deployment Implementation Plan

## Goal
Successfully deploy the GesturAI Industrial Dashboard to Cloudflare Pages by overcoming asset size limits and dependency resolution issues.

## Proposed Changes

### Data Layer
- **Split Large Assets**: Implement a script to divide JSON files larger than 25MB into chunks.
- **Static Asset Hosting**: Move data files to `public/data/results/` for reliable static serving.
- **Manifest Generation**: Create a `manifest.json` for fast subject indexing without filesystem lookups.

### API Layer
- **Chunk Reassembly**: Update API routes to detect and stitch together split data chunks at runtime.
- **Path Resolution**: Correct `process.cwd()` usage to ensure compatibility with the Cloudflare Worker environment.

### Build Configuration
- **Dependency Migration**: Move deployment tools (`wrangler`, `@opennextjs/cloudflare`) to `dependencies` in `package.json`.
- **Wrangler Config**: Optimize `wrangler.toml` with `nodejs_compat` and correct build output directories.

## Verification
- [x] Verify data splitting script handles 1.2GB dataset.
- [x] Test API routes with both single and multi-chunk files.
- [x] Confirm `manifest.json` correctly lists all subjects.
