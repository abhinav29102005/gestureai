# Render.com Deployment Walkthrough

We have successfully pivoted the deployment strategy from Cloudflare Pages to **Render.com**. This move simplifies the infrastructure and ensures that the large industrial dataset (1.2GB) is handled reliably without complex chunking or adapter logic.

## Changes Made

### 1. Render.com Integration
- **`render.yaml`**: Added a configuration blueprint for Render. This specifies:
  - **Build Command**: `npm install && npm run build`
  - **Start Command**: `npm run start` (Starts the standard Next.js production server).
  - **Environment**: Node.js production mode.

### 2. Configuration Cleanup
- **`next.config.ts`**: Removed Cloudflare-specific experimental flags and comments. The app now uses the standard Next.js build pipeline, which is fully compatible with Render.

### 3. Data Stability
- The data remains in `public/data/results/`. On Render, the `fs` (FileSystem) API works natively and consistently, ensuring that your 51 subjects and global statistics load without issue.

## Next Steps for USER

> [!IMPORTANT]
> To deploy, follow these steps:

1. **Push the changes**:
   ```bash
   git add .
   git commit -m "chore: prepare for Render.com deployment"
   git push origin main
   ```

2. **Connect to Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository.
   - Render will automatically detect the `render.yaml` and configure the build.
   - **Note**: For 1.2GB of static data, we recommend the **Starter** plan (or higher) to ensure smooth build performance and sufficient disk caching.

## Verification Plan
- Once the build is complete on Render, visit the provided `.onrender.com` URL.
- Verify that the "Global Intelligence" dashboard loads correctly (calling `/api/global-stats`).
- Select a subject to verify that the detailed analysis (calling `/api/subject/[id]`) is functional.
