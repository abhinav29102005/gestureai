# Cloudflare Deployment Task List

- [x] Create `scripts/split_data.py` to handle large JSON files
- [x] Run splitting script on 1.2GB dataset
- [x] Generate `public/data/results/manifest.json`
- [x] Update `app/api/subjects/route.ts` to use manifest
- [x] Update `app/api/subject/[id]/route.ts` to support chunk stitching
- [x] Move build dependencies from `devDependencies` to `dependencies` in `package.json`
- [x] Configure `wrangler.toml` for OpenNext and `nodejs_compat`
- [x] Commit all changes (including split data and `lib/` directory)
