import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required by @cloudflare/next-on-pages
  // Tells Next.js to target the edge (Cloudflare Workers) runtime
  // while keeping Node.js compatibility via the wrangler.toml nodejs_compat flag
  experimental: {
    // allows reading bundled assets via the Node fs API on Cloudflare
  },

  images: {
    unoptimized: true, // Cloudflare Pages does not support Next.js Image Optimization
  },
};

export default nextConfig;
