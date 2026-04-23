import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output bundles everything needed to run on any server
  output: "standalone",

  // Allow serving large data files from the data/ directory
  serverExternalPackages: [],

  // Ensure static files in /public are served correctly
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
