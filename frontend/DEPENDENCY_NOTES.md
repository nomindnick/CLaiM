# Frontend Dependency Notes

## Current Status (2025-05-29)

### Security Vulnerabilities
- **2 moderate severity vulnerabilities** in esbuild (via vite)
- Issue: esbuild <=0.24.2 allows any website to send requests to dev server
- This only affects development environment, not production builds

### Deprecation Warnings
1. **eslint@8.57.1** - ESLint 8.x is deprecated, but upgrading to 9.x requires significant config changes
2. **glob@7.2.3** - Used by internal dependencies
3. **rimraf@3.0.2** - Used by internal dependencies
4. **inflight@1.0.6** - Known to leak memory, used by internal dependencies
5. **@humanwhocodes packages** - ESLint-related, will be resolved when upgrading ESLint

## Recommended Actions

### Immediate (Low Risk)
1. The esbuild vulnerability only affects development servers, not production
2. Current setup is safe for development use

### Future Updates (When Time Permits)
1. **Upgrade to Vite 6.x** - This will fix the esbuild vulnerability but may require:
   - Testing all build processes
   - Updating vite config if needed
   - Ensuring all plugins are compatible

2. **Upgrade to ESLint 9.x** - Major version with breaking changes:
   - New flat config format required
   - Update all ESLint plugins
   - Rewrite .eslintrc to eslint.config.js

3. **Update TypeScript ESLint** - Should be done with ESLint upgrade

## Current Workarounds
- Added `.npmrc` with `legacy-peer-deps=true` to handle peer dependency conflicts
- Set loglevel=error to reduce noise during installs

## Production Safety
- None of these issues affect production builds
- The app is safe to deploy
- All vulnerabilities are in dev dependencies only