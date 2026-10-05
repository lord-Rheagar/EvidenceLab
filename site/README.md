# EvidenceLab Landing Page

A static Vite site with a labeled, interactive CLI preview. It makes no model or service calls.
Node.js 22.12+ and npm are needed only for website development.

## Development

From the repository root:

```shell
cd site
npm ci
npm run dev
```

Vite prints the URL and chooses another port if its default is occupied.

## Build and Browser Tests

```shell
npm run build
npx playwright install chromium
npm test
```

The build outputs static assets in `site/dist/`. The base is `/EvidenceLab/` for GitHub Pages.
For a domain root, use `npx vite build --base=/` instead. `npm run preview` previews the build.

## Deploy

`.github/workflows/pages.yml` builds and deploys the site on pushes to `main`.
Select **GitHub Actions** in repository Settings > Pages. The target URL is:

https://lord-rheagar.github.io/EvidenceLab/

For a fork or repository rename, update links and canonical metadata in `index.html`,
the base in `package.json`, and the URLs in the root README.

Tabs support arrow keys and Home/End; animation respects reduced motion. Copy buttons
show success and failure states and include a Clipboard API fallback. Browser tests
cover responsive layout, navigation, commands, provider selection, and JavaScript errors.
