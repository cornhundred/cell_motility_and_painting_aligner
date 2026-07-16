# Building the widget JS bundles

Both widgets ship as **prebuilt, offline** deck.gl bundles in
`src/motility_painting/static/` (`aligner.js`, `scrubber.js`). You do **not**
need to rebuild them to use the package.

## Why we don't `npm install`

This workbench's npm registry access is ~140 s **per package**, and `deck.gl`
pulls hundreds of transitive deps — a normal `npm install` takes hours and dies.
Direct tarball downloads, however, are fast. So instead of installing the full
dependency tree we **vendor deck.gl's self-contained standalone bundle** and bundle
against just that.

## Rebuilding

Prereqs (already vendored in the repo):

- `js/vendor/deckgl.min.cjs` — deck.gl 9.2.8 standalone UMD (`dist.min.js` from the
  npm tarball). **The `.cjs` extension is mandatory**: as `.js`, esbuild aliases the
  UMD namespace to an empty object and every deck.gl symbol becomes `undefined`
  (blank widget, no error). Re-fetch with:
  `curl -sSL https://registry.npmjs.org/deck.gl/-/deck.gl-9.2.8.tgz | tar -xzO package/dist.min.js > js/vendor/deckgl.min.cjs`
- `js/.build-tools/bin/esbuild` — the linux-x64 esbuild binary (run it directly;
  the npm JS shim is broken here). Re-fetch with:
  `curl -sSL https://registry.npmjs.org/@esbuild/linux-x64/-/linux-x64-0.25.12.tgz | tar -xzO package/bin/esbuild > js/.build-tools/bin/esbuild && chmod +x js/.build-tools/bin/esbuild`

Build the scrubber (offline; only needs the vendored bundle):

```bash
cd js
./.build-tools/bin/esbuild scrubber_widget.mjs --bundle --format=esm \
  --outfile=../src/motility_painting/static/scrubber.js \
  --platform=browser --target=es2022 --legal-comments=none --define:define.amd=false
```

Build the dual-view QC widget (motility | stitched Cell Painting) the same way —
it only uses core `BitmapLayer`/`ScatterplotLayer`/`TextLayer`, so it builds
offline against the vendored bundle (no editable-layers, unlike the aligner):

```bash
cd js
./.build-tools/bin/esbuild dual_view_widget.mjs --bundle --format=esm \
  --outfile=../src/motility_painting/static/dual_view.js \
  --platform=browser --target=es2022 --legal-comments=none --define:define.amd=false
```

`scrubber_widget.mjs` imports deck.gl from the vendored file
(`import * as deckgl from './vendor/deckgl.min.cjs'`), so no `node_modules` tree is
required. Verify a rebuild by checking the bundle contains `__commonJS` (`grep -c`
≥ 1) — that confirms the UMD interop captured the deck.gl symbols. `aligner.js` was
prebuilt on a machine with normal npm and committed.

On a machine with working npm, `npm install && npm run build` also works
(`aligner_widget.mjs` imports `deck.gl`/`@deck.gl-community/*` normally).
