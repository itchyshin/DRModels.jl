# DRModels.jl visual identity

The site mark, the palette it implies, and the constraints that produced both.
This is an internal design note: `docs/design/` is outside `docs/src/`, so
nothing here is published. Written 2026-09-05, when the mark was adopted.

## The mark

Two density curves that share one mean and differ in spread, over a dashed
centre line, on a hexagonal badge.

`docs/src/assets/drmodels-mark.svg` is the editable master for the two-curve
vector mark. The approved four-curve raster badge used in the navbar and hero
is separate artwork; it must not be regenerated from this older vector mark.

The mark states the package's subject rather than decorating it. A mean-only
model can only say where a distribution sits. Distributional regression is the
claim that the *rest* of the shape is modelled too, and the smallest honest
picture of that is two curves a mean-only model could not tell apart. The dashed
line is the shared mean; the two dots mark the peaks the curves reach because
their scales differ. A reader who understands the mark has understood the
tagline.

Two constraints made it that and not something else:

- **It must survive 24 px.** DocumenterVitepress renders the navbar logo at
  exactly that size (`vitepress_config.jl:143`), so the silhouette carries the
  identity and detail is a bonus, never the message.
- **It must not encode meaning in colour alone.** The narrow and wide curves
  are distinguishable by shape at every size, with hue as reinforcement.

### The hexagon, and its honest caveat

The badge shape is the Julia-ecosystem convention, which is why it is here. It
is also the shape drmTMB uses. For a package whose whole positioning is *the
Julia twin of drmTMB*, family resemblance is defensible and arguably correct.
It is still a deliberate choice and not an accident, and if DRModels.jl should ever
read as visually independent, the same curves on a rounded square carry the
idea unchanged. That variant was drawn and set aside, not overlooked.

### Rejected alternative

A "convergence mark" — three coloured paths, for mean, spread and association,
meeting a fitted-distribution disc — was proposed alongside this one
(`archive/identity-convergence-mark-20260905`, with two further candidates).
The idea is richer. It fails the first constraint: it carries an embedded
`DRModels.jl` wordmark that renders about 2 px tall at navbar size, and its light
tile becomes a bright rectangle against a dark navbar. Read at a distance it
also suggests a pipeline, which is not what this package is.

Both are kept at that tag if the question is ever reopened.

## Geometry, and a defect worth not repeating

The badge outline is a rounded hexagon drawn as line segments joined by
quadratic corners. The apex is rounded by the final curve, from `240 20` through
control `256 10` to `272 20`.

The first version of this path began at `M256 20`, the top vertex. Its closing
curve ended at `272 20`, so `Z` drew a straight segment back across the apex and
produced a visible flat spur — on both the outer badge and the inner ring. It
survived review because it is invisible until magnified or rendered large.

**The rule: a closed outline must start where its closing curve ends.** Both
paths now begin at `272 20` (and `269 35` for the ring). If you redraw the
badge, check the apex at 512 px before shipping.

## Palette

| Token | Hex | Where it is allowed |
|---|---:|---|
| Navy | `#0b1f3a` | badge ground; dark-mode card surface |
| Teal | `#12a3a3` | the narrow curve; fills, edges, hover |
| Indigo | `#5c7cf0` | the wide curve; badge ring; accents |
| Teal (text) | `#0b7c7c` | light-mode links, buttons, brand token |
| Teal (dark text) | `#4fd4d4` | dark-mode links, buttons, brand token |

### A logo colour is not a text colour

This is the part that is easy to get wrong, so it is measured rather than
asserted. `--vp-c-brand-1` has roughly fifty consumers in the VitePress theme
and colours body links. Contrast against white, where WCAG AA body text needs
4.5:1:

| Colour | On white | Verdict |
|---|---:|---|
| `#12a3a3` mark teal | 3.09:1 | fails as text |
| `#5c7cf0` mark indigo | 3.75:1 | fails as text |
| `#0b7c7c` darkened sibling | 5.01:1 | passes |

Note the mark's teal failed this test *before* it was brightened for small
sizes, at 3.98:1. Brightening did not cause the problem; a saturated mid-tone
teal is simply not a body-text colour. So the mark keeps its own hues for fills
and edges, and text and buttons take a darkened sibling. Dark mode inverts the
relationship: `#4fd4d4` measures 9.57:1 on the VitePress dark ground, and white
on the dark brand button measures 6.04:1.

Graphical elements need only 3:1, which is why the mark's own hues are fine
inside the badge: teal reaches 5.34:1 and indigo 4.40:1 against the navy.

## Files, and how they reach the page

| File | Role |
|---|---|
| `docs/src/assets/drmodels-mark.svg` | editable two-curve vector mark, 512 viewBox |
| `docs/src/assets/drmodels-favicon.svg` | simplified two-curve sibling, 128 viewBox, fewer strokes |
| `docs/src/assets/logo.png` | approved compact badge, navbar |
| `docs/src/assets/drmodels-full-logo.png` | approved full four-curve badge, landing-page hero |
| `docs/src/assets/favicon.ico` | approved compact badge, browser-tab icon |

DocumenterVitepress wires the navbar itself: if `logo.png` exists it is copied
to the published root and the config gets `logo: { src: '/logo.png', width: 24,
height: 24 }`. There is nothing to add to `make.jl`. The hero references
`/drmodels-full-logo.png` from the front matter of `docs/src/index.md`.

The two SVGs are the source of truth for the two-curve vector mark only. The
four-curve raster artwork was adopted without a matching tracked vector source;
do not regenerate it from the SVGs or hand-edit it. A future visual revision
should add an editable source asset before replacing the approved raster.

## Maintenance boundary

**Theme customisation belongs in `docs/src/.vitepress/theme/overrides.css`, and
a file named `theme/style.css` must never be created.** DocumenterVitepress
writes its 322-line default stylesheet only when `theme/style.css` is *absent*
(`vitepress_config.jl:61-63`), so creating that file silently deletes the
default: the only JuliaMono `@font-face`, the only rule that makes display
equations scrollable, the rule keeping MathJax glyphs theme-coloured, and every
rule behind the sidebar drawer toggle. The build still exits 0 and nothing
warns. `overrides.css` is already imported after the theme at
`.vitepress/theme/index.ts:25`.

Keep `overrides.css` to shared tokens and the few documented landing
components. It should not grow into a second layout system.

## Accessibility

Both SVGs carry `<title>` and `<desc>` referenced by `aria-labelledby`. The hero
image has descriptive alt text. Colour reinforces the narrow/wide distinction
but never carries it alone. Contrast is stated above and was measured, not
estimated. Hover transitions are disabled under `prefers-reduced-motion`.

## Provenance

The mark originated in `archive/identity-hex-assets-20260905`, adopted whole;
this lane fixed the apex defect and brightened the curves for small dark-mode
sizes. The rejected alternative and its candidates are at
`archive/identity-convergence-mark-20260905`.
