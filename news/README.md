# News fragments (conflict-free NEWS.md entries)

`NEWS.md`'s `## Development` section used to grow by every PR editing the same
few lines at the top of that section — which meant two PRs landing close
together flipped each other's GitHub mergeability to CONFLICTING even though
`NEWS.md` is marked `merge=union` (GitHub's mergeability check does not honor
local/custom merge drivers, and `.gitattributes` for this repo lives only in
`.git/info/attributes`, which is never committed anyway). `test/runtests.jl`
had the same problem for a different reason; see its own CONFLICT-FREE
REGISTRATION note.

## Convention

Instead of editing `NEWS.md` directly, add one small Markdown file per PR to
this directory:

```
news/<slug>.md
```

`<slug>` is a short, unique identifier for the PR — the issue number, the
branch name, or a short feature slug (e.g. `827-student-crossed-fixes.md`,
`734-tweedie-aghq.md`). Uniqueness only needs to avoid colliding with a fragment
someone else is adding in the SAME window, so a PR/issue number is the safest
choice.

The file's content is exactly what would have gone into `NEWS.md`'s
`## Development` section: one or more bullets in the existing style (a bold,
short lead sentence, then plain-prose detail). For example:

```markdown
- **Short bold lead sentence (#NNN).** Plain-prose detail: what changed,
  why, and any numbers that back the claim.
```

Do not edit `NEWS.md` itself in the PR. Because every PR's fragment is its own
new file, two PRs adding fragments at the same time never touch the same
lines, so GitHub never reports a conflict on this account.

## Assembling at release time

`tools/assemble_news.jl` folds every fragment in this directory into
`NEWS.md`'s `## Development` section (newest fragment first, i.e. prepended)
and deletes the fragment files it consumed. Run it from the repo root:

```sh
julia tools/assemble_news.jl          # writes NEWS.md, deletes consumed fragments
julia tools/assemble_news.jl --dry-run  # preview only, no files touched
```

This is a release-time step (folded in when cutting a tag / bumping the
version in `NEWS.md`), not something every PR runs.
