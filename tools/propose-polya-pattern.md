# Cross-package proposal: the Pólya pattern (scout + creative combination)

Two **tailored** issue drafts to file on the sister repos after Phase 0 lands.
Tailored because the repos are at different starting points:

- **GLLVM.jl** has no issues-ledger, no scripted workflows, and no scout → gets
  the **full better-team pattern**.
- **gllvmTMB** already runs an Issues + ROADMAP + after-task ledger → gets only
  the **Pólya scout + creative-combination persona** (the genuinely new part).

File with:

```bash
gh issue create --repo itchyshin/GLLVM.jl   --title "<title 1>" --body-file <(sed -n '/^### GLLVM.jl/,/^### gllvmTMB/p' tools/propose-polya-pattern.md)
gh issue create --repo itchyshin/gllvmTMB   --title "<title 2>" --body-file <(sed -n '/^### gllvmTMB/,$p' tools/propose-polya-pattern.md)
```

(or just copy the relevant section into the GitHub New Issue box).

---

### GLLVM.jl — adopt the DRModels.jl "better-team" pattern

**Title:** Proposal: issues-ledger + scripted workflows + a Pólya scout persona (from DRModels.jl)

DRModels.jl (the distributional-regression sibling) trialled three practices in Phase 0
that might help GLLVM.jl too. Sharing in case they're useful — adopt, adapt, or
decline.

1. **GitHub Issues as the work ledger.** Milestones = phases; one issue → one
   branch → one PR (`closes #NN`). Roadmap issues (pinned, one per milestone)
   carry checklists; article/family slices get promoted to real issues when
   their phase opens, so the tracker stays actionable. Makes "what's done /
   what's next" legible at a glance and survives across sessions.

2. **Scripted workflows** (`.claude/workflows/*.js`) instead of the manual
   Workflow-Q checklist. The engine-quality battery, pre-publish audit, etc.
   become versioned, runnable artifacts. (GLLVM.jl already *describes*
   Workflow Q / W0 / A / … — this just makes them real scripts.)

3. **A "Pólya" persona — scouting + creative combination** (see below). Watches
   the sibling packages + literature on a cadence and proposes original method
   combinations as `idea` issues. Reciprocal: DRModels.jl's Pólya already watches
   GLLVM.jl; if GLLVM.jl runs one too, the scouting compounds both ways.

Happy to share the DRModels.jl scaffold (AGENTS.md, the workflow scripts, the label
set) if any of this looks worth copying.

---

### gllvmTMB — add a "Pólya" scout + creative-combination persona

**Title:** Proposal: a routine "Pólya" scout + creative-combination persona

gllvmTMB already runs a solid Issues + ROADMAP + after-task ledger, so this is
just the one piece DRModels.jl added that gllvmTMB doesn't have yet: a standing
**scout** persona.

**Pólya** does two things:

1. **Routine scouting** (weekly + at each phase-start): watches drmTMB / DRModels.jl /
   GLLVM.jl capabilities + NEWS and the relevant statistics/ecology literature,
   diffs against the last snapshot, and files one `idea` issue per actionable
   signal so nothing evaporates.

2. **Creative combination:** at each phase-start, proposes a few *original*
   pairings of methods that could open a slice (e.g. a verified fast engine ×
   boundary inference; a model × a structured layer a sibling already has).
   Pólya **proposes; it does not implement** — ideas land as issues.

Reciprocal value: DRModels.jl's Pólya already scouts gllvmTMB. If gllvmTMB runs one
too, each package feeds the others. Low cost (one cadence job + an `idea`
label), and it keeps the family of packages cross-pollinating deliberately
rather than by accident.
