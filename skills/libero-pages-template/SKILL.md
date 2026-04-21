---
name: libero-pages-template
description: Use when editing the LIBERO-ROBOT GitHub Pages site in docs/, especially docs/index.html, docs/styles.css, or docs/assets/videos. Enforces the clean case-study template with overview, separate single-task and multi-task sections, standalone thinking-process blocks, statistics, and before/after videos.
---

# LIBERO Pages Template

When editing the GitHub Pages site for this repo, follow `docs/PAGES_TEMPLATE.md`.

## Required Workflow

1. Read `docs/PAGES_TEMPLATE.md` before changing `docs/index.html` or `docs/styles.css`.
2. Preserve the clean case-study structure:
   - Centered nav and intro.
   - Overview of dataset, task family, and environment.
   - One large `Single-task policy` section.
   - One large `Multi-task policy` section.
3. In both task sections, keep the thinking process as a standalone block before statistics and videos.
4. Keep statistics visually separate from the thinking process.
5. Keep before/after videos in matching two-column comparison cards on desktop.
6. Verify that all labels truthfully describe the video/checkpoint being shown.

## Guardrails

- Do not revert to a timeline-heavy or marketing-style page.
- Do not collapse the page into a minimal summary.
- Do not merge single-task and multi-task reasoning into one generic process.
- Do not introduce extra frontend complexity unless the user explicitly asks for it.
- Do not publish low-quality video if a native high-resolution render can be generated.

## Validation

Before finishing:

```bash
python3 -m html.parser docs/index.html
```

If videos changed, verify dimensions with `imageio` in the `lerobot` conda environment and prefer `1024x1024` or better for displayed rollout clips.
