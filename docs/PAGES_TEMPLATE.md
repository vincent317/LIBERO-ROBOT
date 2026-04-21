# GitHub Pages Template Contract

Use this contract whenever editing `docs/index.html` or `docs/styles.css`.

## Page Structure

The page must use the clean case-study template:

1. Centered top navigation with exactly these anchors:
   - `Overview`
   - `Single-task`
   - `Multi-task`
2. Centered intro block:
   - Small kicker
   - Main title
   - One concise project framing paragraph
3. Overview section:
   - Dataset
   - Task family
   - Environment
   - Compact setup stats
4. Single-task section:
   - Section goal and best-result score card
   - A standalone `Single-task thinking process` block
   - Statistics cards
   - Before/after video comparison
5. Multi-task section:
   - Section goal and best-mean score card
   - A standalone `Multi-task thinking process` block
   - Statistics cards
   - Per-task behavior table
   - Before/after video comparison
6. Footer links:
   - GitHub repository
   - Single-task log
   - Multi-task log

## Content Rules

- Preserve the user's experiment reasoning as first-class content.
- Keep single-task and multi-task thinking processes separate.
- Do not collapse the page into only metrics and videos.
- Do not add a marketing-style landing page, timeline-heavy layout, or decorative frontend effects.
- Keep the design restrained, readable, and report-like.
- Cards are allowed for repeated data blocks, but avoid nested cards.
- Keep videos large and comparable in size.
- Video labels must be truthful about what was actually rendered.

## Video Rules

- Prefer native high-resolution rollout renders.
- Do not use an upscaled low-resolution clip if a native render can be regenerated.
- If the exact checkpoint for a label is not available, change the label rather than implying a false comparison.
- Before committing, verify displayed clips with `imageio` or equivalent and keep them at least `1024x1024` when feasible.

## CSS Rules

- The top navigation and hero text should be centered on desktop.
- Main content width should stay around `1180px`.
- The two major sections should share the same visual grammar.
- Use responsive one-column stacking below tablet width.
- Avoid gradients, complex animation, and heavy visual ornamentation.
