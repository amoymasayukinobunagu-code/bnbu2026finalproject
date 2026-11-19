# Repository Guidelines

## Project Structure & Module Organization

This repository is a final-thesis writing snapshot for SAM-based polyp segmentation. Read `README.md`, `charter/SCOPE.md`, and `numbers/NUMBERS_INDEX.md` before editing.

- `charter/`: official project requirements and scope.
- `numbers/official_core/`: authoritative main-comparison JSON results; `appendix_evidence/` holds supplementary comparisons, and `ablation_box_expand/` holds box-expansion experiments.
- `splits/`: dataset split lists and notes, without dataset images.
- `manuscript/tables/`: verified CSV and LaTeX exports; `figures/`: destination for new figures; `ref.bib`: references requiring verification.
- `manuscript/template/`: formatting references. `legacy_reference/` contains unverified historical drafts, tables, and figures.
- `SOURCE_MANIFEST.md` and `.csv`: snapshot provenance and SHA256 records.

Training code, weights, and datasets remain in `E:\Project\FYP`; this directory contains no application source or test suite.

## Build, Test, and Development Commands

There is no local build, development server, or automated test command. The June LaTeX root is a formatting reference, not a final build target. Use these PowerShell checks from the repository root:

```powershell
Get-Content numbers/NUMBERS_INDEX.md
# Locate authoritative evidence and protocol notes.
Get-Content numbers/official_core/sam_v3/summary.json -Raw | ConvertFrom-Json
# Inspect a result and check JSON syntax.
Get-FileHash numbers/official_core/sam_v3/summary.json -Algorithm SHA256
# Compare a file against its recorded snapshot hash.
```

## Style & Naming Conventions

Use UTF-8, descriptive Markdown headings, and repository-relative paths in documentation. Preserve existing JSON keys, numeric precision, and two-space indentation. Follow existing names such as `table_main_boxprompt.csv` and `table_main_boxprompt.tex`; keep paired exports consistent. No formatter or linter is configured. Follow `manuscript/template/FORMAT_GUIDE.md` for thesis formatting.

## Validation & Evidence Guidelines

No testing framework or coverage threshold is defined. Validate changed tables against source JSON, including dataset counts, rounding, captions, and protocols. JSON takes precedence over prose. Verify bibliography entries and inspect new figures visually.

Keep prompt-free comparisons separate from box-prompt rankings. Evaluation uses fixed box expansion, normally 30 pixels; ± denotes variation across test images. Treat PP-SAM as an existing baseline. New post hoc size and boundary-proxy analyses are documented in `SUPPLEMENT_STATUS.md`; retain their definitions and limitations. Multi-seed and SOTA claims remain unsupported. Preserve snapshot provenance when changing copied evidence.

## Commit & Pull Request Guidelines

This checkout has no Git metadata, so historical commit conventions cannot be established. If version-controlled, use concise imperative subjects, for example `docs: clarify box-prompt evaluation`. Describe changed artifacts, cite evidence paths, and report validation. Include rendered previews for table, figure, or layout changes and link relevant issues when available.
