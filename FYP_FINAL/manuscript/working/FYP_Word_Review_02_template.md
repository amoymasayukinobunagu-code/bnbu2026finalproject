# Template execution contract

Reference: E:\Project\FYP_FINAL\templet\[GroupID] FYP Thesis.docx
SHA256: 95d7913d014745ca1ee731174deb04273c007d5dcbb35f94705b22f2e50d33cc
Reference render: manuscript/working/qa/template/reference.pdf (12 pages, Microsoft Word).
Reference has two A4 sections, a cover, abstract, contents, declarations, acknowledgement,
chapter examples, references and formatting appendix. Rendered all pages to PNG contact sheet.
Actual margins around 3.17 cm horizontally and 2.54 cm vertically differ from written rules.
Actual Normal is 10 pt, Heading 1 Arial 14 pt; written rules specify Times New Roman
12 pt body, 16/14 pt headings, 2 cm left/right and 2.5 cm top/bottom margins.

This review uses the written template rules for font sizes and margins, A4 portrait,
black headings, justified body and retained footer parts. Chinese planning text uses SimSun.
Source style IDs Normal, Heading1, Heading2, Heading3 and Caption are reused.
Body rhythm: 1.15 lines, 6 pt after, no first-line indent, widow control.
Explicit breaks separate overview, chapter outlines, sample, and review notes.
The review is intentionally not a full thesis: cover placeholders, declarations, sample
references, figure, TOC and formatting appendix are removed from body content.
Only the final body section is reused; section count becomes one intentionally.
No author, approval, signature, or publication-ready status is inferred.

Editable parts: word/document.xml (all body slots replaced with review content; last sectPr
retained with documented geometry changes and decimal pagination), word/styles.xml (selected
paragraph styles), word/settings.xml (updateFields). Preserve-only: every other OPC part,
including all relationships, media, numbering, custom XML, headers/footers and metadata.
Builder verifies byte equality for every preserve-only part and unchanged reference hash.
The output keeps unused original package assets rather than deleting opaque components.
No TOC or cross-reference cache is introduced; inherited PAGE fields update on Word export.
QA uses native Microsoft Word PDF export, then Poppler PNGs, because packaged render_docx.py
was attempted and failed: soffice.exe is absent. The native Word export succeeded.
Final page and structural audit are recorded in qa/review_02/layout_audit.json.
