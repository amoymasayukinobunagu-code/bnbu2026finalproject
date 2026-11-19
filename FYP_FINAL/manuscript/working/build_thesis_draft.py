"""Render the editable thesis work in progress from chapter source files."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import argparse
import re

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

MANUSCRIPT = Path(__file__).resolve().parents[1]
BASE = MANUSCRIPT/'drafts/FYP_Word_Review_02.docx'
DEST = MANUSCRIPT/'drafts/FYP_Thesis_Working_Draft.docx'
SOURCES = [MANUSCRIPT/'working'/name for name in (
    'chapter_01_introduction.md', 'chapter_02_related_work.md', 'chapter_03_methodology.md',
    'chapter_04_implementation.md',
    'chapter_05_results.md',
    'chapter_06_conclusion.md',
)]
ADDITIONAL_REFERENCES = MANUSCRIPT/'working/references_chapters_01_to_03.md'
APPENDICES = MANUSCRIPT/'working/thesis_appendices.md'
CITATION = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')
FIGURES_WITH_INTRO = {
    'figures/implementation/lora_daisy_trial/lora_samed_medsam_architecture.png',
    'figures/implementation/msa_daisy_trial/msa_architecture_overview.png',
    'figures/implementation/ppsam_daisy_trial/ppsam_architecture.png',
}
TABLE_WIDTHS = {
    '3.3': (3.45, 2.8, 6.25, 3.8),
    '1': (4.5, 5.9, 5.9),
    '4.1': (3.5, 6.4, 6.4),
    '5.1': (3.65, 3.3, 1.60, 1.95, 1.65, 2.25, 1.90),
    '5.2': (3.65, 3.3, 1.60, 1.95, 1.65, 2.25, 1.90),
    '5.3': (4.5, 5.9, 5.9),
    '5.4': (3.7, 3.15, 3.15, 3.15, 3.15),
    '5.5': (3.2, 1.9, 1.7, 2.9, 3.5, 3.1),
    '5.6': (3.3, 6.5, 6.5),
    'A.1': (4.5, 5.9, 5.9),
    'B.1': (6.3, 5.0, 5.0),
}


def add_table(doc, lines, column_widths=None):
    split = lambda line: [cell.strip() for cell in line.strip().strip('|').split('|')]
    body = [split(lines[0]), *(split(line) for line in lines[2:])]
    if column_widths is None:
        column_widths = (3.6, 12.7) if len(body[0]) == 2 else (4.0, 2.1, 10.2)
    widths = [Cm(width) for width in column_widths]
    row_spacing = Pt(2 if body[0][-1] == 'HD95 (px)' else 3)
    table = doc.add_table(rows=0, cols=len(widths))
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = width
    for i, values in enumerate(body):
        row = table.add_row()
        assert len(values) == len(widths)
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement('w:cantSplit'))
        if i == 0:
            row_properties.append(OxmlElement('w:tblHeader'))
        for cell, width, value in zip(row.cells, widths, values):
            cell.width = width
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            form = para.paragraph_format
            form.left_indent = Pt(0)
            form.right_indent = Pt(0)
            form.space_before = row_spacing
            form.space_after = row_spacing
            form.line_spacing = 1.07
            form.keep_with_next = i < len(body) - 1
            run = para.add_run(value)
            run.bold = i == 0
            run.font.size = Pt(11)
            if i == 0:
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'E7EDF3')
                cell._tc.get_or_add_tcPr().append(shade)
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'bottom', 'left', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:'+side)
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:color'), 'D9D9D9')
        borders.append(border)
    table._tbl.tblPr.append(borders)


def add_front_matter(doc):
    cover = (
        ("Promptable Medical Image Segmentation Foundation Models for Endoscopic Polyp Segmentation", 38, 18),
        ("by", 0, 12),
        ("[Student name(s) and student ID(s)]", 0, 27),
        ("A Final Year Project Thesis\nsubmitted in partial fulfillment of the requirements\nfor the degree of", 0, 18),
        ("Bachelor of Science\nin\nData Science\nat", 0, 18),
        ("Beijing Normal-Hong Kong Baptist University", 0, 25),
        ("[Month, Year]", 0, 0),
    )
    for i, (line, before, after) in enumerate(cover):
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        if i == 0:
            p.runs[0].font.size = Pt(18)
            p.runs[0].bold = True

    def page_heading(title, include_in_contents=False):
        p = doc.add_paragraph(title, style="Heading 1" if include_in_contents else "Normal")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.page_break_before = True
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(20)
        p.runs[0].font.size = Pt(16)
        p.runs[0].bold = True

    page_heading("ABSTRACT", include_in_contents=True)
    for paragraph in (MANUSCRIPT/'working/thesis_abstract.md').read_text(encoding='utf-8').strip().split('\n\n'):
        doc.add_paragraph(paragraph)
    page_heading("Contents")
    text_width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    for level in range(1, 4):
        style_id = f'TOC{level}'
        style = next((item for item in doc.styles if item.style_id == style_id), None)
        if style is None:
            style = doc.styles.add_style(f'TOC {level}', WD_STYLE_TYPE.PARAGRAPH, builtin=True)
        style.base_style = doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(11)
        style.font.bold = level == 1
        form = style.paragraph_format
        form.alignment = WD_ALIGN_PARAGRAPH.LEFT
        form.left_indent = Cm(0.55 * (level - 1))
        form.right_indent = Pt(0)
        form.first_line_indent = Pt(0)
        form.space_before = Pt(5 if level == 1 else 0)
        form.space_after = Pt(5)
        form.line_spacing = 1.0
        form.keep_with_next = level == 1
        form.keep_together = True
        form.tab_stops.clear_all()
        form.tab_stops.add_tab_stop(text_width, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    toc = doc.add_paragraph()
    field_start = OxmlElement('w:fldChar')
    field_start.set(qn('w:fldCharType'), 'begin')
    field_start.set(qn('w:dirty'), 'true')
    toc.add_run()._r.append(field_start)
    field_code = OxmlElement('w:instrText')
    field_code.set(qn('xml:space'), 'preserve')
    field_code.text = r' TOC \o "1-3" \h \z '
    toc.add_run()._r.append(field_code)
    for field_type in ('separate', 'end'):
        node = OxmlElement('w:fldChar')
        node.set(qn('w:fldCharType'), field_type)
        toc.add_run()._r.append(node)
        if field_type == 'separate':
            toc.add_run('Update the table of contents in Microsoft Word.')
    page_heading("DECLARATION")
    doc.add_paragraph(
        "I hereby declare that all the work done in this project is of my independent effort. "
        "I also certify that I have never submitted the idea and product of this project for any academic or employment credits."
    )
    for label in ("Signature", "Student Name", "Student ID", "Date"):
        doc.add_paragraph(f"{label}: ___________________________")

    page_heading("Beijing Normal-Hong Kong Baptist University")
    p = doc.add_paragraph("Data Science Program")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        'We hereby recommend that the project submitted by [Student name(s)], entitled '
        '"Promptable Medical Image Segmentation Foundation Models for Endoscopic Polyp Segmentation", '
        'is accepted in partial fulfillment of the requirements for the degree of Bachelor of Science '
        '(Honours) in Data Science Program.'
    )
    doc.add_paragraph("Supervisor: ___________________________")
    doc.add_paragraph("Date: ___________________________")

    page_heading("ACKNOWLEDGEMENT")
    doc.add_paragraph(
        "We thank our supervisor, Dr. Donglong Chen, for his guidance on this final year project. "
        "We also acknowledge the researchers who made the datasets, models and software used in this study publicly available."
    )

    doc.add_section(WD_SECTION_START.NEW_PAGE)
    for section, number_format in zip(doc.sections, ('upperRoman', 'decimal')):
        section.different_first_page_header_footer = False
        props = section._sectPr
        numbering = props.find(qn('w:pgNumType'))
        if numbering is None:
            numbering = OxmlElement('w:pgNumType')
            props.insert_element_before(numbering, 'w:cols', 'w:docGrid')
        numbering.set(qn('w:fmt'), number_format)
        numbering.set(qn('w:start'), '1')


def add_source(doc, content, start_on_new_page=False):
    table_widths = None
    for block in content.strip().split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if block.startswith('|'):
            add_table(doc, block.splitlines(), table_widths)
            table_widths = None
            continue
        if block.startswith('!['):
            image_markup, caption = block.split('\n', 1)
            match = re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)', image_markup)
            if match is None or not caption.startswith('Figure '):
                raise ValueError(f'Invalid figure block: {block}')
            alt, relative = match.groups()
            intro_example = relative == 'figures/introduction/prompted_polyp_task.png'
            if relative in FIGURES_WITH_INTRO:
                intro = doc.paragraphs[-1].paragraph_format
                intro.keep_together = True
                intro.keep_with_next = True
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            form = para.paragraph_format
            form.left_indent = form.right_indent = Pt(0)
            form.keep_with_next = True
            form.keep_together = True
            picture = para.add_run().add_picture(
                str(MANUSCRIPT / relative), width=Cm(15.0 if intro_example else 16.3)
            )
            picture._inline.docPr.set('descr', alt)
            caption_para = doc.add_paragraph(caption, style='Caption')
            caption_para.paragraph_format.left_indent = Pt(0)
            caption_para.paragraph_format.right_indent = Pt(0)
            caption_para.paragraph_format.keep_together = True
            caption_para.paragraph_format.keep_with_next = False
            if intro_example:
                caption_para.paragraph_format.line_spacing = 1.05
            continue
        heading = re.match(r'^(#{1,3}) (.+)$', block)
        if heading:
            para = doc.add_paragraph(heading.group(2), style=f'Heading {len(heading.group(1))}')
            if start_on_new_page:
                para.paragraph_format.page_break_before = True
                start_on_new_page = False
        elif re.match(r'^Table (?:\d+|(?:\d+|[A-Z])\.\d+) ', block):
            if block.startswith('Table 3.3 '):
                intro = doc.paragraphs[-1].paragraph_format
                intro.keep_together = True
                intro.keep_with_next = True
            para = doc.add_paragraph(block, style='Caption')
            para.paragraph_format.keep_with_next = True
            table_widths = TABLE_WIDTHS.get(block.split()[1])
        elif re.match(r'^\[\d+\]', block):
            para = doc.add_paragraph(block)
            para.runs[0].font.size = Pt(11)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.paragraph_format.keep_together = True
            para.paragraph_format.line_spacing = 1.0
            para.paragraph_format.space_after = Pt(4)
        elif block.startswith('Surface HD95 sample counts.'):
            for cell in doc.tables[-1].rows[-1].cells:
                cell.paragraphs[0].paragraph_format.keep_with_next = True
            para = doc.add_paragraph(block, style='Caption')
            para.paragraph_format.keep_together = True
            para.paragraph_format.keep_with_next = False
        else:
            para = doc.add_paragraph()
            for part in re.split(r'\b([xy]_(?:min|max))\b', block):
                coord = re.fullmatch(r'([xy])_(min|max)', part)
                if coord:
                    para.add_run(coord[1]).italic = True
                    para.add_run(coord[2]).font.subscript = True
                else:
                    para.add_run(part)
        para.paragraph_format.left_indent = Pt(0)
        para.paragraph_format.right_indent = Pt(0)
        if re.match(r'^RQ[12]\. ', block):
            para.paragraph_format.keep_with_next = True


def assemble_sources():
    """Number references by first mention across the compiled chapters."""
    bibliography = {}
    ordering = {}
    chapters = []
    for source in SOURCES:
        main, _, local_refs = source.read_text(encoding='utf-8').partition(
            '\n## References cited in this chapter draft\n')
        chapters.append(main)
        for item in re.finditer(r'^\[(\d+)\] (.+)$', local_refs, flags=re.MULTILINE):
            key, entry = item.groups()
            assert key not in bibliography
            bibliography[key] = entry
    for item in re.finditer(r'^\[(\d+)\] (.+)$', ADDITIONAL_REFERENCES.read_text(encoding='utf-8'),
                            flags=re.MULTILINE):
        key, entry = item.groups()
        assert key not in bibliography
        bibliography[key] = entry

    def renumber(match):
        numbers = []
        for key in re.split(r',\s*', match.group(1)):
            assert key in bibliography, f'Unmapped citation {key}'
            numbers.append(ordering.setdefault(key, len(ordering) + 1))
        return '[' + ', '.join(str(number) for number in sorted(set(numbers))) + ']'

    chapters = [CITATION.sub(renumber, chapter) for chapter in chapters]
    references = ['# References']
    references.extend(f'[{number}] {bibliography[key]}' for key, number in ordering.items())
    return chapters + ['\n\n'.join(references)]


def empty_document():
    doc = Document(BASE)
    body = doc._element.body
    section = deepcopy(body.sectPr)
    for child in list(body):
        body.remove(child)
    body.append(section)
    return doc


def build_paper():
    doc = empty_document()
    source = MANUSCRIPT/'working/fyp_research_paper.md'
    add_source(doc, source.read_text(encoding='utf-8'))
    destination = MANUSCRIPT/'drafts/FYP_Research_Paper_Draft.docx'
    doc.save(destination)
    print(destination)


def main():
    doc = empty_document()
    add_front_matter(doc)
    for i, content in enumerate(assemble_sources()):
        add_source(doc, content, start_on_new_page=i > 0)
    for appendix in APPENDICES.read_text(encoding='utf-8').strip().split('\n\n<!-- appendix break -->\n\n'):
        add_source(doc, appendix, start_on_new_page=True)
    buffer = BytesIO()
    doc.save(buffer)
    with ZipFile(BASE) as original, ZipFile(buffer) as fresh, ZipFile(DEST, 'w') as output:
        changed = {'[Content_Types].xml', 'word/document.xml', 'word/_rels/document.xml.rels',
                   'word/styles.xml'}
        for member in original.infolist():
            parts = fresh if member.filename in changed else original
            output.writestr(member, parts.read(member.filename))
        for member in fresh.infolist():
            if member.filename.startswith('word/media/') and member.filename not in original.namelist():
                output.writestr(member, fresh.read(member.filename))
    print(DEST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', action='store_true', help='Build the separate research-paper draft.')
    arguments = parser.parse_args()
    if arguments.paper:
        build_paper()
    else:
        main()
