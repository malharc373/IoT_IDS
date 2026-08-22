#!/usr/bin/env python3
"""Render the editable corrected-status Markdown report to a polished PDF."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "PROJECT_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "IOT_IDS_Corrected_Technical_Report.pdf"


def _inline(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def _styles():
    base = getSampleStyleSheet()
    ink = colors.HexColor("#15202B")
    blue = colors.HexColor("#176B87")
    muted = colors.HexColor("#52606D")
    return {
        "cover": ParagraphStyle("Cover", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=28, leading=33, textColor=ink, alignment=TA_LEFT,
                                spaceAfter=12),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica",
                                   fontSize=12, leading=17, textColor=muted, spaceAfter=12),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=20, leading=24, textColor=ink, spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, textColor=blue, spaceBefore=6,
                             spaceAfter=8),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.3, leading=13.3, textColor=ink, alignment=TA_LEFT,
                               spaceAfter=7),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="Helvetica",
                                 fontSize=9.1, leading=12.8, textColor=ink, leftIndent=14,
                                 firstLineIndent=0, bulletIndent=2, spaceAfter=3),
        "quote": ParagraphStyle("Quote", parent=base["BodyText"], fontName="Helvetica-Bold",
                                fontSize=9.3, leading=13.5, textColor=colors.HexColor("#7A3E00"),
                                backColor=colors.HexColor("#FFF4E5"), borderColor=colors.HexColor("#F2B24B"),
                                borderWidth=0.8, borderPadding=8, spaceBefore=5, spaceAfter=10),
        "code": ParagraphStyle("Code", fontName="Courier", fontSize=7.7, leading=10.2,
                               textColor=colors.HexColor("#EAF2F8"), backColor=colors.HexColor("#15202B"),
                               borderPadding=8, spaceBefore=4, spaceAfter=10),
        "table": ParagraphStyle("Table", fontName="Helvetica", fontSize=7.5, leading=9.4,
                                textColor=ink),
        "table_head": ParagraphStyle("TableHead", fontName="Helvetica-Bold", fontSize=7.5,
                                     leading=9.4, textColor=colors.white),
    }


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8DEE4"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#66788A"))
    canvas.drawString(18 * mm, 10.5 * mm, "IoT-IDS corrected technical status edition - 22 August 2026")
    canvas.drawRightString(192 * mm, 10.5 * mm, str(doc.page))
    canvas.restoreState()


def _table(rows, styles):
    data = []
    for ridx, row in enumerate(rows):
        style = styles["table_head"] if ridx == 0 else styles["table"]
        data.append([Paragraph(_inline(cell), style) for cell in row])
    widths = [174 * mm / len(data[0])] * len(data[0])
    if len(data[0]) >= 3:
        widths[0] *= 1.25
        remaining = 174 * mm - widths[0]
        widths[1:] = [remaining / (len(widths) - 1)] * (len(widths) - 1)
    tbl = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F7F9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5DF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def parse_markdown(text: str, styles):
    lines = text.splitlines()
    story = []
    para = []
    bullets = []
    in_code = False
    code = []
    first_heading = True
    i = 0

    def flush_para():
        nonlocal para
        if para:
            story.append(Paragraph(_inline(" ".join(x.strip() for x in para)), styles["body"]))
            para = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.extend(Paragraph(_inline(x), styles["bullet"], bulletText="-") for x in bullets)
            story.append(Spacer(1, 4))
            bullets = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_para(); flush_bullets()
            if in_code:
                story.append(Preformatted("\n".join(code), styles["code"]))
                code = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if line.startswith("|"):
            flush_para(); flush_bullets()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story.append(_table(rows, styles))
            story.append(Spacer(1, 9))
            continue
        if line.startswith("# "):
            flush_para(); flush_bullets()
            if first_heading:
                story.append(Spacer(1, 44 * mm))
                story.append(Paragraph(_inline(line[2:]), styles["cover"]))
                first_heading = False
            else:
                story.append(PageBreak())
                story.append(Paragraph(_inline(line[2:]), styles["h1"]))
        elif line.startswith("## "):
            flush_para(); flush_bullets()
            if line[3:].strip() == "Conclusion":
                story.append(PageBreak())
            story.append(Paragraph(_inline(line[3:]), styles["h2"]))
        elif line.startswith("> "):
            flush_para(); flush_bullets()
            quote = [line[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("> "):
                quote.append(lines[i][2:]); i += 1
            story.append(Paragraph(_inline(" ".join(quote)), styles["quote"]))
            continue
        elif line.startswith("- "):
            flush_para(); bullets.append(line[2:])
        elif not line.strip():
            flush_para(); flush_bullets()
        else:
            if bullets: flush_bullets()
            para.append(line)
        i += 1
    flush_para(); flush_bullets()
    return story


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, title="IoT-IDS Corrected Technical Report",
                          author="Mahi Patel, Malhar Falke, Yugandhar Pise, Vaibhav Tayade",
                          leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_footer)])
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), styles)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
