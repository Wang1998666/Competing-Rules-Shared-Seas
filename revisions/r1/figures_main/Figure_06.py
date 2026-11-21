"""Rebuild Figure 6 with PyMuPDF: python Figure_06.py REPRODUCTION.zip"""

import argparse
import csv
import re
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pymupdf as pdf


GROUPS = {"EU": "#424F91", "BRI": "#107B75", "Other": "#B66349"}
PLANS = [
    ("Global", "Global plan", "Global", "#315C6E"),
    ("EU", "EU route objective", "EU route", "#6797A3"),
    ("BRI", "Closed BRI plan", "Closed BRI", "#9CBAC0"),
]
INK = (0.14, 0.21, 0.25)
RULE = (0.81, 0.84, 0.85)


def rgb(value):
    return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))


def text(page, x, baseline, label, size=10.6, bold=False, align="left"):
    face = "tibo" if bold else "tiro"
    width = pdf.get_text_length(label, fontname=face, fontsize=size)
    if align == "center":
        x -= width / 2
    elif align == "right":
        x -= width
    page.insert_text((x, baseline), label, fontsize=size,
                     fontname="BodyBold" if bold else "Body", color=INK)


def panel_data(archive, name, key):
    raw = archive.read(f"EF-Figures-v5/panel_data/{name}.csv").decode("utf-8-sig")
    return {row[key]: row for row in csv.DictReader(StringIO(raw))}


def draw_maps(page, sources, plans):
    # show_pdf_page clips the picture, but some PDF readers still extract every
    # label on the original page. Keep only the text painted inside each crop.
    excerpts = {}

    def copy(source, crop, target):
        clip = pdf.Rect(crop)
        source_page = sources[source][0]
        traces = source_page.get_texttrace()
        visible = tuple(clip.intersects(pdf.Rect(span["bbox"])) for span in traces)
        key = source, visible
        if key not in excerpts:
            snippet = pdf.open()
            snippet.insert_pdf(sources[source], from_page=0, to_page=0)
            xref, = snippet[0].get_contents()
            stream = snippet.xref_stream(xref)
            # These archived Matplotlib pages emit one BT/ET block per text
            # span. Leave the surrounding vector geometry and map marks intact.
            text_blocks = re.compile(rb"^BT\r?\n.*?^ET(?=\r?$)", re.MULTILINE | re.DOTALL)
            if len(text_blocks.findall(stream)) != len(visible):
                raise ValueError(f"Unrecognized text layout in archived map {source}")
            labels = iter(visible)
            snippet.update_stream(xref, text_blocks.sub(lambda block: block[0] if next(labels) else b"", stream))
            excerpts[key] = snippet
        page.show_pdf_page(pdf.Rect(target), excerpts[key], 0, clip=clip)

    # Map crops, registration and legends follow the archived compose.py layout.
    for row, (key, title, _, _) in enumerate(PLANS):
        y = 8 + 129 * row
        text(page, 7, y + 10, f"{'abc'[row]}   {title}", bold=True)
        text(page, 224, y + 10, f"{'def'[row]}   European ports", bold=True)
        if row == 0:
            copy("S4", (12, 27, 336, 216), (2, y + 17, 217, y + 115))
        else:
            dy = 114.22 * row
            copy("M6", (12, 28 + dy, 305, 126 + dy), (2, y + 18, 217, y + 114))
        dx = 190.1 * row
        copy("S6", (12 + dx, 26, 188 + dx, 170), (221, y + 17, 372, y + 114))
        data = plans[key]
        ports, routes = int(data["ports"]), int(data["positive_routes"])
        text(page, 7, y + 123, f"{ports} ports, {routes:,} routes", size=10.3, bold=True)

    copy("M6", (12, 351, 205, 375), (9, 396, 203, 419))
    copy("M6", (323, 351, 499, 375), (225, 396, 399, 419))
    for snippet in excerpts.values():
        snippet.close()


def draw_stats(page, plans, rates):
    text(page, 381, 18, "g   Worldwide abatement (Mt/yr)", bold=True)
    left, length = 438, 102
    for tick in (0, 50, 100):
        x = left + tick * length / 100
        page.draw_line((x, 31), (x, 105), color=RULE, width=0.55)
        text(page, x, 119, str(tick), size=10.1, align="center")
    for row, (key, _, short, tint) in enumerate(PLANS):
        y = 44 + 26 * row
        value = float(plans[key]["annual_Mt"])
        text(page, 381, y + 3.5, short)
        page.draw_rect((left, y - 4.5, left + length * value / 100, y + 4.5),
                       color=None, fill=rgb(tint))
        text(page, 578, y + 3.5, f"{value:.2f}", align="right")

    text(page, 381, 147, "h   Selected ports by group", bold=True)
    centers = (456, 503, 554)
    for (group, tint), x in zip(GROUPS.items(), centers):
        width = pdf.get_text_length(group, fontname="tiro", fontsize=10.6)
        page.draw_circle((x - width / 2 - 4, 165), 2.5, color=None, fill=rgb(tint))
        text(page, x + 3, 168.5, group, align="center")
    page.draw_line((381, 175), (578, 175), color=RULE, width=0.65)
    for row, (key, _, short, _) in enumerate(PLANS):
        y = 192 + 23 * row
        text(page, 381, y, short)
        for group, x in zip(GROUPS, centers):
            text(page, x, y, plans[key][group], size=11, align="center")

    text(page, 381, 276, "i   Global selection rates", bold=True)
    text(page, 454, 299, "Share", align="center")
    text(page, 578, 299, "Selected / feasible", align="right")
    for row, (group, tint) in enumerate(GROUPS.items()):
        y = 317 + 25 * row
        rate = float(rates[group]["rate_pct"])
        selected = int(rates[group]["selected"])
        feasible = int(rates[group]["feasible"])
        text(page, 381, y + 3.5, group)
        page.draw_rect((413, y - 4.5, 413 + 67 * rate / 30, y + 4.5),
                       color=None, fill=rgb(tint))
        text(page, 512, y + 3.5, f"{rate:.1f}%", align="right")
        text(page, 578, y + 3.5, f"{selected}/{feasible}", align="right")


def build(archive_path, output):
    with ZipFile(archive_path) as archive, pdf.open() as document:
        plans = panel_data(archive, "06de_plan_comparison", "plan")
        rates = panel_data(archive, "S04b_selection_rates", "camp")
        sources = {s: pdf.open(stream=archive.read(f"map_layout/map_sources/{s}.pdf"),
                               filetype="pdf") for s in ("M6", "S4", "S6")}
        page = document.new_page(width=584, height=420)
        for alias, face in (("Body", "tiro"), ("BodyBold", "tibo")):
            page.insert_font(fontname=alias, fontbuffer=pdf.Font(fontname=face).buffer)
        draw_maps(page, sources, plans)
        draw_stats(page, plans, rates)
        document.save(output, garbage=4, deflate=True, no_new_id=True)
        for source in sources.values():
            source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Reproduction ZIP containing map_layout")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".pdf"))
    options = parser.parse_args()
    build(options.archive, options.output)
