"""Read-only evidence checks plus a vector figure atlas for the TNNLS redraw."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/tnnls_graphics_20261027"
FIG = ROOT / "experiments/figures/paper"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(path):
    p = subprocess.run(["pdftotext", "-bbox", str(path), "-"], capture_output=True, check=True)
    # Some TeX math fonts map delimiters to XML-forbidden control characters.
    # Strip those extraction artifacts only; numeric bounding boxes are intact.
    xml = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", p.stdout.decode("utf-8"))
    tree = ET.fromstring(xml)
    out = []
    for number, page in enumerate((e for e in tree.iter() if e.tag.endswith("}page")), 1):
        width,height = float(page.attrib["width"]),float(page.attrib["height"])
        for word in (e for e in page.iter() if e.tag.endswith("}word")):
            a = {k:float(v) for k,v in word.attrib.items()}
            if a["xMin"] < -.5 or a["yMin"] < -.5 or a["xMax"]>width+.5 or a["yMax"]>height+.5:
                out.append({"page":number,"text":word.text,"bbox":a})
    return out


def main():
    original = json.loads((OUT / "before/experiments/figures/paper/figure_data.json").read_text())
    current = json.loads((FIG / "figure_data.json").read_text())
    assert all(current[k] == v for k,v in original.items())
    redraw = json.loads((OUT / "redraw_manifest.json").read_text())
    assert all(sha(ROOT / "experiments" / k) == v for k,v in redraw["input_sha256"].items())
    unchanged_tables = {}
    for path in (OUT / "before/paper").rglob("*.tex"):
        relative = path.relative_to(OUT / "before")
        old,new = path.read_text(encoding="utf-8"),(ROOT / relative).read_text(encoding="utf-8")
        pattern = r"\\begin\{tabular\}.*?\\end\{tabular\}"
        a,b = re.findall(pattern,old,re.S),re.findall(pattern,new,re.S)
        assert a==b, f"Table cell content changed: {relative}"
        if a:unchanged_tables[str(relative)] = len(a)
    assert (ROOT / "paper/generated/numbers.tex").read_text(encoding="utf-8") == (OUT / "before/paper/generated/numbers.tex").read_text(encoding="utf-8")
    order = ["fig1_bounded_effect","fig_phase_diagram","fig2_posthoc_correction","fig3_quantile_scale",
             "fig5_lambda_curves","fig_diagnostic_sequence","fig4_theorem3_conditions","fig6_structure_map","fig_phase_full"]
    writer = PdfWriter();figures = {}
    for name in order:
        path = FIG / f"{name}.pdf"
        reader = PdfReader(path)
        assert len(reader.pages)==1 and not list(reader.pages[0].images)
        fonts = subprocess.check_output(["pdffonts", str(path)],text=True)
        assert "Type 3" not in fonts
        embedded = re.findall(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+", fonts)
        assert embedded and all(e[0]=="yes" for e in embedded)
        outside = bounds(path)
        assert not outside, f"Figure text outside PDF page: {name}: {outside}"
        figures[name] = {"width_in":float(reader.pages[0].mediabox.width)/72,
            "height_in":float(reader.pages[0].mediabox.height)/72,"all_fonts_embedded":True,
            "type3_fonts":False,"raster_images":0,"text_outside_page":[],"sha256":sha(path)}
        writer.add_page(reader.pages[0])
    writer.add_metadata({"/Title":"TNNLS revised figures - 2026-10-27","/Author":""})
    atlas = OUT / "TNNLS_figures_redrawn.pdf"
    with atlas.open("wb") as stream:writer.write(stream)
    manuscripts = {}
    for stem in ("tnnls_main","tnnls_supp"):
        path = ROOT / "paper" / f"{stem}.pdf"
        log = (ROOT / "paper" / f"{stem}.log").read_text(errors="replace")
        assert not re.search(r"^!|Overfull|undefined", log, re.M)
        outside = bounds(path)
        assert not outside, f"Manuscript text outside page: {stem}: {outside}"
        manuscripts[stem] = {"pages":len(PdfReader(path).pages),"sha256":sha(path),
            "undefined_references_or_citations":False,"overfull_boxes":False,"text_outside_page":[]}
    report = {"date":"2026-10-27","previous_figure_statistic_groups_unchanged":len(original),
        "raw_plot_input_hashes_unchanged":len(redraw["input_sha256"]),"table_blocks_unchanged":unchanged_tables,
        "new_training_or_evaluation":False,"generated_number_macros_unchanged_normalizing_CRLF":True,
        "figures":figures,"manuscripts":manuscripts,
        "atlas_sha256":sha(atlas)}
    (OUT / "verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"figures":len(figures),"unchanged_table_blocks":sum(unchanged_tables.values()),
        "unchanged_input_files":len(redraw["input_sha256"]),"manuscripts":manuscripts},indent=2))


if __name__=="__main__":main()
