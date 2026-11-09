"""Verify the early-figure layout while keeping the main paper within 15 pages."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader, PdfWriter
from verify_tnnls_graphics import bounds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/tnnls_layout_20261027"
FIG = ROOT / "experiments/figures/paper"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(stem):
    text = (ROOT/"paper"/(stem+".aux")).read_text(encoding="utf-8")
    return {key:{"number":number,"page":int(page)} for key,number,page in
            re.findall(r"\\newlabel\{(fig:[^}]+)\}\{\{([^{}]+)\}\{(\d+)\}",text)}


def main():
    old = json.loads((OUT/"before/experiments/figures/paper/figure_data.json").read_text())
    new = json.loads((FIG/"figure_data.json").read_text())
    assert all(new[k]==v for k,v in old.items())
    earlier = json.loads((ROOT/"output/tnnls_graphics_20261027/verification.json").read_text())
    assert all(sha(FIG/(k+".pdf"))==v["sha256"] for k,v in earlier["figures"].items())
    manifest = json.loads((OUT/"redraw_manifest.json").read_text())
    assert all(sha(ROOT/"experiments"/k)==v for k,v in manifest["input_sha256"].items())
    tables = 0
    for path in (OUT/"before/paper/sections").rglob("*.tex"):
        current = ROOT/path.relative_to(OUT/"before")
        pattern = r"\\begin\{tabular\}.*?\\end\{tabular\}"
        a = re.findall(pattern,path.read_text(encoding="utf-8"),re.S)
        b = re.findall(pattern,current.read_text(encoding="utf-8"),re.S)
        assert a==b, str(current)
        tables += len(a)
    assert (ROOT/"paper/tnnls_preamble.tex").read_bytes()==(OUT/"before/paper/tnnls_preamble.tex").read_bytes()
    docs = {}
    for stem in ("tnnls_main","tnnls_supp"):
        path = ROOT/"paper"/(stem+".pdf")
        log = (ROOT/"paper"/(stem+".log")).read_text(errors="replace")
        assert not re.search(r"^!|Overfull|undefined",log,re.M)
        assert not bounds(path), stem
        docs[stem] = {"pages":len(PdfReader(path).pages),"figures":labels(stem),"sha256":sha(path)}
    assert docs["tnnls_main"]["pages"]<=15
    assert docs["tnnls_main"]["figures"]["fig:overview"]["page"]<=3
    assert docs["tnnls_main"]["figures"]["fig:matched-protocol"]["page"]<=5
    assert "fig:phase" not in docs["tnnls_main"]["figures"]
    assert "fig:phase" in docs["tnnls_supp"]["figures"]
    names = ["fig_study_overview","fig_matched_protocol","fig1_bounded_effect","fig2_posthoc_correction",
        "fig3_quantile_scale","fig5_lambda_curves","fig_diagnostic_sequence","fig4_theorem3_conditions",
        "fig6_structure_map","fig_phase_full","fig_phase_diagram"]
    writer = PdfWriter()
    for name in names:
        path = FIG/(name+".pdf")
        reader = PdfReader(path)
        assert len(reader.pages)==1 and not list(reader.pages[0].images)
        fonts = subprocess.check_output(["pdffonts",str(path)],text=True)
        assert "Type 3" not in fonts
        embedded = re.findall(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+",fonts)
        assert embedded and all(row[0]=="yes" for row in embedded)
        assert not bounds(path), name
        writer.add_page(reader.pages[0])
    writer.add_metadata({"/Title":"TNNLS figures - early-page layout","/Author":""})
    atlas = OUT/"TNNLS_figures_layout.pdf"
    with atlas.open("wb") as f:writer.write(f)
    report = {"main_page_limit":15,"page_limit_met":True,"new_figures_are_schematic":True,
        "previous_nine_figure_PDFs_unchanged":True,"all_previous_figure_statistics_unchanged":True,
        "plot_inputs_unchanged":len(manifest["input_sha256"]),"section_table_blocks_unchanged":tables,
        "body_font_margins_and_table_style_unchanged":True,"undefined_references":False,"overfull":False,
        "all_eleven_figures_vector_with_embedded_fonts":True,"documents":docs,"atlas_sha256":sha(atlas)}
    (OUT/"layout_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
