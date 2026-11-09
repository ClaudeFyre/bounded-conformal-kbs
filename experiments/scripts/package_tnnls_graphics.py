"""Package only the inputs needed to reproduce the eleven paper figures."""
import hashlib
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/tnnls_layout_20261027"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, default=ROOT/"submission_tnnls")
    args = parser.parse_args()
    manifest = json.loads((OUT/"redraw_manifest.json").read_text(encoding="utf-8"))
    names = {(Path("experiments")/k).as_posix() for k in manifest["input_sha256"]}
    names.update({"experiments/scripts/make_figures.py","experiments/scripts/redraw_tnnls_figures.py","experiments/scripts/tnnls_early_figures.py",
                  "experiments/figures/paper/figure_data.json"})
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"experiments/figures/paper").iterdir()
                 if p.suffix in {".pdf",".svg",".png"})
    destination = OUT / "TNNLS_graphics_sources.zip"
    with zipfile.ZipFile(args.submission_dir/"03_code_and_results.zip") as source:
        assert names.issubset(set(source.namelist()))
        with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as target:
            for name in sorted(names):target.writestr(name,source.read(name))
            target.writestr("README.md", """# TNNLS revised figures (2026-10-27)

Contains eleven vector PDFs, editable SVGs, 400 dpi PNGs, the renderer, the frozen
figure-coordinate manifest and the 260 existing result records needed to redraw.
No raw data, model checkpoints, new training or new prediction is required.
The input copies use the anonymized paths from the submission code archive.

Install NumPy and Matplotlib, then run from the extracted root:

    python experiments/scripts/make_figures.py

Outputs: experiments/figures/paper/ . The renderer checks that all seven original
figure-statistic groups remain identical, retains every seed, and uses Arial
(with STIX mathematical glyphs). Keep Arial available for the tested typography.
The source-data manifest is required, including its frozen structure coordinates.
""")
    with zipfile.ZipFile(destination) as package:
        assert package.testzip() is None
    (OUT/"graphics_bundle.json").write_text(json.dumps({"members":len(names)+1,
        "plot_input_records":len(manifest["input_sha256"]),"sha256":hashlib.sha256(destination.read_bytes()).hexdigest()},indent=2),encoding="utf-8")
    print(destination)


if __name__=="__main__":main()
