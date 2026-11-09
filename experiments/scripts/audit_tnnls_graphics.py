"""Contact sheets and basic PDF checks for the 2026-10-27 figure redesign."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/tnnls_graphics_20261027"


def contact(files, path, columns=3, width=380, height=530):
    canvas = Image.new("RGB", (columns * width, ((len(files)+columns-1)//columns) * height), "#e8ebef")
    draw = ImageDraw.Draw(canvas)
    for i, file in enumerate(files):
        tile = ImageOps.contain(Image.open(file).convert("RGB"), (width-14, height-30))
        x, y = (i % columns) * width, (i // columns) * height
        canvas.paste(tile, (x+(width-tile.width)//2, y+25))
        draw.text((x+8, y+7), file.stem, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["before", "after"])
    args = parser.parse_args()
    directory = OUT / f"qa_{args.phase}"
    for name in ("main", "supp"):
        files = sorted(directory.glob(f"{name}-*.png"))
        contact(files, directory / f"{name}_contact.png")
    graphics = ROOT / "experiments/figures/paper" if args.phase == "after" else OUT / "before/experiments/figures/paper"
    contact(sorted(graphics.glob("*.png")), directory / "figures_contact.png", columns=2, width=650, height=360)
    results = {}
    for path in sorted(graphics.glob("*.pdf")):
        proc = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True, check=True)
        results[path.name] = {"font_inventory": proc.stdout, "type3": "Type 3" in proc.stdout}
    (directory / "font_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{args.phase}: sheets generated, {len(results)} vector figure font inventories saved")


if __name__ == "__main__":
    main()
