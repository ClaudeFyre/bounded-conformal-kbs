"""Paper audits (run after make_numbers.py):
  1. claim tags: every Results paragraph carries '% [Cx ...]'; the macros it uses must belong to the tagged claims;
  2. evidence roles: confirmatory, developmental and exploratory wording must not cross levels within a paragraph.

    python scripts/audit_paper.py      -> exit code 1 on any violation
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PAPER = Path(__file__).resolve().parents[2] / "paper"
RESULTS = ("results_bounded.tex", "results_posthoc.tex", "results_atoms.tex", "results_era5.tex")
CONF = re.compile(r"preregistered (test|comparison|check|validity check|crossover analysis|confirmation)[^.]{0,40}"
                  r"(showed|did not confirm)")
DEV = re.compile(r"[Ii]n development experiments")
EXW = re.compile(r"[Pp]ost hoc")


def main():
    gen = (PAPER / "generated" / "numbers.tex").read_text(encoding="utf8")
    claim_of = {m.group(1): set(re.findall(r"C\d+", m.group(2)))
                for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{.*?\}  % .*\[(.*?)\]$", gen, flags=re.M)}
    bad = 0
    for f in RESULTS:
        text = (PAPER / "sections" / f).read_text(encoding="utf8")
        for b in re.split(r"\n(?=% \[)", text)[1:]:
            tag = re.match(r"% \[(.*?)\]", b).group(1)
            body = "\n".join(l for l in b.split("\n") if not l.lstrip().startswith("%"))
            body = body.split(r"\begin{table}")[0].split(r"\begin{figure}")[0]
            used = {c for m in re.findall(r"\\([A-Za-z]+)", body) if m in claim_of for c in claim_of[m]}
            extra = used - set(re.findall(r"C\d+", tag))
            if extra:
                bad += 1
                print(f"CLAIM {f}: [{tag}] uses macros of untagged claims {sorted(extra)}")
            roles = {"C"} if "PR-C" in tag else set()
            roles |= {"D"} if ("PR-D" in tag or "dev" in tag) else set()
            roles |= {"X"} if "EX" in tag else set()
            found = {"C": bool(CONF.search(body)), "D": bool(DEV.search(body)), "X": bool(EXW.search(body))}
            for r, present in found.items():
                if present and r not in roles:
                    bad += 1
                    print(f"ROLE  {f}: [{tag}] uses {r}-level wording")
    print(f"audit: {bad} violations")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
