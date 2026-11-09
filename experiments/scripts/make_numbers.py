"""Generate paper/generated/numbers.tex: every experimental number used in the paper, as a LaTeX macro.

    python scripts/make_numbers.py           -> write numbers.tex, then run the checks below
    python scripts/make_numbers.py --check   -> checks only (numbers.tex must exist)

Rule: this script computes nothing new. Each number is read from the artefact that RESULTS_MANIFEST.md names
for its claim, with the same convention:
  * figures/paper/figure_data.json (written by make_figures.py) for everything drawn in a figure;
  * the stdout of the analysis script that produced the claim (paired.py, counterfactual.py --summary,
    analyze_m0.py, analyze_ladder.py), captured and parsed;
  * the saved text reports (era5_report.txt, o1_verdict.txt, resops_ext.txt, conv_report.txt, o15_*_check.txt).
The only operations performed here are counting (e.g. how many datasets satisfy a stated condition), ratios of two
reported means (marked "derived" in the source comment), and formatting.

Checks (warnings, exit code 1 on errors):
  * macros used in paper/ but not defined here (error);
  * macros defined here but never used in paper/sections (warning; expected while sections are unwritten);
  * suspicious hand-typed numbers in paper/sections/*.tex: percentages, "p =", decimals, "k/n" counts. A line can
    be exempted by ending it with the comment "% num-ok" (use only for mathematical constants, not results).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT.parent / "paper"
OUT = PAPER / "generated" / "numbers.tex"

DS5 = ["syn_full", "syn_extreme", "seaice", "bucket", "resops"]
SYN = ["syn_full", "syn_extreme"]
REAL = ["seaice", "bucket", "resops"]
FAMS = ["trunc_gaussian", "logit_normal", "beta", "kumaraswamy"]
TAG = {"syn_full": "SynFull", "syn_extreme": "SynExtreme", "seaice": "SeaIce", "bucket": "Bucket",
       "resops": "ResOps", "syn_outatom": "SynOutAtom", "era5_cloud": "Era", "syn_atom": "SynAtom"}
FTAG = {"trunc_gaussian": "TruncGauss", "logit_normal": "LogitNormal", "beta": "Beta", "kumaraswamy": "Kum"}
NUMWORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
           9: "nine", 10: "ten"}

MACROS = {}   # name -> (tex, source, claim)


# --------------------------------------------------------------------------- helpers
def put(name, tex, source, claim):
    assert re.fullmatch(r"[A-Za-z]+", name), f"macro name must be letters only: {name}"
    assert name not in MACROS, f"duplicate macro {name}"
    MACROS[name] = (tex, source, claim)


def pct(x, signed=True, nd=1):
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return r"\ensuremath{" + s.replace("-", "-") + r"\%}"


def num(x, nd=3, signed=False):
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return r"\ensuremath{" + s + "}"


def pval(p):
    return r"\ensuremath{<0.001}" if p < 0.001 else r"\ensuremath{" + f"{p:.3f}" + "}"


def frac(k, n):
    return r"\ensuremath{" + f"{k}/{n}" + "}"


def count(name, k, source, claim, n=None):
    """Integer count: \\Name = digits, \\NameWord = English word (for 'on two of the three ...')."""
    put(name, str(k), source, claim)
    put(name + "Word", NUMWORD.get(k, str(k)), source, claim)
    if n is not None:
        put(name + "Frac", frac(k, n), source, claim)


def run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, capture_output=True,
                       text=True, check=True)
    return r.stdout


def grab(pattern, text, what, n=None):
    m = re.findall(pattern, text, flags=re.M)
    if not m or (n is not None and len(m) != n):
        raise RuntimeError(f"could not parse {what}: pattern {pattern!r} gave {len(m)} matches")
    return m


# --------------------------------------------------------------------------- C1, C2, C3 (paired.py + fig1)
def bounded_effect(fd):
    src = "figure_data.json fig1 (mean of per-seed relative change)"
    for ds in DS5:
        for fam in FAMS:
            for key, kt in (("biw", ""), ("biw_clipped", "Feas")):
                v = fd["fig1"][f"{ds}/{fam}/{key}"]
                put(f"Eff{kt}{TAG[ds]}{FTAG[fam]}", pct(v["mean"]), src, "C1")
                put(f"Eff{kt}{TAG[ds]}{FTAG[fam]}Sd", pct(v["sd"], signed=False), src, "C1")
    n_seeds, fail_c3, qg_lt, no_fn, pvals = set(), 0, [], 0, []
    for ds in DS5:
        out = run("paired.py", "--dataset", ds)
        row = grab(r"^\| main \(TN-hetero\) \| (\d+) \| [\d.]+ \| [\d.]+ \| [+-][\d.]+% \| ([\d.]+) \|", out,
                   f"paired.py main row {ds}", 1)[0]
        n_seeds.add(int(row[0]))
        put(f"PEff{TAG[ds]}Kum", pval(float(row[1])), f"paired.py --dataset {ds} (Wilcoxon, raw width)", "C1")
        pvals.append(float(row[1]))
        mt, mm, verdict = grab(r"C3 m\(tau=0\.1\) = ([\d.]+) vs m_- = ([\d.]+)\s+-> (\w+)", out, f"C3 {ds}", 1)[0]
        fail_c3 += verdict == "FAILS"
        put(f"MTau{TAG[ds]}", num(float(mt), 4), f"paired.py --dataset {ds} (m, tau = 0.1)", "C2")
        put(f"MMinus{TAG[ds]}", num(float(mm), 4), f"paired.py --dataset {ds} (m_-)", "C2")
        qg, qs = map(float, grab(r"q\^G = ([\d.]+), q\* = ([\d.]+)", out, f"quantiles {ds}", 1)[0])
        put(f"QGauss{TAG[ds]}", num(qg), f"paired.py --dataset {ds}", "C2")
        put(f"QStar{TAG[ds]}", num(qs), f"paired.py --dataset {ds}", "C2")
        if qg < qs:
            qg_lt.append(ds)
        fn = float(grab(r"FN vs h\* = ([\d.]+)", out, f"FN {ds}", 1)[0])
        no_fn += fn == 0.0
        put(f"FNStar{TAG[ds]}", num(fn, 4), f"paired.py --dataset {ds} (false negatives vs h*)", "C3")
        leak = float(grab(r"C1 leakage \(all test units\) = ([\d.]+)", out, f"leakage {ds}", 1)[0])
        put(f"Leak{TAG[ds]}", num(leak), f"paired.py --dataset {ds} (Gaussian leakage, mean over test units)",
            "C1 (EX)")
    assert len(n_seeds) == 1
    # sign pattern of the raw-width effect across the four bounded families (fig1)
    narrower, wider, mixed = [], [], []
    for ds in DS5:
        s = [fd["fig1"][f"{ds}/{fam}/biw"]["mean"] > 0 for fam in FAMS]
        (narrower if all(s) else wider if not any(s) else mixed).append(ds)
    count("BoundedNarrowerDatasets", len(narrower), "fig1: all four families narrower (raw width)", "C1", len(DS5))
    count("BoundedWiderDatasets", len(wider), "fig1: all four families wider (raw width)", "C1", len(DS5))
    count("BoundedMixedDatasets", len(mixed), "fig1: families disagree in sign (raw width)", "C1", len(DS5))
    assert narrower == ["syn_full", "syn_extreme", "seaice"] and wider == ["bucket", "resops"], (narrower, wider)
    agree = sum((fd["fig1"][f"{d}/kumaraswamy/biw"]["mean"] > 0) == (fd["fig1"][f"{d}/kumaraswamy/biw_clipped"]["mean"]
                                                                       > 0) for d in DS5)
    count("KumFeasAgreeDatasets", agree, "fig1: Kumaraswamy raw and feasible width agree in sign", "C1", len(DS5))
    count("PEffSignificantDatasets", sum(p < 0.05 for p in pvals), "paired.py: Wilcoxon p < 0.05 (Kumaraswamy)",
          "C1", len(DS5))
    # Holm step-down adjustment over the five preregistered primary comparisons (multiplicity check)
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    holm, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        holm[i] = running
    count("PEffHolmSignificantDatasets", sum(p < 0.05 for p in holm),
          "paired.py p-values, Holm adjustment over the five primary comparisons", "C1", len(DS5))
    put("PEffHolmMax", pval(max(holm)), "paired.py p-values, largest Holm-adjusted p (primary comparisons)", "C1")
    count("SeedsMain", n_seeds.pop(), "paired.py (seeds per dataset, main comparison)", "C1")
    count("NumDiscoveryDatasets", len(DS5), "fixed design", "design")
    count("NumSynDatasets", len(SYN), "fixed design", "design")
    count("NumRealDatasets", len(REAL), "fixed design", "design")
    count("NumFamilies", len(FAMS), "fixed design (bounded latent families)", "design")
    put("EpsBoundary", r"\ensuremath{0.05}", "protocol constant (RESULTS_MANIFEST.md header)", "design")
    count("QuantileConditionFailures", fail_c3, "paired.py: m(tau=0.1) <= m_- (Prop. 1 key condition)", "C2", len(DS5))
    count("QuantileContractionDatasets", len(qg_lt), "paired.py: q^G < q* on latent boundary group", "C2", len(DS5))
    count("GroupingNoFNDatasets", no_fn, "paired.py: mean FN vs h* exactly 0 (Theorem 1)", "C3", len(DS5))


# --------------------------------------------------------------------------- C4 phase diagram, ladder
def regime(fd):
    n_phase = len(list((RES / "phase").rglob("*.json")))
    count("PhaseRuns", n_phase, "results/phase/**/*.json", "C4")
    cells = sorted(p for p in (RES / "phase").iterdir() if p.is_dir())
    per_cell = {len(list(c.glob("gaussian__*.json"))) for c in cells}
    assert len(per_cell) == 1
    count("SeedsPhase", per_cell.pop(), "results/phase/<cell>/gaussian__*.json (seeds per cell)", "C4")
    count("PhaseCells", len(cells), "results/phase/<cell> directories", "C4")
    # phase_summary.json is written by analyze_phase.py; a cell's sign is +1/-1 if >= 4/5 seeds agree, else 0
    ps = json.loads((RES / "phase_summary.json").read_text())
    cells_all = []
    for fam in FAMS:
        cells = [c for row in ps[fam] for c in row if c]
        cells_all += cells
        count(f"PhaseConsistent{FTAG[fam]}", sum(c["sign"] != 0 for c in cells), "phase_summary.json", "C4",
              len(cells))
        count(f"PhaseBoundedBetter{FTAG[fam]}", sum(c["sign"] > 0 for c in cells), "phase_summary.json", "C4")
        count(f"PhaseGaussBetter{FTAG[fam]}", sum(c["sign"] < 0 for c in cells), "phase_summary.json", "C4")
    count("PhaseCellsPerFamily", len(cells), "phase_summary.json (cells with a defined primary metric)", "C4")
    put("PhaseLeakMin", num(min(c["M"] for c in cells_all), 2), "phase_summary.json realised Gaussian leakage", "C4")
    put("PhaseLeakMax", num(max(c["M"] for c in cells_all), 2), "phase_summary.json realised Gaussian leakage", "C4")
    n_ladder = sum(len(list((RES / f"ladder_B{b}").glob("*.json"))) for b in range(6))
    count("LadderRuns", n_ladder, "results/ladder_B*/*.json", "C4")
    out = run("analyze_ladder.py")
    obs = out.split("## grouping = oracle")[0]
    lv_word = ["Zero", "One", "Two", "Three", "Four", "Five"]
    for fam, ft in (("logit_normal", "LogitNormal"), ("kumaraswamy", "Kum")):
        block = obs.split(f"### {fam} vs gaussian")[1].split("###")[0]
        rows = grab(r"^\| B(\d)[^|]*\| ([+-][\d.]+)% \| (\d)/(\d) \|(?:[^|]*\|){5} ([\d.]+) \|", block,
                    f"ladder rows {fam}", 6)
        for lv, dw, pos, n, leak in rows:
            w = lv_word[int(lv)]
            put(f"LadderB{w}{ft}", pct(float(dw)), "analyze_ladder.py, obs grouping", "C4")
            put(f"LadderB{w}{ft}Pos", frac(int(pos), int(n)), "analyze_ladder.py (seeds with bounded narrower)", "C4")
            if ft == "Kum":
                put(f"LadderLeakB{w}", num(float(leak)), "analyze_ladder.py (Gaussian leakage L)", "C4")
    crit = grab(r"criterion 1 .*: levels (.+)$", obs, "ladder criterion 1", 1)[0]
    count("LadderStableReversalLevels", 0 if crit.strip() == "none" else len(eval(crit)), "analyze_ladder.py", "C4")
    seeds = len(list((RES / "ladder_B0").glob("gaussian__*.json")))
    count("SeedsLadder", seeds, "results/ladder_B0 (seeds per family)", "C4")
    # post hoc placement of the discovery datasets on the C axis (phase_placement.py -> figure_data fig_phase)
    fp = fd["fig_phase"]
    for ds in DS5:
        put(f"PlaceC{TAG[ds]}", num(fp["placement_C"][ds], 3), "figure_data.json fig_phase (median posterior sd)",
            "C4 (EX)")
    for c, ct in ((0.05, "FiveHundredths"), (0.01, "OneHundredth")):
        row = [x for fam in FAMS for x in fp["cells"][fam] if abs(x["C"] - c) < 1e-9]
        count(f"PhaseRow{ct}Bounded", sum(x["sign"] > 0 for x in row), "fig_phase cells (all families)", "C4",
              len(row))
    for c, ct in ((0.05, "FiveHundredths"), (0.01, "OneHundredth")):
        put(f"PhaseRow{ct}C", num(c, 2), "phase grid value of C (design)", "design")
        i = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2].index(c)
        ms = [x["M"] for x in ps["kumaraswamy"][i] if x]
        put(f"PhaseRow{ct}LeakMin", num(min(ms), 2), "phase_summary.json (Gaussian leakage in that row)", "C4")
        put(f"PhaseRow{ct}LeakMax", num(max(ms), 2), "phase_summary.json (Gaussian leakage in that row)", "C4")


# --------------------------------------------------------------------------- C5-C8 post-hoc correction
def posthoc(fd):
    f2 = fd["fig2"]
    rows_per_ds, uniq = set(), set()
    for ds in DS5:
        rows = json.loads((RES / "counterfactual" / f"{ds}.json").read_text())
        rows_per_ds.add(len(rows))
        uniq.add(len({(r["seed"], r["W_a"], r["W_e"]) for r in rows}))
        put(f"PostHocLogRatio{TAG[ds]}", num(f2[ds]["median_log_ratio"]), "figure_data.json fig2", "C5")
    assert len(rows_per_ds) == 1 and len(uniq) == 1
    n = sum(v["n"] for v in f2.values())
    k = sum(v["n_wider"] for v in f2.values())
    assert n == len(DS5) * uniq.copy().pop(), "fig2 must count one pair per (seed, dataset)"
    put("PostHocWorseCount", frac(k, n), "figure_data.json fig2 (unique seed-dataset pairs)", "C5")
    count("PostHocWider", k, "figure_data.json fig2", "C5")
    count("PostHocPairs", n, "figure_data.json fig2", "C5")
    rpd = rows_per_ds.pop()
    count("GateBRowsPerDataset", rpd, "results/counterfactual/{ds}.json (table rows)", "C5")
    count("GateBRowsTotal", rpd * len(DS5), "results/counterfactual/*.json (table rows, with repeats)", "C5")
    for ds in DS5:
        put(f"PostHocWider{TAG[ds]}", frac(f2[ds]["n_wider"], f2[ds]["n"]), "figure_data.json fig2", "C5")
    # decomposition pass (theory_check.py, independent Monte Carlo draws): median over seeds of log(W*/W^G)
    tc = json.loads((RES / "theory_check.json").read_text())
    for ds in DS5:
        v = sorted(r["log_Wratio"] for r in tc[ds])
        med = (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2
        put(f"TruncLogRw{TAG[ds]}", num(med, signed=True), "theory_check.json (median over seeds)", "C7")
    tw = sum(fd["fig4"][d]["wider"] for d in DS5), sum(fd["fig4"][d]["n"] for d in DS5)
    put("DecompPassWiderCount", frac(*tw), "figure_data.json fig4 (decomposition pass)", "C7")
    count("GateBUniquePairsPerDataset", uniq.pop(), "distinct (seed, W_a, W_e) per dataset", "C5")

    f4 = fd["fig4"]
    applicable = []
    for ds in DS5:
        put(f"ResidSkew{TAG[ds]}", num(f4[ds]["median_skew"], 2, signed=True), "figure_data.json fig4", "C6")
        put(f"ScaleRatioCV{TAG[ds]}", num(f4[ds]["median_ratio_cv"]), "figure_data.json fig4", "C6")
        # classification rule for "conditions (C) and (R) hold approximately" (stated in the paper's Fig. 4
        # caption): |median skewness| < 0.5 and median CV of the scale ratio < 0.1
        if abs(f4[ds]["median_skew"]) < 0.5 and f4[ds]["median_ratio_cv"] < 0.1:
            applicable.append(ds)
    assert applicable == SYN, f"Theorem 3 applicability changed: {applicable}"
    count("ThmThreeApplicableDatasets", len(applicable), "fig4 medians, rule |skew|<0.5 and CV<0.1", "C6", len(DS5))
    count("ThmThreeFailingRealDatasets", len([d for d in REAL if d not in applicable]), "as above", "C6", len(REAL))

    f3 = fd["fig3"]
    amp, real_scale = 0, 0
    for ds in DS5:
        t = f3["truncation"][ds]
        put(f"TruncLogRq{TAG[ds]}", num(t["median_log_Rq"], signed=True), "figure_data.json fig3 truncation", "C7")
        put(f"TruncLogRs{TAG[ds]}", num(t["median_log_Rsigma"], signed=True), "figure_data.json fig3 truncation", "C7")
        amp += t["median_log_Rsigma"] > 0
        if ds in REAL and t["median_log_Rsigma"] > t["median_log_Rq"]:
            real_scale += 1
            assert ds in ("seaice", "bucket"), "the paper names sea ice and Bucket as scale-dominant; update the text"
        k = f3["kumaraswamy_vs_gaussian"][ds]
        put(f"KumLogRq{TAG[ds]}", num(k["median_log_Rq"], signed=True), "figure_data.json fig3 kum vs gauss", "C7")
        put(f"KumLogRs{TAG[ds]}", num(k["median_log_Rsigma"], signed=True), "figure_data.json fig3 kum vs gauss", "C7")
    count("ScaleAmplifiedDatasets", amp, "fig3 truncation: median log R_sigma > 0", "C7", len(DS5))
    count("RealScaleDominantDatasets", real_scale, "fig3 truncation: log R_sigma > log R_q (real data)", "C7",
          len(REAL))

    kdom = sum(abs(f3["kumaraswamy_vs_gaussian"][d]["median_log_Rsigma"]) >
               abs(f3["kumaraswamy_vs_gaussian"][d]["median_log_Rq"]) for d in DS5)
    count("KumScaleDominantDatasets", kdom, "fig3 kum vs gauss: |log R_sigma| > |log R_q|", "C7 (EX)", len(DS5))

    out = run("counterfactual.py", "--summary")
    si = grab(r"^\| (\w+) \| seaice \| ([+-][\d.]+)[^|]*\| [+-][\d.]+ \| [+-][\d.]+ \| ([+-][\d.]+) \|", out,
              "Gate B sea-ice rows", len(FAMS))
    ratios = [abs(float(inter)) / abs(float(gap)) for _, gap, inter in si]
    count("GateBSeaIceInteractionExceeds", sum(r > 1 for r in ratios), "counterfactual.py --summary, sea ice: "
          "|interaction| > |gap|", "C8", len(FAMS))
    put("GateBSeaIceInteractionRatioMin", num(min(ratios), 0), "counterfactual.py --summary (derived ratio)", "C8")
    put("GateBSeaIceInteractionRatioMax", num(max(ratios), 0), "counterfactual.py --summary (derived ratio)", "C8")
    for fam in FAMS:
        k = int(grab(rf"^- {fam}: (\d)/5 datasets pass", out, f"Gate B {fam}", 1)[0])
        count(f"GateBPass{FTAG[fam]}", k, "counterfactual.py --summary", "C8", len(DS5))
    k = int(grab(r"Gate B: (\d)/4 families pass", out, "Gate B verdict", 1)[0])
    count("GateBFamiliesPass", k, "counterfactual.py --summary", "C8", len(FAMS))


# --------------------------------------------------------------------------- C9 Gate M0
def gate_m0():
    out = run("analyze_m0.py")
    side = out.split("## Side analyses")[1]
    for ds in ("bucket", "seaice", "resops", "syn_atom", "syn_full"):
        line = grab(rf"^- {ds}: (.*)$", side, f"M0 side {ds}", 1)[0]
        for a, b, name in (("4", "1b", "MixVsSameScore"), ("1b", "1", "ScoreEffect")):
            rel, p = grab(rf"{re.escape(a)} vs {re.escape(b)} [\d.]+ vs [\d.]+ \(([+-][\d.]+)%, p=([\d.]+)\)",
                          line, f"M0 {ds} {a} vs {b}", 1)[0]
            put(f"MZero{name}{TAG[ds]}", pct(float(rel)), "analyze_m0.py side analyses", "C9")
            put(f"PMZero{name}{TAG[ds]}", pval(float(p)), "analyze_m0.py side analyses", "C9")
    seeds = len(list((RES / "m0" / "bucket").glob("model1__s*.json")))
    count("SeedsMZero", seeds, "results/m0/bucket/model1__s*.json", "C9")
    conv = (RES / "conv_report.txt").read_text()
    w = [float(x) for x in grab(r"\|W2000-W1000\|/W2000 = ([\d.]+)%", conv, "conv width")]
    c = [float(x) for x in grab(r"\|cov2000-cov1000\| = ([\d.]+)", conv, "conv coverage")]
    stable = grab(r"ranking stable for G >= 1000: (\w+)", conv, "conv ranking")
    assert all(s == "True" for s in stable)
    put("ConvMaxWidthChange", pct(max(w), signed=False, nd=2), "conv_report.txt (max over datasets and scores)", "C9")
    put("ConvMaxCovChange", num(max(c), 4), "conv_report.txt (max over datasets and scores)", "C9")


# --------------------------------------------------------------------------- C10-C11 Gate O1
def gate_o1(fd):
    t = (RES / "o1_verdict.txt").read_text()
    mt = {"A": "A", "B": "B", "C": "C", "D": "D", "E1": "EOne", "E2": "ETwo", "E3": "EThree", "E4": "EFour"}
    o1_ds = ("seaice", "resops", "syn_outatom")
    for ds in o1_ds:
        block = t.split(f"## {ds} ")[1].split("##")[0]
        rows = grab(r"^\| (\w+) \| ([\d.]+) \| ([\d.]+) \| ([\w.]+) \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \| "
                    r"([\d.]+) \|", block, f"O1 table {ds}", 8)
        for m, mu, na, prec, a05, cov, covb in rows:
            src = f"o1_verdict.txt {ds} table (mean over seeds)"
            put(f"OOne{TAG[ds]}{mt[m]}MuVal", num(float(mu), 4), src, "C10")
            put(f"OOne{TAG[ds]}{mt[m]}Atoms", num(float(na), 2), src, "C10")
            put(f"OOne{TAG[ds]}{mt[m]}AmbVal", num(float(a05), 4), src, "C10")
            put(f"OOne{TAG[ds]}{mt[m]}Cov", num(float(cov)), src, "C10")
            if prec != "nan":
                put(f"OOne{TAG[ds]}{mt[m]}Precision", num(float(prec), 2), src, "C10")
        # condition-3 lines: relative change vs A of mu and A_0.05, with Wilcoxon p, and the verdict
        cond = grab(rf"^   {ds} (\w+): mu ([+-][\d.]+)% \(p=([\d.]+)\), A_0\.05 ([+-][\d.]+)% \(p=([\d.]+)\)"
                    rf"(?:;[^>]*)? -> (\w+)", t, f"O1 condition 3 {ds}")
        valid = 0
        for m, mu, pmu, a05, pa, ok in cond:
            src = "o1_verdict.txt condition 3 (vs A, paired Wilcoxon)"
            put(f"OOne{TAG[ds]}{mt[m]}Mu", pct(float(mu)), src, "C10")
            put(f"POOne{TAG[ds]}{mt[m]}Mu", pval(float(pmu)), src, "C10")
            put(f"OOne{TAG[ds]}{mt[m]}Amb", pct(float(a05)), src, "C10")
            put(f"POOne{TAG[ds]}{mt[m]}Amb", pval(float(pa)), src, "C10")
            valid += ok == "True"
        count(f"OOneValidGains{TAG[ds]}", valid, "o1_verdict.txt condition 3 (methods with a valid gain)", "C10",
              len(cond))
    put("OOneResOpsFreeAtomRate", MACROS["OOneResOpsBAtoms"][0], "o1_verdict.txt resops B atoms/set", "C10")
    put("OOneSeaIceContinuousMu", MACROS["OOneSeaIceAMuVal"][0], "o1_verdict.txt seaice A mu", "C10")
    put("OOneResOpsContinuousMu", MACROS["OOneResOpsAMuVal"][0], "o1_verdict.txt resops A mu", "C10")
    # E5 development self-check (discovery data only; not used to change the design): mean over seeds
    e5 = json.loads((RES / "e5_dev.json").read_text())
    worse = 0
    for ds in o1_ds:
        v = [r["mu"] for r in e5 if r["dataset"] == ds]
        mu = sum(v) / len(v)
        put(f"EFiveDev{TAG[ds]}Mu", num(mu, 4), "e5_dev.json (mean over seeds)", "C13 (dev self-check)")
        worse += mu > float(MACROS[f"OOne{TAG[ds]}AMuVal"][0].strip(r"\ensuremath{}"))
    count("EFiveDevWorseThanA", worse, "e5_dev.json vs o1_verdict.txt A mu", "C13 (dev self-check)", len(o1_ds))
    share = grab(r"D retains (\d+)%", t, "O1 condition 2", 1)[0]
    put("OOneResOpsDRetains", pct(float(share), signed=False, nd=0), "o1_verdict.txt condition 2", "C10")
    seeds = len({r["seed"] for r in json.loads((RES / "o1.json").read_text()) if r["dataset"] == "seaice"})
    count("SeedsOOne", seeds, "results/o1.json", "C10")
    ext = (RES / "resops_ext.txt").read_text()
    rel, p, n = grab(r"CONFIRMATORY .*\(([+-][\d.]+)%\), Wilcoxon p = ([\d.]+), n = (\d+)", ext, "resops ext", 1)[0]
    put("ResOpsExtEffect", pct(float(rel)), "resops_ext.txt confirmatory line", "C10")
    put("PResOpsExt", pval(float(p)), "resops_ext.txt confirmatory line", "C10")
    count("SeedsResOpsExt", int(n), "resops_ext.txt confirmatory line", "C10")

    f5 = fd["fig5"]
    for ds, nd in (("seaice", 3), ("syn_outatom", 3), ("era5_cloud", 4)):
        for m, v in f5[ds].items():
            ls = v["lambda_star_median"]
            if m == "A" or ls is None:
                continue
            tex = num(ls, nd) if ls > 0 else r"\ensuremath{<0}"
            put(f"LambdaStar{TAG[ds]}{m.replace('4', 'Four').replace('5', 'Five')}", tex,
                "figure_data.json fig5 (median of per-seed lambda*)", "C11 (ERA5: PR-C; others: EX)")


# --------------------------------------------------------------------------- C12 O1.5
def structure(fd):
    tags = {"Sea ice": "SeaIce", "ResOpsUS": "ResOps", "Snow_cci CryoClim": "CryoClim", "MODIS SCFG": "Scfg",
            "MODIS SCFV": "Scfv", "ERA5 cloud": "Era"}
    for k, v in fd["fig6"].items():
        for key, kt in (("p0", "PZero"), ("p1", "POne"), ("interior", "Interior")):
            nd = 4 if 0 < v[key] < 0.01 else 3        # 3 decimals would round 0.0045 ambiguously
            put(f"Struct{tags[k]}{kt}", num(v[key], nd), "figure_data.json fig6", "C12")
    n_ok = 0
    for fn in ("o15_cloud_check.txt", "o15_snow_check.txt", "o15_modis_scfg_check.txt", "o15_modis_scfv_check.txt"):
        n_ok += "MEETS the O1.5 criteria" in (RES / fn).read_text()
    count("OOneFiveCandidates", 4, "results/o15_*_check.txt (number of candidates checked)", "C12")
    count("OOneFivePassing", n_ok, "results/o15_*_check.txt", "C12", 4)


# --------------------------------------------------------------------------- C13 ERA5
def era5(fd):
    t = (RES / "era5_report.txt").read_text()
    # lambda*: per-seed values, median over seeds in which the method output atoms (report), checked against fig5
    for m, med, n in grab(r"^  (\w+): median ([+-]?[\d.]+), IQR \[[^]]*\], n = (\d+)", t, "ERA5 lambda*", 4):
        f5 = fd["fig5"]["era5_cloud"][m]["lambda_star_median"]
        assert abs(f5 - float(med)) < 5e-5, f"lambda* mismatch for {m}: fig5 {f5} vs report {med}"
        count(f"EraLambdaSeeds{m.replace('4', 'Four').replace('5', 'Five')}", int(n),
              "era5_report.txt (seeds with at least one atom, used for lambda*)", "C11 (PR-C)")
    rows = grab(r"^\| (\w+) \| ([\d.]+) \(([\d.]+)\) \| ([\d.]+) \| ([\w.]+) \| ([\d.]+) \| ([\d.]+) \| "
                r"([\d.]+) \(([\d.]+)\) \|", t, "ERA5 table")
    means = {}
    for m, mu, mu_med, na, prec, cov, covb, a05, a05_med in rows:
        mt = m.replace("4", "Four").replace("5", "Five").replace("1", "One").replace("2", "Two")
        means[m] = (float(mu), float(a05))
        put(f"EraMu{mt}", num(float(mu)), "era5_report.txt table (mean over seeds)", "C13")
        put(f"EraMuMedian{mt}", num(float(mu_med)), "era5_report.txt table (median over seeds)", "C13")
        put(f"EraAmb{mt}", num(float(a05)), "era5_report.txt table", "C13")
        put(f"EraAtoms{mt}", num(float(na), 2), "era5_report.txt table", "C13")
        put(f"EraCov{mt}", num(float(cov)), "era5_report.txt table", "C13")
        if prec != "nan":
            put(f"EraPrecision{mt}", num(float(prec)), "era5_report.txt table", "C13")
    covs = [float(r[5]) for r in rows]
    viol = grab(r"coverage outside 0\.90 \+/- 0\.02 \(width results invalid\): (\w+)", t, "ERA5 coverage", 1)[0]
    assert viol == "none" and all(abs(c - 0.90) <= 0.02 for c in covs)
    put("EraCovMin", num(min(covs)), "era5_report.txt table", "C13")
    put("EraCovMax", num(max(covs)), "era5_report.txt table", "C13")
    tests = grab(r"^- (\w+) vs (\w+) on (mu|A_0\.05): [\d.]+ vs [\d.]+, median diff [+-][\d.]+, p = ([\d.]+)", t,
                 "ERA5 tests", 7)
    for a, b, metric, p in tests:
        name = f"PEra{a}Vs{b}{'Mu' if metric == 'mu' else 'Amb'}".replace("4", "Four").replace("5", "Five")
        put(name, pval(float(p)), "era5_report.txt pre-registered comparisons", "C13")
    for m, mt_ in (("B", "B"), ("C", "C"), ("E5", "EFive")):
        put(f"Era{mt_}VsAMuRel", pct(100 * (means[m][0] / means["A"][0] - 1)),
            "derived: ratio of the two mu means in era5_report.txt", "C13")
        put(f"Era{mt_}VsAAmbRel", pct(100 * (means[m][1] / means["A"][1] - 1), nd=0),
            "derived: ratio of the two A_0.05 means in era5_report.txt", "C13")
    rel = 100 * (means["E5"][1] / means["E4"][1] - 1)
    put("EraEFiveVsEFourAmbRel", pct(rel, nd=0), "derived: ratio of the two A_0.05 means in era5_report.txt", "C13")
    seeds = len(json.loads((RES / "era5_confirm.json").read_text()))
    count("SeedsEra", seeds, "results/era5_confirm.json", "C13")


# --------------------------------------------------------------------------- C14 Theorem 2 check (review B)
def shift_check():
    sc = json.loads((RES / "shift_check.json").read_text())
    med = lambda rows, k: sorted(r[k] for r in rows)[len(rows) // 2 - 1: len(rows) // 2 + 1]
    src = "results/shift_check.json (median over seeds)"
    widen = agree = away = n = 0
    for ds in DS5:
        rows = sc[ds]
        for k, nd, signed, mt in (("b_star", 2, True, "BStar"), ("b_median", 2, True, "BMed"), ("eta", 2, False, "Eta"),
                                  ("log_shift", 3, True, "LogShift"), ("log_remainder", 3, True, "LogRemainder"),
                                  ("log_pred_common", 3, True, "LogPred")):
            m = sum(med(rows, k)) / 2
            put(f"Shift{mt}{TAG[ds]}", num(m, nd, signed=signed), src, "C14 (EX)")
        widen += sum(r["log_shift"] > 0 for r in rows)
        agree += sum((r["log_pred_common"] > 0) == (r["log_shift"] > 0) for r in rows)
        away += sum(r["away"] for r in rows)
        n += len(rows)
        count(f"ShiftWiden{TAG[ds]}", sum(r["log_shift"] > 0 for r in rows), src, "C14 (EX)", len(rows))
        count(f"ShiftAgree{TAG[ds]}", sum((r["log_pred_common"] > 0) == (r["log_shift"] > 0) for r in rows), src,
              "C14 (EX)", len(rows))
    put("ShiftWidenCount", frac(widen, n), "results/shift_check.json (seed-dataset pairs)", "C14 (EX)")
    put("ShiftPredAgreeCount", frac(agree, n), "results/shift_check.json (seed-dataset pairs)", "C14 (EX)")
    put("ShiftAwayCount", frac(away, n), "results/shift_check.json (seed-dataset pairs)", "C14 (EX)")
    etas = [sum(med(sc[d], "eta")) / 2 for d in DS5]
    put("ShiftEtaMin", num(min(etas), 2), "results/shift_check.json (smallest dataset median)", "C14 (EX)")
    count("ShiftRemainderNegDatasets", sum(sum(med(sc[d], "log_remainder")) / 2 < 0 for d in DS5),
          "results/shift_check.json (median remainder < 0)", "C14 (EX)", len(DS5))


# --------------------------------------------------------------------------- C16 TCN robustness (review A)
def tcn_check():
    tc = json.loads((RES / "tcn_check.json").read_text())
    src = "results/tcn_check.json (tcn_check.py, preregistered)"
    ds3 = ("bucket", "resops", "seaice")
    for ds in ds3:
        r = tc[ds]
        put(f"Tcn{TAG[ds]}Rho", pct(r["rho_mean"]), src, "C16 (PR-C)")
        put(f"Tcn{TAG[ds]}RhoSd", pct(r["rho_sd"], signed=False), src, "C16 (PR-C)")
        put(f"PTcn{TAG[ds]}", pval(r["p"]), src, "C16 (PR-C)")
    count("TcnReproduced", sum(tc[d]["reproduced"] for d in ds3), src, "C16 (PR-C)", len(ds3))
    put("TcnTruncWider", frac(sum(tc[d]["trunc_wider"] for d in ds3), sum(tc[d]["trunc_n"] for d in ds3)), src,
        "C16 (PR-C)")
    covs = [c for d in ds3 for c in tc[d]["trunc_cov"]] + [tc[d][k] for d in ds3 for k in ("coverage_min", "coverage_max")]
    assert all(abs(c - 0.90) <= 0.02 for c in covs), "TCN coverage outside the validity band"
    put("TcnCovMin", num(min(covs)), src, "C16 (PR-C)")
    put("TcnCovMax", num(max(covs)), src, "C16 (PR-C)")
    assert tc["reversal_independent_of_gru"] is True or tc["reversal_independent_of_gru"] is False
    put("TcnSeeds", str(len(tc["bucket"]["rho"])), src, "C16 (PR-C)")


# --------------------------------------------------------------------------- C17 latent alignment of the swap (review D)
def latent_alignment():
    la = json.loads((RES / "latent_alignment.json").read_text())
    src = "results/latent_alignment.json (latent_alignment.py, post hoc, criterion fixed before running)"
    s = la["summary"]
    ok = [d for d in s if s[d]["adequate"]]
    count("AlignAdequate", len(ok), src, "C17 (EX)", len(s))
    put("AlignCorrMinAll", num(min(s[d]["min_corr_min"] for d in s), 2), src, "C17 (EX)")
    put("AlignCorrCrit", num(la["criterion"]["corr_min"], 1), "latent_alignment.py (criterion)", "design")
    for d in ("seaice",):
        put(f"AlignMadGObs{TAG[d]}", num(s[d]["mad_G_obs"]), src, "C17 (EX)")
        put(f"AlignMadBObs{TAG[d]}", num(s[d]["mad_B_obs"]), src, "C17 (EX)")
        put(f"AlignMadGB{TAG[d]}", num(s[d]["mad_GB"]), src, "C17 (EX)")
        put(f"AlignObsNoise{TAG[d]}", num(s[d]["obs_noise"]), src, "C17 (EX)")
    assert [d for d in s if not s[d]["adequate"]] == ["seaice"], "text assumes sea ice is the only failure"
    put("AlignMadGBMaxOther", num(max(s[d]["max_mad_GB"] for d in s if d != "seaice")), src, "C17 (EX)")


# --------------------------------------------------------------------------- C18 temporal extrapolation (review E)
def temporal_check():
    tc = json.loads((RES / "temporal_check.json").read_text())
    src = "results/temporal_check.json (temporal_check.py, preregistered)"
    cl = "C18 (PR-C)"
    for m, t in (("kumaraswamy", "Kum"), ("trunc_star", "Trunc")):
        put(f"TempDid{t}", num(tc["did"][m]["mean"], 3, signed=True), src, cl)
        put(f"PTempDid{t}", pval(tc["did"][m]["p"]), src, cl)
    n_fail = sum(v["differential_failure"] for v in tc["did"].values())
    assert n_fail == 0 and tc["P1_any_differential_failure"] is False, "text assumes no differential failure"
    count("TempDiffFail", n_fail, src, cl, len(tc["did"]))
    covs = [tc["coverage"][m][a][k] for m in tc["coverage"] for a in ("C", "T") for k in ("coverage", "coverage_boundary")]
    assert all(abs(c - 0.90) <= 0.02 for c in covs), "temporal coverage outside the validity band"
    put("TempCovMin", num(min(covs)), src, cl)
    put("TempCovMax", num(max(covs)), src, cl)
    nt = tc["nontemporal"]
    put("TempRhoKum", pct(nt["rho_W_kum_T_mean"]), src, cl + " non-temporal")
    put("TempTruncWider", frac(nt["trunc_wider_T"], nt["n"]), src, cl + " non-temporal")
    dr = tc["drift"]
    put("TempMeanYCtrl", num(dr["C"]["mean_y"]), src, cl + " descriptor")
    put("TempMeanYLate", num(dr["T"]["mean_y"]), src, cl + " descriptor")
    put("TempBndShareCtrl", num(dr["C"]["boundary_share"]), src, cl + " descriptor")
    put("TempBndShareLate", num(dr["T"]["boundary_share"]), src, cl + " descriptor")
    put("TempDidThr", num(tc["design"]["did_min"], 2), "temporal_check.py (criterion)", "design")
    put("TempSeeds", str(nt["n"]), src, cl)
    mtag = {"gaussian": "Gauss", "trunc_star": "Trunc", "kumaraswamy": "Kum", "trunc_gaussian": "TruncGauss",
            "logit_normal": "LogitNormal", "beta": "Beta"}
    for m, t in mtag.items():
        for arm, at in (("C", "Ctrl"), ("T", "Late")):
            put(f"TempCov{t}{at}", num(tc["coverage"][m][arm]["coverage"]), src, cl)
            put(f"TempCovB{t}{at}", num(tc["coverage"][m][arm]["coverage_boundary"]), src, cl)


# --------------------------------------------------------------------------- C19 endpoint-inclusive size (review F)
def endpoint_cost():
    ec = json.loads((RES / "endpoint_cost.json").read_text())
    src = "results/endpoint_cost.json (endpoint_cost.py, post hoc, plan committed before running)"
    cl = "C19 (EX)"
    s = ec["summary"]
    rb = ec["reconstruction_by_method"]
    put("EndReconMaxAtom", num(max(rb[m] for m in ("A", "B", "C", "E4", "E5")), 5), src, cl)
    for ds in ("seaice", "resops", "syn_outatom", "era5_cloud"):
        put(f"EndAContains{TAG[ds]}", pct(100 * s[ds]["A_frac_contains_endpoint"], signed=False), src, cl)
        for m, t in (("B", "B"), ("C", "C"), ("E4", "EFour"), ("E5", "EFive")):
            put(f"End{TAG[ds]}{t}Diff", num(s[ds][m]["nu_vs_A_meandiff"], 3, signed=True), src, cl)
            put(f"PEnd{TAG[ds]}{t}", pval(s[ds][m]["nu_vs_A_p"]), src, cl)
    a = ec["answers"]
    # the text relies on these answers
    assert a["Q1_era5_any_atom_aware_sig_smaller_nu"] is False
    assert a["Q2_real_dev_any_sig_smaller_nu"] == {"seaice": True, "resops": False}
    assert a["Q3_synoutatom_B_sig_smaller_nu"] is True
    assert s["seaice"]["C"]["nu_sig_smaller"] and not s["seaice"]["B"]["nu_sig_smaller"] and not s["seaice"]["E4"]["nu_sig_smaller"]
    assert s["seaice"]["B"]["nu_sig_larger"] and s["era5_cloud"]["B"]["nu_sig_larger"] and s["era5_cloud"]["E4"]["nu_sig_larger"]
    assert not s["era5_cloud"]["C"]["nu_sig_larger"] and not s["era5_cloud"]["E5"]["nu_sig_larger"]
    put("EndLambda", num(ec["lambda"], 2), "endpoint_cost.py (design)", "design")


# --------------------------------------------------------------------------- C20-C22 review checks (post hoc)
def review_checks():
    rc = json.loads((RES / "review_checks.json").read_text())
    src = "results/review_checks.json (review_checks.py, post hoc, plan committed before running)"
    # C20: fit quality of the bounded-latent comparison
    r1, cl = rc["R1_fit_quality"], "C20 (EX)"
    cells = {(c["dataset"], c["family"]): c for c in r1["cells"]}
    for (ds, fam), c in cells.items():
        t = TAG[ds] + FTAG[fam]
        put(f"FitDval{t}", num(c["dval_mean"], 3, signed=True), src, cl)
        put(f"PFitDval{t}", pval(c["dval_p"]), src, cl)
        put(f"FitDylp{t}", num(c["dylp_mean"], 3, signed=True), src, cl)
        put(f"PFitDylp{t}", pval(c["dylp_p"]), src, cl)
        put(f"FitRho{t}", pct(c["rho_raw"]), src, cl)
        put(f"PFitRho{t}", pval(c["rho_raw_p"]), src, cl)
    # the text relies on this reading
    assert r1["c_claim_survives"] is True
    assert cells[("resops", "kumaraswamy")]["fit_confounded"] and cells[("resops", "beta")]["fit_confounded"]
    for fam in ("logit_normal", "trunc_gaussian"):
        c = cells[("resops", fam)]
        assert c["sig_widen"] and not c["fits_worse"], fam
    for fam in ("kumaraswamy", "beta"):
        c = cells[("bucket", fam)]
        assert c["sig_widen"] and not c["fits_worse"], fam
    sea = [cells[("seaice", f)] for f in FTAG]
    assert all(c["fits_worse"] and c["sig_narrow"] for c in sea), "sea ice: every family fits worse yet narrows"
    put("FitDvalSeaIceMin", num(min(c["dval_mean"] for c in sea), 2, signed=True), src, cl)
    put("FitDvalSeaIceMax", num(max(c["dval_mean"] for c in sea), 2, signed=True), src, cl)
    count("FitCleanWiden", len(r1["c_clean_widening_cells"]), src, cl)
    sp = r1["e_spearman_dval_rho"]
    assert sp["p"] > 0.05, "text says the fit gap does not order the effect"
    put("FitSpearman", num(sp["rho"], 2, signed=True), src, cl)
    put("PFitSpearman", pval(sp["p"]), src, cl)
    # C21: 95% intervals for the held-out ERA5 comparison
    r2, cl = rc["R2_era5_intervals"], "C21 (EX)"
    mtag = {"B": "B", "C": "C", "E4": "EFour", "E5": "EFive"}
    for m, row in r2.items():
        for metric, mt in (("mu", "Mu"), ("A_0.05", "Amb"), ("nu", "Nu")):
            v = row[metric]
            put(f"EraCi{mt}{mtag[m]}Lo", pct(v["rel_ci_lo_pct"]), src, cl)
            put(f"EraCi{mt}{mtag[m]}Hi", pct(v["rel_ci_hi_pct"]), src, cl)
            put(f"EraAdv{mt}{mtag[m]}", pct(v["largest_compatible_advantage_pct"], signed=False), src, cl)
    adv = lambda metric: {m: r2[m][metric]["largest_compatible_advantage_pct"] for m in r2}
    assert max(adv("mu"), key=adv("mu").get) == "B", "text names the free-atom set for the largest mu advantage"
    assert max(adv("nu"), key=adv("nu").get) == "E5", "text names E5 for the largest endpoint-inclusive advantage"
    assert all(adv("A_0.05")[m] == 0.0 for m in ("B", "C", "E4")), "text: no advantage in A_0.05 for B, C, E4"
    # C22: scale of the bounded-latent effect
    r3, cl = rc["R3_scale"], "C22 (EX)"
    for ds, v in r3.items():
        put(f"BndShare{TAG[ds]}", pct(100 * v["boundary_share"], signed=False, nd=0), src, cl)
        put(f"WidthG{TAG[ds]}", num(v["boundary_width"], 3), src, cl)


# --------------------------------------------------------------------------- C15 decision example (review C)
def decision_check():
    dc = json.loads((RES / "decision_check.json").read_text())
    assert dc["reconstruction_max_abs_diff"] < 1e-4, "set reconstruction does not match o1.json"
    src = "results/decision_check.json (mean over seeds)"
    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    for part, tags in (("latent", {"gaussian": "Gauss", "kumaraswamy": "Kum"}),
                       ("atoms", {"A": "A", "B": "B", "C": "C", "E4": "EFour"})):
        for m, t in tags.items():
            rows = dc[part][m]
            put(f"Dec{t}Decisive", pct(100 * mean(rows, "decisive"), signed=False), src, "C15 (EX)")
            put(f"Dec{t}Error", pct(100 * mean(rows, "error"), signed=False), src, "C15 (EX)")
    put("DecThreshold", num(0.15, 2), "decision_check.py (ice-edge threshold, design)", "design")


# --------------------------------------------------------------------------- implementation tables
def write_impl():
    """paper/generated/implementation.tex: data and training settings read from the code's own configuration
    (dataclass defaults, argparse defaults, module constants), so the supplement cannot drift from the code."""
    import importlib
    import inspect
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    D = {m: importlib.import_module(f"scrl.{m}") for m in ("data_synthetic", "data_phase", "data_outatom", "data_bucket",
                                                          "data_seaice", "data_resops", "data_era5", "models", "o1_eval")}
    src = (ROOT / "scripts" / "run_synthetic.py").read_text()
    arg = {k: v for k, v in re.findall(r'add_argument\("--(\w+)", type=\w+, default=([\w.e-]+)\)', src)}
    eps = re.search(r"EPS_GRID = \(([^)]*)\)", src).group(1)
    o1src = (ROOT / "scripts" / "o1.py").read_text()
    taus = re.search(r"TAUS = \[([^\]]*)\]", o1src).group(1)
    cf_S = re.search(r"EPS, ALPHA, S = [^,]+, [^,]+, (\d+)", (ROOT / "scripts" / "counterfactual.py").read_text()).group(1)
    mdl = inspect.signature(D["models"].Model.__init__).parameters
    syn, ph, oa = D["data_synthetic"].SynConfig(), D["data_phase"].PhaseConfig(), D["data_outatom"].OutAtomConfig()
    bk, si, ro, er = (D["data_bucket"].BucketConfig(), D["data_seaice"].SeaIceConfig(), D["data_resops"].ResOpsConfig(),
                      D["data_era5"].ERA5Config())
    n3 = lambda c: f"{c.n_train:,}/{c.n_cal:,}/{c.n_test:,}".replace(",", "{,}")
    frac = lambda f: f"{round(100 * f)}\\%"
    rows = [
        ("Syn-Full, Syn-Extreme", syn.d, syn.K, f"{syn.T} steps", "final state", n3(syn), "independent trajectories"),
        ("Phase generator", 1, 1, f"{ph.T} obs.", "static latent", n3(ph), "independent units"),
        ("Syn-OutAtom", oa.d, 1, f"{oa.T} obs.", "static latent", n3(oa), "independent units"),
        ("Bucket", 2, 1, f"{bk.T} days", "next day", n3(bk),
         f"{frac(bk.train_site_frac)} of sites for training; evaluation from the other sites"),
        ("Sea ice", 1, 4, f"{si.T} days", f"{si.horizon} days", n3(si),
         f"{frac(si.train_block_frac)} of spatial blocks, {si.train_years[0]}--{si.train_years[1]}; evaluation "
         f"blocks, {si.eval_years[0]}--{si.eval_years[1]}"),
        ("ResOpsUS", 1, 4, f"{ro.T} days", f"{ro.horizon} days", n3(ro),
         f"{frac(ro.train_dam_frac)} of dams, {ro.train_years[0]}--{ro.train_years[1]}; other dams, "
         f"{ro.eval_years[0]}--{ro.eval_years[1]}"),
        ("ERA5 cloud cover", 1, 1, f"{er.T} hours", f"{er.horizon} hours", n3(er),
         f"{er.n_train_sites} of 50 sites, 2010--2016; other sites, 2017--2019; split redrawn per seed"),
    ]
    L = [r"% AUTO-GENERATED by experiments/scripts/make_numbers.py (write_impl) -- do not edit by hand.",
         r"\begin{table}[h]", r"\centering", r"\small",
         r"\caption{Datasets as configured in the code. $d$: latent dimension; $K$: number of regimes; window: length "
         r"of the input window; horizon: lead time of the target; counts: training/calibration/test units per seed. "
         r"Ten percent of the training units are held back for early stopping and for the selection steps of "
         r"\cref{sec:protocol-models}. Calibration and test units are drawn independently, with replacement, from "
         r"the same evaluation pool of (location, start time) pairs; windows may overlap in time.}",
         r"\label{tab:impl-data}",
         r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.5cm}cc>{\raggedright\arraybackslash}p{1.3cm}"
         r">{\raggedright\arraybackslash}p{1.3cm}>{\raggedright\arraybackslash}p{2.9cm}"
         r">{\raggedright\arraybackslash}p{3.8cm}}",
         r"\toprule", r"Dataset & $d$ & $K$ & Window & Horizon & Counts & Split \\", r"\midrule"]
    L += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    oe = D["o1_eval"]
    fmt_list = lambda xs: "\\{" + ", ".join(f"{x:g}" for x in xs) + "\\}"
    hp = [
        ("Encoder", f"GRU, one layer, hidden size {mdl['hidden'].default}; input: observation window and one-hot regime"),
        ("Decoder", f"MLP with two hidden layers of width {mdl['dec_hidden'].default} (SiLU); heteroscedastic "
                    "truncated normal on $[0,1]$"),
        ("Optimizer", f"AdamW, learning rate {float(arg['lr']):g}, weight decay $10^{{-4}}$, cosine schedule"),
        ("Training", f"batch size {arg['batch']}, at most {arg['epochs']} epochs, early stopping with patience "
                     f"{arg['patience']}; KL weight {float(arg['beta']):g}"),
        ("Posterior samples", f"{arg['S_train']} per unit in training, {arg['S_eval']} at evaluation, {cf_S} in the "
                              "counterfactual swap"),
        ("Conformal", f"$\\alpha={float(arg['alpha']):g}$; boundary thresholds $\\epsilon\\in\\{{{eps}\\}}$, primary "
                      "$\\epsilon=0.05$"),
        ("HPD grid", f"{oe.G} cells on $[0,1]$"),
        ("Output-head models", "for the atom-aware experiments: the same encoder, optimizer and training settings, "
                               "with a continuous, an atom or a quantile output head"),
        ("Atom gate $\\tau$ (set C)",f"$\\tau\\in\\{{{taus}\\}}$"),
        ("CPCI (E4)", f"$r\\in\\{{{oe.R_GRID[0]:g},{oe.R_GRID[1]:g},\\dots,{oe.R_GRID[-1]:g}\\}}$"),
        ("E5", f"$r_0,r_1\\in{fmt_list(oe.E5_GRID)}$ with $r_0+r_1\\le0.5$; $\\Lambda={fmt_list(oe.LAMBDA_GRID[:2])[:-2]}"
               f",\\dots,{oe.LAMBDA_GRID[-1]:g}\\}}$"),
    ]
    import platform
    import numpy
    import scipy
    import torch
    hp.append(("Software", f"environment used to generate these tables: Python {platform.python_version()}, "
                           f"PyTorch {torch.__version__}, NumPy {numpy.__version__}, SciPy {scipy.__version__}"))
    L += [r"\begin{table}[h]", r"\centering", r"\small",
          r"\caption{Model, training and method settings as configured in the code.}", r"\label{tab:impl-train}",
          r"\begin{tabular}{>{\raggedright\arraybackslash}p{3cm}>{\raggedright\arraybackslash}p{11.5cm}}",
          r"\toprule", r"Setting & Value \\", r"\midrule"]
    L += [f"{a} & {b} \\\\" for a, b in hp]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "generated" / "implementation.tex").write_text("\n".join(L) + "\n", encoding="utf8")
    print("wrote paper/generated/implementation.tex")


# --------------------------------------------------------------------------- output and checks
def write():
    write_impl()
    fd = json.loads((ROOT / "figures" / "paper" / "figure_data.json").read_text())
    bounded_effect(fd)
    regime(fd)
    posthoc(fd)
    gate_m0()
    gate_o1(fd)
    structure(fd)
    era5(fd)
    shift_check()
    decision_check()
    tcn_check()
    latent_alignment()
    temporal_check()
    endpoint_cost()
    review_checks()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by experiments/scripts/make_numbers.py -- do not edit by hand.",
             "% Each macro: value, then source artefact and RESULTS_MANIFEST claim.", ""]
    claim = None
    for name, (tex, src, c) in sorted(MACROS.items(), key=lambda kv: (int((re.search(r"\d+", kv[1][2]) or re.search("0", "0")).group()),
                                                                      kv[0])):
        if c.split()[0] != claim:
            claim = c.split()[0]
            lines.append(f"\n% ---- {claim}")
        lines.append(rf"\newcommand{{\{name}}}{{{tex}}}  % {src} [{c}]")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"wrote {len(MACROS)} macros to {OUT.relative_to(ROOT.parent)}")


SUSPICIOUS = [
    (r"\d+(\.\d+)?\s*\\?%", "percentage"),
    (r"\bp\s*[=<>]\s*\d", "p-value"),
    (r"(?<![\w.])\d+\.\d+", "decimal"),
    (r"(?<![\w.])(?!1/2(?![\d.]))\d+\s*/\s*\d+(?![\w.])", "count k/n"),
    (r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|all|both)\b(?: of the \w+)?[\w\s-]{0,25}?"
     r"\b(?:datasets?|seeds?|pairs?|comparisons?|families|candidates)\b", "count in words"),
]


def check():
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", OUT.read_text(encoding="utf8")))
    sections = sorted((PAPER / "sections").rglob("*.tex"))
    tex_all = "\n".join(p.read_text(encoding="utf8") for p in sections + [PAPER / "main.tex"])
    used = set(re.findall(r"\\([A-Za-z]+)", tex_all))
    errors = 0
    # a used macro that starts with one of our prefixes but is not defined is almost surely a typo or a stale name
    prefixes = ("Eff", "PEff", "PostHoc", "Era", "PEra", "OOne", "POOne", "LambdaStar", "Struct", "Seeds", "GateB",
                "MZero", "PMZero", "TruncLog", "KumLog", "Resid", "ScaleRatio", "ScaleAmplified", "ThmThree",
                "Quantile", "Grouping", "RealScale", "NumDiscovery", "NumSyn", "NumReal", "Ladder", "Phase", "Conv",
                "QGauss", "QStar", "ResOpsExt", "PResOpsExt")
    for u in sorted(used - defined):
        if u.startswith(prefixes):
            print(f"ERROR undefined number macro \\{u}")
            errors += 1
    unused = sorted(defined - used)
    print(f"{len(defined)} macros defined, {len(defined & used)} used, {len(unused)} unused "
          f"(expected while Results are unwritten)")
    # hand-typed numbers
    n_sus = 0
    for p in sections:
        for i, line in enumerate(p.read_text(encoding="utf8").splitlines(), 1):
            body = line.split("%")[0] if not re.search(r"\\%", line) else re.sub(r"(?<!\\)%.*$", "", line)
            if line.rstrip().endswith("% num-ok") or not body.strip():
                continue
            body = re.sub(r"\\(label|ref|cref|Cref|cite\w*|input|includegraphics|texorpdfstring)(\[[^]]*\])?"
                          r"\{[^}]*\}", "", body)
            for pat, what in SUSPICIOUS:
                for m in re.finditer(pat, body, flags=re.I):
                    print(f"WARN {p.relative_to(PAPER)}:{i}: hand-typed {what} '{m.group()}'")
                    n_sus += 1
    print(f"{n_sus} suspicious hand-typed numbers; {errors} errors")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only run the checks")
    a = ap.parse_args()
    if not a.check:
        write()
    if not (PAPER / "main.tex").exists():
        print("manuscript sources not present: macros written, checks against the text skipped")
        return
    sys.exit(1 if check() else 0)


if __name__ == "__main__":
    main()
