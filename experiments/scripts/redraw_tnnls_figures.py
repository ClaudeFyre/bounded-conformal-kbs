"""Publication-sized TNNLS graphics from the existing, unchanged result records.

No training, statistical re-selection or new evaluation is performed. All legacy
figure-data entries are preserved and checked; new entries document full-grid
values and the diagnostic diagram. Structure coordinates use the frozen,
published figure-data manifest rather than regenerating any data split.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import LogLocator, FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures/paper"
AUDIT = ROOT.parent / "output/tnnls_layout_20261027"
DS5 = ["syn_full", "syn_extreme", "seaice", "bucket", "resops"]
LABEL = {"syn_full":"Syn-Full", "syn_extreme":"Syn-Extreme", "seaice":"Sea ice", "bucket":"Bucket", "resops":"ResOpsUS", "syn_outatom":"Syn-OutAtom", "era5_cloud":"ERA5 cloud"}
FAMS = ["trunc_gaussian", "logit_normal", "beta", "kumaraswamy"]
FAM_LABEL = {"trunc_gaussian":"Truncated Gaussian", "logit_normal":"Logit-normal", "beta":"Beta", "kumaraswamy":"Kumaraswamy"}
BLUE, ORANGE, TEAL, PURPLE, INK = "#0072B2", "#D55E00", "#009E73", "#8B5BA7", "#253342"
FAMILY_COLORS = [BLUE, ORANGE, TEAL, PURPLE]
COL = dict(zip(DS5, [BLUE, TEAL, PURPLE, ORANGE, "#8A6D19"]))
MARK = dict(zip(DS5, ["o", "s", "D", "^", "v"]))
DATA, INPUTS, FONT_SIZES = {}, {}, {}


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update({"font.family":"Arial", "font.size":9.5, "axes.labelsize":9.5,
        "axes.titlesize":10, "axes.titleweight":"bold", "xtick.labelsize":9, "ytick.labelsize":9,
        "legend.fontsize":9, "mathtext.fontset":"stix", "axes.spines.top":False,
        "axes.spines.right":False, "axes.linewidth":.7, "axes.edgecolor":INK,
        "text.color":INK, "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK,
        "xtick.major.width":.6, "ytick.major.width":.6, "lines.linewidth":1.35,
        "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
        "savefig.bbox":None, "savefig.facecolor":"white"})


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for legend in fig.legends:
        for ax in fig.axes:
            if ax.xaxis.label.get_text() and legend.get_window_extent(renderer).overlaps(ax.xaxis.label.get_window_extent(renderer)):
                raise ValueError(f"Legend overlaps an x-axis label: {name}")
    from matplotlib.text import Text
    FONT_SIZES[name] = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(Text) if t.get_text() and t.get_visible()})
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", metadata={"Creator":"TNNLS figure renderer"} if ext == "pdf" else None)
    fig.savefig(OUT / f"{name}.png", dpi=400)
    plt.close(fig)


def clean(ax, axis="x"):
    ax.grid(axis=axis, color="#D8DEE5", lw=.5, zorder=0)
    ax.set_axisbelow(True)


def width(ds, fam, seed, key):
    r = read(RES / ds / f"{fam}__tn_hetero__raw__s{seed}.json")
    return next(c for c in r["conformal"] if abs(c["obs"]["eps"]-.05)<1e-9)["obs"][key]


def fig1_bounded_effect():
    fig = plt.figure(figsize=(7.16, 3.75))
    grid = fig.add_gridspec(2, 2, left=.145, right=.97, top=.88, bottom=.28,
                           height_ratios=[4,1.5], hspace=.6, wspace=.13)
    axes = [fig.add_subplot(grid[0, j]) for j in range(2)]
    lower = [fig.add_subplot(grid[1, j]) for j in range(2)]
    data = {}
    for ax, key, title in zip(axes, ["biw", "biw_clipped"], ["(a) Raw interval width", "(b) Feasible interval width"]):
        for i, ds in enumerate(DS5):
            for j, fam in enumerate(FAMS):
                rel = np.array([100*(1-width(ds, fam, s, key)/width(ds, "gaussian", s, key)) for s in range(10)])
                target = ax if i<4 else lower[0 if key=="biw" else 1]
                limits = (-35,40) if i<4 else (-155,42)
                if rel.min()<limits[0] or rel.max()>limits[1]:
                    raise ValueError("A seed point would be clipped in the width-effect plot")
                y = (i if i<4 else 0)+(j-1.5)*.16
                target.scatter(rel, y+np.linspace(-.038,.038,10), s=6, color=FAMILY_COLORS[j], alpha=.3, linewidths=0, zorder=2)
                target.errorbar(rel.mean(), y, xerr=rel.std(ddof=1), fmt=["o","s","D","^"][j], ms=4,
                    color=FAMILY_COLORS[j], capsize=2, elinewidth=1, markeredgecolor="white", markeredgewidth=.4, zorder=3)
                data[f"{ds}/{fam}/{key}"] = {"mean":float(rel.mean()), "sd":float(rel.std(ddof=1))}
        ax.axvline(0, color=INK, lw=.85)
        ax.set(title=title, yticks=range(4), yticklabels=[LABEL[d] for d in DS5[:4]] if key=="biw" else [], xlim=(-35,40), ylim=(3.55,-.55))
        ax.set_xticks([-25,0,25]);clean(ax)
    for j, ax in enumerate(lower):
        ax.axvline(0,color=INK,lw=.85)
        ax.set(yticks=[0],yticklabels=["ResOpsUS"] if j==0 else [],xlim=(-155,42),ylim=(.5,-.5),xticks=[-150,-100,-50,0,25])
        clean(ax)
    handles = [Line2D([],[],ls="",marker=m,color=c,ms=5,label=FAM_LABEL[f]) for f,c,m in zip(FAMS,FAMILY_COLORS,["o","s","D","^"])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52,.005), ncol=4, frameon=False, columnspacing=1.05, handletextpad=.4)
    fig.text(.56,.17,"Width reduction relative to Gaussian (%)",ha="center",fontsize=9.5)
    fig.text(.56,.115,"ResOpsUS uses separate linear axes; error bars: seed SD",ha="center",fontsize=9)
    DATA["fig1"] = data
    save(fig,"fig1_bounded_effect")


def fig2_posthoc_correction():
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    fig.subplots_adjust(left=.2,right=.98,top=.98,bottom=.34)
    data, allv = {}, []
    for ds in DS5:
        rows = read(RES / "counterfactual" / f"{ds}.json")
        per_seed = {r["seed"]:(r["W_a"],r["W_e"]) for r in rows}
        wa,we = map(np.array,zip(*per_seed.values()))
        ax.scatter(wa,we,s=22,color=COL[ds],marker=MARK[ds],edgecolor="white",lw=.35,label=LABEL[ds],zorder=3)
        allv.extend(wa);allv.extend(we)
        data[ds] = {"n":len(wa),"n_wider":int((we>wa).sum()),"median_log_ratio":float(np.median(np.log(we/wa)))}
    lo,hi = min(allv)*.82,max(allv)*1.25
    ax.set(xscale="log",yscale="log",xlim=(lo,hi),ylim=(lo,hi),xlabel=r"Original width $W_G$",ylabel=r"Truncated-posterior width $W_\star$")
    ax.plot([lo,hi],[lo,hi],ls="--",color=INK,lw=1,zorder=1)
    ax.set_aspect("equal",adjustable="box")
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        axis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:g}"))
        axis.set_minor_formatter(plt.NullFormatter())
    n_wider, n_all = sum(d["n_wider"] for d in data.values()), sum(d["n"] for d in data.values())
    ax.text(.04,.97,f"{n_wider}/{n_all} pairs widen",transform=ax.transAxes,ha="left",va="top",fontsize=9.5,fontweight="bold")
    clean(ax,"both")
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,.005),ncol=2,frameon=False,columnspacing=1,handletextpad=.25)
    DATA["fig2"] = data
    save(fig,"fig2_posthoc_correction")


def fig3_decomposition():
    tc = read(RES / "theory_check.json")
    fig,axes = plt.subplots(1,2,figsize=(7.16,3.05),sharex=True,sharey=True)
    fig.subplots_adjust(left=.145,right=.985,top=.84,bottom=.29,wspace=.14)
    data = {"truncation":{},"kumaraswamy_vs_gaussian":{}}
    for i,ds in enumerate(DS5):
        rq = [r["log_Rq"] for r in tc[ds]];rs = [r["log_Rsigma"] for r in tc[ds]]
        lq,ls = [],[]
        for seed in range(10):
            g = read(RES / ds / f"gaussian__tn_hetero__raw__s{seed}.json")
            k = read(RES / ds / f"kumaraswamy__tn_hetero__raw__s{seed}.json")
            cg,ck = [next(c for c in v["conformal"] if abs(c["obs"]["eps"]-.05)<1e-9)["obs"] for v in (g,k)]
            lq.append(math.log(cg["q_boundary"]/ck["q_boundary"]))
            ls.append(math.log(cg["biw"]/ck["biw"])-lq[-1])
        for ax,a,b,entry in zip(axes,[rq,lq],[rs,ls],["truncation","kumaraswamy_vs_gaussian"]):
            for off,values,col in [(-.17,a,BLUE),(.17,b,ORANGE)]:
                ax.barh(i+off,np.median(values),height=.28,color=col,alpha=.85,zorder=2)
                ax.scatter(values,i+off+np.linspace(-.055,.055,len(values)),s=7,color=INK,alpha=.65,zorder=3,linewidths=0)
            data[entry][ds] = {"median_log_Rq":float(np.median(a)),"median_log_Rsigma":float(np.median(b))}
    for ax,title in zip(axes,["(a) Truncated / original posterior","(b) Gaussian / Kumaraswamy"]):
        ax.axvline(0,color=INK,lw=.8);ax.set(title=title,yticks=range(5),yticklabels=[LABEL[d] for d in DS5],ylim=(4.55,-.55),xlim=(-.78,.9),xlabel="Log ratio")
        clean(ax)
    fig.legend(handles=[Rectangle((0,0),1,1,color=BLUE,label=r"Quantile: $\log R_q$"),Rectangle((0,0),1,1,color=ORANGE,label=r"Scale: $\log R_\sigma$"),Line2D([],[],ls="",marker="o",color=INK,ms=3,label="Individual seed")],loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.55,.005))
    DATA["fig3"] = data
    save(fig,"fig3_quantile_scale")


def fig4_theorem3_conditions():
    tc = read(RES / "theory_check.json")
    fig,ax = plt.subplots(figsize=(3.5,3.45))
    fig.subplots_adjust(left=.2,right=.97,top=.96,bottom=.37)
    data = {}
    for ds in DS5:
        sk = [r["skew"] for r in tc[ds]];cv = [r["ratio_cv"] for r in tc[ds]]
        ax.scatter(sk,cv,s=22,color=COL[ds],marker=MARK[ds],edgecolor="white",linewidth=.3,label=LABEL[ds])
        data[ds] = {"median_skew":float(np.median(sk)),"median_ratio_cv":float(np.median(cv)),"wider":sum(r["log_Wratio"]>0 for r in tc[ds]),"n":len(tc[ds])}
    ax.set(xscale="symlog",xlabel="Normalized-residual skewness",ylabel="Scale-ratio coefficient of variation")
    ax.set_xscale("symlog",linthresh=.1)
    ax.axvline(0,color=INK,lw=.65,ls=":");ax.axhline(0,color=INK,lw=.65,ls=":")
    clean(ax,"both")
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.54,.005),ncol=2,frameon=False,handletextpad=.3,columnspacing=1)
    DATA["fig4"] = data
    save(fig,"fig4_theorem3_conditions")


def fig5_lambda_curves():
    lams = np.arange(11)*.01
    o1,era = read(RES/"o1.json"),read(RES/"era5_confirm.json")
    panels = [("seaice",[r for r in o1 if r["dataset"]=="seaice"],["A","B","C","E4"]),("syn_outatom",[r for r in o1 if r["dataset"]=="syn_outatom"],["A","B","C","E4"]),("era5_cloud",era,["A","B","C","E4","E5"])]
    names = {"A":"Continuous HPD","B":"Free-atom HPD","C":"Gated-atom HPD","E4":"CPCI","E5":"Two-boundary (E5)"}
    colors,styles = dict(zip(names,[BLUE,ORANGE,TEAL,PURPLE,INK])),dict(zip(names,["-","--","-.",":",(0,(5,2,1,2))]))
    fig,axes = plt.subplots(1,3,figsize=(7.16,3.0))
    fig.subplots_adjust(left=.09,right=.97,top=.87,bottom=.41,wspace=.3)
    data = {}
    for ax,(ds,rows,meths),letter in zip(axes,panels,"abc"):
        data[ds] = {}
        for method in meths:
            mu = st.mean(r["methods"][method]["mu"] for r in rows)
            na = st.mean(r["methods"][method]["n_atom"] for r in rows)
            ax.plot(lams,mu+lams*na,color=colors[method],ls=styles[method],label=names[method])
            cross = [(r["methods"]["A"]["mu"]-r["methods"][method]["mu"])/r["methods"][method]["n_atom"] for r in rows if method!="A" and r["methods"][method]["n_atom"]>0]
            data[ds][method] = {"mu":mu,"n_atom":na,"lambda_star_median":float(np.median(cross)) if cross else None}
            if cross and 0<np.median(cross)<.1:
                ax.axvline(np.median(cross),color=colors[method],ls=":",lw=.75,alpha=.6)
        ax.set(title=f"({letter}) {LABEL[ds]}",xlabel=r"Atom cost $\lambda$",xlim=(0,.1),xticks=[0,.05,.1])
        clean(ax,"y")
    axes[0].set_ylabel(r"Mean ambiguity $A_\lambda$")
    fig.legend(handles=[Line2D([],[],color=colors[m],ls=styles[m],label=names[m]) for m in names],loc="lower center",bbox_to_anchor=(.53,.055),ncol=3,frameon=False,columnspacing=1.4)
    fig.text(.53,.02,"Dotted vertical guides: median per-seed crossover",ha="center",fontsize=9)
    DATA["fig5"] = data
    save(fig,"fig5_lambda_curves")


def fig_phase():
    ps,pl = read(RES/"phase_summary.json"),read(RES/"phase_placement.json")
    pb,cs = [.1,.2,.4,.6,.8],[.005,.01,.02,.05,.1,.2]
    placement = {ds:float(np.median([r["median_sigma"] for r in pl[ds]])) for ds in DS5}
    fig = plt.figure(figsize=(7.16,5.25))
    grid = fig.add_gridspec(3,2,left=.1,right=.98,top=.94,bottom=.19,hspace=.67,wspace=.23,height_ratios=[1,1,.72])
    data = {"placement_C":placement,"cells":{}}
    tint = {1:"#D3E8F4",-1:"#F7DFD2",0:"#ECEFF1"}
    for k,fam in enumerate(FAMS):
        ax = fig.add_subplot(grid[k//2,k%2])
        data["cells"][fam] = []
        for i,c in enumerate(cs):
            for j,p in enumerate(pb):
                cell = ps[fam][i][j+1];sg = cell["sign"]
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=tint[sg],edgecolor="white",lw=1))
                ax.text(j,i,{1:"+",-1:"−",0:"~"}[sg],ha="center",va="center",fontsize=10.5,color=INK)
                data["cells"][fam].append({"C":c,"p_b":p,"sign":sg,"rel":cell["rel"]})
        ax.set(xlim=(-.5,4.5),ylim=(-.5,5.5),xticks=range(5),xticklabels=[f"{p:g}" for p in pb],yticks=range(6),yticklabels=[f"{c:g}" for c in cs],title=f"({chr(97+k)}) {FAM_LABEL[fam]}")
        ax.tick_params(length=0,pad=3)
        ax.set_xlabel(r"Boundary fraction $p_b$",labelpad=2)
        if k%2==0:ax.set_ylabel(r"Scale $C$")
        for spine in ax.spines.values():spine.set_visible(False)
    tableax = fig.add_subplot(grid[2,:]);tableax.axis("off")
    columns = ["Observed dataset","Median $C$","Trunc. G.","Logit-N.","Beta","Kumar."]
    celltext = [[LABEL[d],f"{placement[d]:.4f}"]+["+" if DATA["fig1"][f"{d}/{f}/biw"]["mean"]>0 else "−" for f in FAMS] for d in DS5]
    table = tableax.table(cellText=celltext,colLabels=columns,cellLoc="center",colWidths=[.25,.15,.15,.15,.15,.15],bbox=[0,-.1,1,1.2])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (row,col),cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row==0:cell.set_facecolor("#EAF0F5");cell.set_text_props(weight="bold")
        elif col>=2:cell.set_text_props(color=BLUE if cell.get_text().get_text()=="+" else ORANGE,weight="bold")
    fig.legend(handles=[Rectangle((0,0),1,1,facecolor=tint[1],label="+  Bounded narrower"),Rectangle((0,0),1,1,facecolor=tint[-1],label="−  Gaussian narrower"),Rectangle((0,0),1,1,facecolor=tint[0],label="~  Sign unstable")],loc="lower center",bbox_to_anchor=(.54,.055),ncol=3,frameon=False,columnspacing=1)
    fig.text(.54,.025,"Cells: sign agrees in at least 4/5 seeds; table: post hoc observed signs",ha="center",fontsize=9)
    DATA["fig_phase"] = data
    save(fig,"fig_phase_diagram")


def fig_phase_full():
    ps = read(RES/"phase_summary.json")
    pb,cs = [0,.1,.2,.4,.6,.8],[.005,.01,.02,.05,.1,.2]
    cmap = LinearSegmentedColormap.from_list("bounded_diverging",["#A84723","#FFFFFF",BLUE])
    norm = TwoSlopeNorm(vmin=-60,vcenter=0,vmax=60)
    fig,axes = plt.subplots(2,2,figsize=(7.16,5.65))
    fig.subplots_adjust(left=.1,right=.98,top=.94,bottom=.27,wspace=.24,hspace=.48)
    for k,(ax,fam) in enumerate(zip(axes.flat,FAMS)):
        z = np.array([[np.nan if c is None else c["rel"] for c in row] for row in ps[fam]])
        im = ax.pcolormesh(np.arange(7)-.5,np.arange(7)-.5,z,cmap=cmap,norm=norm,edgecolors="#E4E8EC",linewidth=.6)
        for i,row in enumerate(ps[fam]):
            for j,c in enumerate(row):
                if c is None:
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor="#F3F3F3",edgecolor="#D7D7D7",hatch="///",lw=.4))
                    text,color = "NA",INK
                else:
                    number = f"{c['rel']:+.2f}" if abs(c["rel"])<1 else f"{c['rel']:+.1f}"
                    text,color = number+("~" if c["sign"]==0 else ""),"white" if abs(c["rel"])>34 else INK
                ax.text(j,i,text,ha="center",va="center",fontsize=9,color=color)
        ax.set(xticks=range(6),xticklabels=[f"{p:g}" for p in pb],yticks=range(6),yticklabels=[f"{c:g}" for c in cs],xlabel=r"Boundary fraction $p_b$",title=f"({chr(97+k)}) {FAM_LABEL[fam]}")
        if k%2==0:ax.set_ylabel(r"Scale $C$")
        ax.tick_params(length=0,pad=3)
        for spine in ax.spines.values():spine.set_visible(False)
    cax = fig.add_axes([.24,.12,.55,.022])
    cb = fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[-60,-30,0,30,60]);cb.set_label("Raw-width reduction relative to Gaussian (%)",fontsize=9.5)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.solids.set_linewidth(1.0)
    fig.text(.53,.018,r"NA: undefined boundary metric; ~: sign not consistent across seeds",ha="center",fontsize=9)
    DATA["fig_phase_full"] = ps
    save(fig,"fig_phase_full")


def fig6_structure_map():
    pts = DATA["fig6"]
    fig,ax = plt.subplots(figsize=(7.16,3.6))
    fig.subplots_adjust(left=.1,right=.985,top=.97,bottom=.25)
    x = np.linspace(0,.5,501);cap = 1-2*x
    ax.fill_between(x,cap,1.03,color="#F0F1F3",lw=0)
    ax.fill_between(x,0,np.minimum(.2,cap),color="#EAE8E3",lw=0)
    valid = x>=.05
    ax.fill_between(x,np.minimum(.2,cap),cap,where=valid,color="#E1EEF6",lw=0)
    ax.plot(x,cap,color="#AAB3BD",lw=.8)
    ax.plot([.05,.05],[.2,.9],ls="--",color="#8796A6",lw=.8)
    ax.plot([0,.4],[.2,.2],ls="--",color="#8796A6",lw=.8)
    ax.text(.32,.84,"Infeasible probability combinations",ha="center",fontsize=9,color="#65717D")
    ax.text(.19,.43,"Mixed outcomes",ha="center",fontsize=10,color=BLUE)
    ax.text(.245,.135,"Endpoint-dominated",ha="center",fontsize=9)
    ax.annotate("One weak endpoint",xy=(.025,.46),xytext=(.075,.61),fontsize=9,arrowprops={"arrowstyle":"-","color":"#8796A6","lw":.6})
    roles = {"Sea ice":"discovery","ResOpsUS":"discovery","Snow_cci CryoClim":"candidate","MODIS SCFG":"candidate","MODIS SCFV":"candidate","ERA5 cloud":"held-out"}
    offsets = {"Sea ice":(7,8,"left"),"ResOpsUS":(7,-6,"left"),"Snow_cci CryoClim":(7,6,"left"),"MODIS SCFG":(7,6,"left"),"MODIS SCFV":(7,8,"left"),"ERA5 cloud":(7,-2,"left")}
    styles = {"discovery":dict(marker="o",s=38,color=BLUE,edgecolor="white"),"candidate":dict(marker="s",s=35,facecolor="white",edgecolor=INK),"held-out":dict(marker="*",s=95,color=ORANGE,edgecolor="white")}
    for name,p in pts.items():
        pos = min(p["p0"],p["p1"]),p["interior"]
        ax.scatter(*pos,**styles[roles[name]],linewidths=.9,zorder=3)
        dx,dy,ha = offsets[name]
        ax.annotate(name,pos,xytext=(dx,dy),textcoords="offset points",ha=ha,va="center",fontsize=9)
    ax.set(xlim=(-.008,.5),ylim=(0,1.04),xlabel=r"Less frequent endpoint: $\min\{P(Y=0), P(Y=1)\}$",ylabel=r"Interior probability $P(0<Y<1)$")
    handles = [Line2D([],[],ls="",marker="o",color=BLUE,label="Discovery data"),Line2D([],[],ls="",marker="s",markerfacecolor="white",color=INK,label="Screened candidate (failed)"),Line2D([],[],ls="",marker="*",ms=9,color=ORANGE,label="Held-out confirmation")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.54,.02),ncol=3,frameon=False,columnspacing=1.2)
    save(fig,"fig6_structure_map")


def fig_diagnostic_sequence():
    fig,ax = plt.subplots(figsize=(3.5,4.2));ax.set(xlim=(0,3.5),ylim=(0,4.2));ax.axis("off")
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    centers = [3.81,3.17,2.53,1.89,1.17]
    questions = ["Physical bound known\nindependently of the sample?","Exact endpoint\nvalues present?","Endpoint-dominated\n(little interior mass)?","Both endpoints and\ninterior substantial?",r"Report $\bar A_\lambda$ and $\lambda^\star$."+"\n"+r"Is $\lambda^\star$ large for"+"\nrelevant endpoint costs?"]
    answers = ["Framework does\nnot apply","Continuous set:\nclip and measure\non [0, 1]","Classification +\nconditional\nregression","Continuous HPD\nis usually\nsufficient","Continuous\nHPD set"]
    xq,xa,wq,wa = .94,2.78,1.77,1.3
    heights = [.49,.43,.49,.49,.62]
    def node(x,y,w,h,text,fill):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.015,rounding_size=.045",facecolor=fill,edgecolor="#8D9BA9",lw=.75))
        ax.text(x,y,text,ha="center",va="center",fontsize=9,linespacing=1.12)
    def arrow(a,b,label,offset=(0,.08)):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=.8,color="#637589"))
        ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,fontsize=9,ha="center",va="center",bbox=dict(fc="white",ec="none",pad=.2))
    for i,(y,h,q,a) in enumerate(zip(centers,heights,questions,answers)):
        node(xq,y,wq,h,q,"#EAF1F7");node(xa,y,wa,h,a,"white")
        arrow((xq+wq/2+.02,y),(xa-wa/2-.02,y),"yes" if i==2 else "no")
        if i<4:arrow((xq,y-h/2-.025),(xq,centers[i+1]+heights[i+1]/2+.025),"no" if i==2 else "yes",(.15,0))
    node(xq,.32,wq,.44,"Atom-aware routing\nworth considering","#DBEFE5")
    arrow((xq,centers[-1]-heights[-1]/2-.025),(xq,.565),"yes",(.15,0))
    DATA["fig_diagnostic_sequence"] = {"source":"paper/sections/discussion.tex", "logic_changed":False}
    save(fig,"fig_diagnostic_sequence")


def fig_study_overview():
    from tnnls_early_figures import overview
    overview(save)
    DATA["fig_study_overview"] = {"role":"conceptual schematic", "experimental_data":False}


def fig_matched_protocol():
    from tnnls_early_figures import matched_protocol
    matched_protocol(save)
    DATA["fig_matched_protocol"] = {"role":"matched-comparison workflow", "experimental_data":False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only",nargs="+")
    args = parser.parse_args()
    style();OUT.mkdir(parents=True,exist_ok=True)
    previous = json.loads((OUT/"figure_data.json").read_text(encoding="utf-8"))
    DATA.update(previous)
    if args.only and (AUDIT/"redraw_manifest.json").exists():
        existing = json.loads((AUDIT/"redraw_manifest.json").read_text(encoding="utf-8"))
        INPUTS.update(existing["input_sha256"])
        FONT_SIZES.update(existing["font_sizes_pt"])
    funcs = [fig1_bounded_effect,fig2_posthoc_correction,fig3_decomposition,fig4_theorem3_conditions,fig5_lambda_curves,fig6_structure_map,fig_phase,fig_phase_full,fig_diagnostic_sequence,fig_study_overview,fig_matched_protocol]
    for fn in funcs:
        if not args.only or fn.__name__ in args.only:
            fn();print(f"Redrawn: {fn.__name__}",flush=True)
    for key,value in previous.items():
        if key not in ("fig_phase_full","fig_diagnostic_sequence"):
            if DATA[key]!=value:raise ValueError(f"Existing figure statistics changed: {key}")
    (OUT/"figure_data.json").write_text(json.dumps(DATA,indent=1),encoding="utf-8")
    AUDIT.mkdir(parents=True,exist_ok=True)
    manifest = {"date":"2026-10-27","existing_figure_statistics_identical":True,"training_executed":False,
        "input_sha256":INPUTS,"font_sizes_pt":FONT_SIZES,
        "frozen_structure_coordinates":previous["fig6"],
        "output_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.pdf")}}
    (AUDIT/"redraw_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":main()
