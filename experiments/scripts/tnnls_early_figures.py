"""Conceptual figures for the early pages; no experimental data are drawn."""
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, TEAL, INK = "#0072B2", "#D55E00", "#009E73", "#253342"


def overview(save):
    fig = plt.figure(figsize=(7.16, 2.22))
    panels = [.035, .365, .695]
    titles = ["(a) Latent support", "(b) Predictive distribution", "(c) Reported set and size"]
    for left,title in zip(panels,titles):
        fig.text(left+.135,.94,title,ha="center",va="center",fontsize=10,fontweight="bold")
    # Illustrative density shapes, on an unlabeled vertical scale.
    ax = fig.add_axes([.055,.39,.245,.43])
    h = np.linspace(-.4,1.4,401)
    gaussian = np.exp(-.5*((h-.48)/.34)**2)
    inside = (h>=0)&(h<=1)
    ax.axvspan(0,1,color="#E6F0F7")
    ax.fill_between(h,gaussian,where=~inside,color=ORANGE,alpha=.3)
    ax.plot(h,gaussian,color=ORANGE,ls="--",lw=1.3)
    mass = .5*(math.erf((1-.48)/(.34*math.sqrt(2)))-math.erf(-.48/(.34*math.sqrt(2))))
    ax.plot(h[inside],gaussian[inside]/mass,color=BLUE,lw=1.5)
    ax.set(xlim=(-.4,1.4),ylim=(0,1.48),xticks=[0,1],yticks=[])
    ax.text(1.31,.06,r"$h$",fontsize=9)
    ax.text(.48,1.3,r"$\mathcal{M}=[0,1]$",ha="center",fontsize=9)
    ax.text(-.3,1.3,r"$q^G$",ha="center",fontsize=9,color=ORANGE)
    ax.text(1.21,1.3,r"$q^\star$",ha="center",fontsize=9,color=BLUE)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x",length=2)
    fig.text(.17,.17,"Constrain the latent family\nor restrict a trained posterior",ha="center",fontsize=9,linespacing=1.15)
    fig.text(.17,.025,"Physical coordinates required",ha="center",fontsize=9,color=BLUE)
    ax = fig.add_axes([.385,.39,.245,.43])
    y = np.linspace(0,1,201)
    density = .22+3.4*y*(1-y)
    ax.fill_between(y,density,color="#DDF0E8")
    ax.plot(y,density,color=TEAL,lw=1.5)
    for point in (0,1):
        ax.annotate("",xy=(point,1.35),xytext=(point,0),arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=1.5))
    ax.set(xlim=(-.1,1.1),ylim=(0,1.65),xticks=[0,1],yticks=[])
    ax.text(1.055,.06,r"$y$",fontsize=9)
    ax.text(0,1.46,r"$\pi_0\delta_0$",ha="center",fontsize=9)
    ax.text(1,1.46,r"$\pi_1\delta_1$",ha="center",fontsize=9)
    ax.text(.5,.5,r"$\pi_c f_c(y)$",ha="center",fontsize=9)
    ax.spines["left"].set_visible(False);ax.tick_params(axis="x",length=2)
    fig.text(.5,.17,"Model endpoint masses\nand the continuous interior",ha="center",fontsize=9,linespacing=1.15)
    fig.text(.5,.025,"Arrows denote discrete masses",ha="center",fontsize=9,color=TEAL)
    ax = fig.add_axes([.715,.39,.245,.43])
    ax.plot([.2,.72],[.55,.55],color=BLUE,lw=6,solid_capstyle="butt")
    ax.scatter([0,1],[.55,.55],s=32,color=ORANGE,zorder=3)
    ax.set(xlim=(-.1,1.1),ylim=(0,1.2),xticks=[0,1],yticks=[])
    ax.text(.46,.15,r"$C=I\cup Z$",ha="center",fontsize=10)
    ax.text(.46,.92,"interval + isolated endpoints",ha="center",fontsize=9)
    for s in ("left","top","right"):ax.spines[s].set_visible(False)
    ax.tick_params(axis="x",length=2)
    fig.text(.83,.17,"Report feasible width\nand endpoint-inclusive size",ha="center",fontsize=9,linespacing=1.15)
    fig.text(.83,.025,r"$\bar A_\lambda=\mu+\lambda N_{\mathrm{end}}$",ha="center",fontsize=10,color=BLUE)
    for x in (.337,.667):
        fig.add_artist(plt.Line2D([x,x],[.04,.97],transform=fig.transFigure,color="#D6DEE5",lw=.7))
    save(fig,"fig_study_overview")


def matched_protocol(save):
    fig,ax = plt.subplots(figsize=(3.5,2.82))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,3.5),ylim=(0,2.82));ax.axis("off")
    def box(x,y,w,h,text,fill="#EDF3F8",edge="#94A4B4"):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.015,rounding_size=.04",fc=fill,ec=edge,lw=.7))
        ax.text(x,y,text,ha="center",va="center",fontsize=9,linespacing=1.1)
    def arrow(x,y0,y1):
        ax.add_patch(FancyArrowPatch((x,y0),(x,y1),arrowstyle="-|>",mutation_scale=8,color="#667B8D",lw=.8))
    box(1.75,2.58,3.22,.32,"Freeze the trained encoder and decoder")
    xs = [.86,2.64]
    for x,text,fill in zip(xs,[r"Original $q^G(h\mid x)$",r"Restricted $q^\star(h\mid x)$"],["#FAE9DE","#E3EFF7"]):
        arrow(x,2.4,2.24);box(x,2.06,1.53,.32,text,fill)
    for x,text in zip(xs,[r"$\hat y_G(x),\ \hat\sigma_G(x)$",r"$\hat y_\star(x),\ \hat\sigma_\star(x)$"]):
        arrow(x,1.88,1.74);box(x,1.57,1.53,.29,text,"white")
    for x in xs:arrow(x,1.4,1.3)
    box(1.75,1.0,3.22,.55,r"Calibrate $S_G$ and $S_\star$ separately"+"\nSame calibration units and groups","#E5F1EB")
    for x,text in zip(xs,[r"$C_G(x)$",r"$C_\star(x)$"]):
        arrow(x,.71,.59);box(x,.43,1.53,.25,text,"white")
    ax.text(1.75,.12,"Common test units: compare coverage and size",ha="center",va="center",fontsize=9)
    save(fig,"fig_matched_protocol")
