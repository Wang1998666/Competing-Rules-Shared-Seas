"""Draw the RQ2 competition matrix and phased investment comparison."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

PLANS = ["Global", "EU", "BRI", "BRI_open"]
NAMES = dict(zip(PLANS, ["Global", "EU priority", "Closed BRI", "Open BRI"]))
COLORS = dict(zip(PLANS, ["#284F63", "#6061A0", "#12796F", "#AC743D"]))
MARKERS = dict(zip(PLANS, ["o", "s", "^", "D"]))
INK, MUTED, GRID = "#223C4A", "#546E79", "#DFE6E8"
RULE, DOMAIN = "#DBB67E", "#2F827B"
BUDGETS = np.array([5, 10, 15, 20])


def fonts():
    for path in Path("C:/Windows/Fonts").glob("times*.ttf"):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman","font.size":10.5,
                         "axes.labelcolor":INK,"text.color":INK,"axes.linewidth":.65,
                         "xtick.color":MUTED,"ytick.color":MUTED,
                         "pdf.fonttype":42,"ps.fonttype":42,
                         "mathtext.fontset":"custom","mathtext.rm":"Times New Roman",
                         "mathtext.it":"Times New Roman:italic","mathtext.bf":"Times New Roman:bold",
                         "mathtext.fallback":None})


def title(fig, x, y, letter, text):
    fig.text(x,y,letter,fontsize=13.4,fontweight="bold",color=INK,va="center")
    fig.text(x+.030,y,text,fontsize=13.3,fontweight="bold",color=INK,va="center")


def adorn(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_color("#A0AFB6")
    ax.set_axisbelow(True)
    ax.grid(axis="y",color=GRID,linewidth=.65)
    ax.tick_params(axis="both",length=2.7,pad=3)
    ax.set_xticks(range(4),[f"{x}%" for x in BUDGETS])
    ax.set_xlim(-.46,3.46)


def scorecard(ax, original):
    heat = LinearSegmentedColormap.from_list("score",["#F0F5F5","#8DB7BF","#315C70"])
    norm = Normalize(0,105)
    ax.set_xlim(-1.53,3.93); ax.set_ylim(4.12,-.80); ax.axis("off")
    for j, label in enumerate(["World","EU","BRI","Loss"]):
        ax.text(j+.5,-.31,label,va="center",ha="center",fontweight="bold",fontsize=10.4)
    for i, plan in enumerate(PLANS):
        ax.scatter(-1.39,i+.5,s=29,c=COLORS[plan],marker=MARKERS[plan],linewidth=0,clip_on=False,zorder=3)
        ax.text(-1.23,i+.5,NAMES[plan],ha="left",va="center",fontsize=10.7)
        for j,key in enumerate(["global_Mt","EU_Mt","BRI_Mt"]):
            val = original.loc[plan,key]
            ax.add_patch(Rectangle((j+.035,i+.048),.93,.904,fc=heat(norm(val)),ec="none"))
            ax.text(j+.5,i+.5,f"{val:.2f}",ha="center",va="center",fontsize=11,
                    color="white" if val>74 else INK)
        loss = original.loc[plan,"loss_pct"]
        ax.text(3.5,i+.5,f"{loss:.1f}%",ha="center",va="center",color=MUTED,fontsize=10.6)


def make_figure(old, staged, decomposition, args):
    fig=plt.figure(figsize=(7.65,5.54),facecolor="white")
    title(fig,.036,.960,"a","Static competition matrix")
    fig.text(.037,.916,"20% budget  ·  Mt CO₂e per year",fontsize=10.2,color=MUTED)
    title(fig,.566,.960,"b","Worldwide annual abatement")
    title(fig,.036,.485,"c","Port additions by stage")
    title(fig,.566,.485,"d","Closed BRI shortfall")
    scorecard(fig.add_axes([.035,.552,.454,.333]),old)

    b=fig.add_axes([.626,.612,.337,.275])
    c=fig.add_axes([.105,.139,.395,.276])
    d=fig.add_axes([.626,.139,.337,.276])
    for ax in (b,c,d):
        adorn(ax)

    x=np.arange(4)
    for idx,plan in enumerate(PLANS):
        frame=staged.loc[plan]
        b.plot(x,frame.global_Mt,color=COLORS[plan],lw=1.85,marker=MARKERS[plan],
               ms=5.5,mec="white",mew=.55,zorder=3+idx)
        c.bar(x+(idx-1.5)*.18,frame.n_new_ports,width=.156,fc=COLORS[plan],alpha=.91)
    b.set_ylim(25,110); b.set_yticks([30,50,70,90,110])
    b.set_ylabel("Mt CO₂e yr⁻¹",labelpad=5)
    fig.legend(b.lines, [NAMES[plan] for plan in PLANS], loc="center",
               bbox_to_anchor=(.50,.547),ncol=4,frameon=False,handlelength=1.75,
               fontsize=10.1,columnspacing=1.65,handletextpad=.47)

    c.set_ylim(0,72); c.yaxis.set_major_locator(MultipleLocator(20))
    c.set_ylabel("New ports",labelpad=5)
    c.set_xlabel("Cumulative budget share")

    obj = decomposition.regional_objective_pp.to_numpy()
    edge = decomposition.closed_domain_pp.to_numpy()
    total = obj+edge
    d.bar(x,obj,width=.55,fc=RULE,label="Regional objective")
    d.bar(x,edge,bottom=obj,width=.55,fc=DOMAIN,label="Closed investment domain")
    for j,tot in enumerate(total):
        d.text(j,tot+1.28,f"{tot:.1f}",ha="center",va="bottom",fontsize=9.7)
        d.text(j,obj[j]+edge[j]/2,f"{edge[j]:.1f}",ha="center",va="center",color="white",fontsize=10.2)
    d.set_ylim(0,68); d.set_yticks([0,20,40,60])
    d.set_ylabel("Loss versus staged global (%)",labelpad=5)
    d.set_xlabel("Cumulative budget share")
    d.legend(loc="center",bbox_to_anchor=(.794,.036),bbox_transform=fig.transFigure,ncol=2,frameon=False,fontsize=8.9,
             handlelength=1.0,handletextpad=.38,columnspacing=.86)

    for ext,dpi in [("pdf",350),("png",350)]:
        fig.savefig(args.output.with_suffix("."+ext),dpi=dpi,facecolor="white")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static",type=Path)
    parser.add_argument("staged",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    old=pd.read_csv(args.static).set_index("plan").loc[PLANS]
    results=pd.read_csv(args.staged)
    staged=results.query("mode == 'staged'").set_index(["plan","budget_pct"]).sort_index()
    assert len(staged)==16 and set(staged.index.get_level_values(1))==set(BUDGETS)
    glob=staged.loc["Global","global_Mt"]
    opening=staged.loc["BRI_open","global_Mt"]
    closed=staged.loc["BRI","global_Mt"]
    decomposition=pd.DataFrame({"regional_objective_pp":100*(glob-opening)/glob,
                                "closed_domain_pp":100*(opening-closed)/glob})
    assert (decomposition>0).all().all()
    np.testing.assert_allclose(decomposition.sum(axis=1),100*(glob-closed)/glob)
    expected=results.query("mode == 'fresh'").set_index("plan").loc[PLANS,"global_Mt"]
    achieved=staged.xs(20,level="budget_pct").loc[PLANS,"global_Mt"]
    np.testing.assert_allclose(expected,achieved,atol=.01,rtol=0)
    decomposition.to_csv(args.staged.parent/"bri_shortfall.csv")
    old[["global_Mt","EU_Mt","BRI_Mt","loss_pct"]].to_csv(args.staged.parent/"static_matrix.csv")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fonts()
    make_figure(old,staged,decomposition,args)


if __name__=="__main__":
    main()


