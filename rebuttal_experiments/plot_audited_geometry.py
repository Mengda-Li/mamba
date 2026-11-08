"""Descriptive full-rank input angles and active-frame audits, from saved tables.

No model runs, sign changes, nullspace rotations or new bootstrap are performed.
Full-U Q/K eigenangle figures are intentionally not generated here.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

MODELS=(("gpt2",11,"GPT-2","GPT-2"),("retnet",23,"RetNet-1.3B","RetNet"),
        ("mamba2",63,"Codestral-7B (HF)","Codestral\n7B (HF)"),
        ("mamba13",47,"Mamba2-1.3B","Mamba2\n1.3B"))
METRICS=(("histogram_kl_uniform","Histogram KL"),("ks","KS distance"),
         ("angular_w1_midpoint_grid","Angular W1 to grid (rad)"))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with gzip.open(path,"rt",newline="") as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--summary",type=Path,nargs="+")
    group.add_argument("--plotting-rows",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=read(args.plotting_rows) if args.plotting_rows else [r for path in args.summary for r in read(path)]
    sources={str(path.name)+"_"+sha(path)[:12]:sha(path) for path in (args.summary or [args.plotting_rows])}
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":8,"axes.titlesize":8,"axes.labelsize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    selected=[];created=[]
    def one(**filters):
        found=[r for r in rows if all(str(r.get(k,""))==str(v) for k,v in filters.items())]
        if len(found)!=1:raise ValueError(f"Expected one configuration: {filters}; found{len(found)}")
        row=found[0]
        if int(row["n_documents"])!=32:raise ValueError("Only complete32 primary summaries may enter")
        selected.append(row)
        return row
    def save(fig,name,caption):
        for suffix in ("pdf","svg","png"):
            path=args.output/(name+"."+suffix);fig.savefig(path,dpi=220);created.append(path.name)
        (args.output/(name+".caption.md")).write_text(caption+"\n")
        plt.close(fig)

    # These eligible ranges were fixed by measured full numerical row rank,
    # not by the observed angular distances. The input matrix is X, not Q/K.
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.75))
    colors=("#2768ad","#cf6c24","#54924a","#9064b0")
    handles=[]
    for (model,layer,label,_),color in zip(MODELS,colors):
        lengths=(64,128,256,512) if model=="gpt2" else (64,128,256,512,1024)
        for ax,(metric,ylabel) in zip(axes,METRICS):
            measured=[one(model=model,layer=layer,length=length,branch="X",space="token_left",
                active_rank="",metric=metric,sign_convention="solver_native") for length in lengths]
            if any(float(r["structural_completion_columns"])!=0 or float(r["numerical_nullspace_columns"])!=0 for r in measured):
                raise ValueError("Full-U input angle plot contains a structural/numerical nullspace")
            y,lo,hi=[[float(r[k]) for r in measured] for k in ("estimate","ci_low","ci_high")]
            line,=ax.plot(lengths,y,"-o",color=color,linewidth=1.05,markersize=3,label=label)
            ax.fill_between(lengths,lo,hi,color=color,alpha=.13,linewidth=0)
            if metric==METRICS[0][0]:handles.append(line)
            ax.set_xscale("log",base=2);ax.set_xticks([64,128,256,512,1024],["64","128","256","512","1024"],rotation=35)
            ax.set_xlabel("Context tokens");ax.set_ylabel(ylabel);ax.grid(alpha=.17,linewidth=.4);ax.tick_params(pad=1.5)
    fig.suptitle("Final-layer input X: descriptive eigen-angle distances",fontsize=9,y=.995)
    fig.legend(handles,[x[2] for x in MODELS],ncol=4,loc="upper center",bbox_to_anchor=(.5,.935),frameon=False)
    fig.text(.5,.008,"32 documents; numerically full-row-rank X only. Shading: 95% document-bootstrap intervals.",ha="center",fontsize=6.4)
    fig.subplots_adjust(left=.095,right=.99,top=.78,bottom=.24,wspace=.47)
    save(fig,"input_frame_angles",
        "Descriptive final-layer input-frame eigen-angle distances on32 fixed document identities. X is the actual normalized mixer input. GPT-2 includes64–512 tokens; RetNet-1.3B, Codestral-7B(HF), and Mamba2-1.3B include64–1024. Every retained X has full numerical row rank in every document; GPT-2 at1024 is excluded because it has256 structural completion columns. Primary solver-native signs are unchanged. The implemented branch is angle modulo[0,2pi), with histogram KL/KS and W1 to a deterministic uniform midpoint grid; the cited NeurIPS code instead uses np.angle+pi and W1 against an independently sampled uniform reference, so the metrics are not an exact protocol replication. Shading is the saved95% document-bootstrap interval. These distances are descriptive and do not validate a joint Haar law; decreasing distance can reflect finite-size behavior and does not override the active-frame anisotropy shown separately. Codestral-7B is the fixed Transformers4.57.1 global-gated-RMSNorm implementation, not claimed equivalent to the official grouped-norm runtime. No full-U Q/K completion is plotted.")

    fig,axes=plt.subplots(1,2,figsize=(6.6,3.25));x=np.arange(4);width=.31
    coherence=[];overlap=[]
    for model,layer,_,_ in MODELS:
        coherence.append([one(model=model,layer=layer,length=1024,branch=branch,space="token_left",active_rank=16,
            metric="coherence",sign_convention="subspace_or_sign_invariant") for branch in ("Q","K")])
        overlap.append([one(model=model,layer=layer,length=1024,branch="Q_K",space="token_left",active_rank=16,
            metric=metric,sign_convention="subspace_or_sign_invariant") for metric in
            ("mean_squared_canonical_correlation","ordered_squared_overlap_mean")])
    ax=axes[0]
    ref=coherence[0][0]
    # All four models use the same Haar-Stiefel(1024,16) reference ensemble.
    for group in coherence:
        for row in group:
            for key in ("haar_reference_mean","haar_reference_q05","haar_reference_q95"):
                if abs(float(row[key])-float(ref[key]))>1e-12:raise ValueError("Mismatched frame null ensembles")
    ax.axhspan(float(ref["haar_reference_q05"]),float(ref["haar_reference_q95"]),color="#777777",alpha=.18,label="Haar frames: 5–95%")
    ax.axhline(float(ref["haar_reference_mean"]),color="#555555",linestyle="--",linewidth=.8)
    for j,(label,color) in enumerate((("Q (=C)","#2768ad"),("K (=B)","#cf6c24"))):
        values=np.array([float(group[j]["estimate"]) for group in coherence])
        low=np.array([float(group[j]["ci_low"]) for group in coherence]);high=np.array([float(group[j]["ci_high"]) for group in coherence])
        ax.bar(x+(j-.5)*width,values,width,color=color,alpha=.88,label=label,yerr=[values-low,high-values],
            error_kw={"elinewidth":.8,"capsize":2,"capthick":.8})
    ax.set_ylim(0,14.5);ax.set_ylabel("Active-frame coherence");ax.set_title("Token localization of active Q/K frames")
    ax.legend(frameon=False,fontsize=6.1,loc="upper left")
    ax=axes[1]
    for j,(label,color) in enumerate((("Subspace overlap","#2768ad"),("Ordered-basis overlap","#cf6c24"))):
        values=np.array([float(group[j]["estimate"]) for group in overlap])
        low=np.array([float(group[j]["ci_low"]) for group in overlap]);high=np.array([float(group[j]["ci_high"]) for group in overlap])
        bottom=.00035
        ax.bar(x+(j-.5)*width,values-bottom,width,bottom=bottom,color=color,alpha=.88,label=label,yerr=[values-low,high-values],
            error_kw={"elinewidth":.8,"capsize":2,"capthick":.8})
        r=overlap[0][j]
        ax.axhspan(float(r["haar_reference_q05"]),float(r["haar_reference_q95"]),color=color,alpha=.09)
        ax.axhline(float(r["haar_reference_mean"]),color=color,linestyle="--",linewidth=.8)
    ax.axhline(1,color="#333333",linestyle=":",linewidth=.9)
    ax.text(3.48,1.025,"Exact common frame: 1",fontsize=6.4,ha="right",va="bottom")
    ax.set_yscale("log");ax.set_ylim(.00035,1.5);ax.set_ylabel("Token-frame overlap (log scale)");ax.set_title("Subspace and ordered-basis overlap")
    ax.legend(frameon=False,fontsize=6.1,loc="lower left",bbox_to_anchor=(0,.29))
    for ax in axes:
        ax.set_xticks(x,[v[3] for v in MODELS]);ax.grid(axis="y",alpha=.14,linewidth=.4);ax.set_axisbelow(True)
    fig.suptitle("Final-layer active-frame audit: length 1024, rank 16",fontsize=9,y=.995)
    fig.text(.5,.008,"Bars ±95% document-bootstrap CI. Dashed lines/bands: Haar means/5–95% frame variation.",ha="center",fontsize=6.1)
    fig.subplots_adjust(left=.08,right=.99,top=.85,bottom=.205,wspace=.34)
    save(fig,"active_frame_audit",
        "Final-layer active token-frame diagnostics at length1024 and rank16, averaged within each document and then over32 documents. Left: coherence mu=(length/r)max_i||U_r[i,:]||² for Q and K (C and B in Mamba); gray band is the5–95% variation among64 Haar-Stiefel reference frames, and its dashed line is their mean. The gray band is not a confidence interval of an empirical mean. Right: mean squared canonical correlation ||U_Q^T U_K||F²/r (subspace overlap) and ordered overlap sum_a|u_Q,a^T u_K,a|²/r are distinct diagnostics. Dashed lines and lightly shaded bands in the matching colors are the independent-Haar-pair means and5–95% frame-pair variation; the dotted line at1 marks an exact common ordered frame. Overlap is shown on a log scale so both Haar references remain visible. Empirical bars have saved95% document-bootstrap intervals; null variation and empirical uncertainty are different quantities. Coherence departures show anisotropy, while overlap above independent references but below1 shows partial dependent geometry rather than validation of an exact common frame. Shared Mamba factor groups are deduplicated: Codestral heads use two distinct groups; Mamba2-1.3B has one unique B/C group. Codestral-7B(HF) denotes Transformers4.57.1 with global rather than official eight-group gated RMSNorm. Near-uniform input eigenangles in the companion figure do not establish Haar active frames.")

    unique={json.dumps(r,sort_keys=True):r for r in selected};path=args.output/"audited_geometry_plotting_rows.csv.gz"
    with gzip.open(path,"wt",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in unique.values() for k in r}));writer.writeheader();writer.writerows(unique.values())
    manifest=dict(status="audited_descriptive_figures",source_summary_sha256=sources,plotter_sha256=sha(__file__),
        files={name:sha(args.output/name) for name in created},plotting_rows=len(unique),plotting_rows_sha256=sha(path),
        documents=32,geometry_claim="Descriptive input angles plus active-frame anisotropy and partial dependence; no Haar validation",mamba7_runtime="Transformers4.57.1,global gated RMSNorm",
        numerical_rank_gate="Every input-angle row has numerical_nullspace_columns=structural_completion_columns=0",null_band="5–95% variation across64 frames/pairs,not mean confidence interval")
    (args.output/"audited_geometry_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(output=str(args.output),files=created,plotting_rows=len(unique))))

if __name__=="__main__":main()
