"""Isolated numerical audit; NOT a replacement empirical geometry pipeline.

Compares tall factor and square low-rank score SVD completions. It preserves
native SVD signs and reports two angular branch cuts on the identical U.
Seeded nullspace rotations are diagnostic gauges only, never a proposed fix.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lengths", default="64,128,512,1024")
    p.add_argument("--widths", default="64,128")
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--no-scipy", action="store_true")
    args = p.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    import numpy as np
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        import scipy
        import scipy.linalg
    except ImportError:
        scipy = None
    args.output.mkdir(parents=True, exist_ok=True)
    rows, gauge_rows = [], []
    lengths = [int(v) for v in args.lengths.split(",")]
    widths = [int(v) for v in args.widths.split(",")]
    backends = ["cuda_gesvd", "numpy_gesdd"]
    if scipy is not None and not args.no_scipy:
        backends.append("scipy_gesvd")

    def metrics(eig, branch):
        angles = np.angle(eig)
        values = np.mod(angles, 2*np.pi) if branch == "wrapped_0_2pi" else angles + np.pi
        values = np.sort(values)
        n = len(values)
        ordered = values/(2*np.pi)
        indices = np.arange(1, n+1)
        hist = np.histogram(values, bins=50, range=(0, 2*np.pi))[0].astype(float)
        prob = hist/hist.sum() + 1e-10
        prob /= prob.sum()
        # Exact NeurIPS W1 samples depend on RNG call order; this audit uses a
        # fixed independent uniform draw to expose branch-cut sensitivity.
        uniform = np.sort(np.random.RandomState(233).uniform(0, 2*np.pi, size=n))
        return dict(histogram_kl_uniform=float(np.sum(prob*np.log(prob*50))),
            ks=float(max(np.max(indices/n-ordered), np.max(ordered-(indices-1)/n))),
            angular_w1_midpoint_grid=float(np.mean(np.abs(values-(indices-.5)/n*(2*np.pi)))),
            w1_independent_uniform_seed233=float(np.mean(np.abs(values-uniform))),
            eigenvalues_near_one=int(np.sum(np.abs(eig-1)<1e-8)),
            eigenvalues_near_minus_one=int(np.sum(np.abs(eig+1)<1e-8)),
            mean_real_eigenvalue=float(np.mean(eig.real)),
            modulus_max_error=float(np.max(np.abs(np.abs(eig)-1))),
            histogram_counts=hist.astype(int).tolist())

    def save():
        path = args.output/"rows.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in rows))
        plain = [{k:v for k,v in row.items() if k != "histogram_counts"} for row in rows]
        with (args.output/"rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, sorted({k for row in plain for k in row}));w.writeheader();w.writerows(plain)

    last_case = None
    for replicate in range(args.replicates):
        for length in lengths:
            for width in widths:
                rng = np.random.default_rng(args.seed + replicate*100000 + length*100 + width)
                q, k = rng.standard_normal((length, width)), rng.standard_normal((length, width))
                score = q @ k.T
                # Same concrete matrix passed to every backend.
                for kind, matrix in (("gaussian_tall_factor_Q", q), ("square_score_QK_T", score)):
                    for backend in backends:
                        start = time.time()
                        if backend == "cuda_gesvd":
                            work = torch.from_numpy(matrix).to(args.device)
                            u_t, s_t, vh_t = torch.linalg.svd(work, full_matrices=True, driver="gesvd")
                            u, singular, vh = u_t.cpu().numpy(), s_t.cpu().numpy(), vh_t.cpu().numpy()
                        elif backend == "numpy_gesdd":
                            u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
                        else:
                            u, singular, vh = scipy.linalg.svd(matrix, full_matrices=True, lapack_driver="gesvd")
                        rank = min(length, width)
                        reconstruction = (u[:,:len(singular)]*singular) @ vh[:len(singular)]
                        eig = np.linalg.eigvals(u)
                        common = dict(replicate=replicate, length=length, width=width, kind=kind,
                            backend=backend, matrix_dtype=str(matrix.dtype), svd_signs="solver_native",
                            nominal_active_rank=rank,
                            factor_structural_completion_columns=max(0,length-width) if kind.endswith("Q") else 0,
                            algebraic_nullspace_columns=length-rank,
                            factor_householder_near_one_lower_bound=max(0,length-2*width) if kind.endswith("Q") else None,
                            reconstruction_relative_error=float(np.linalg.norm(matrix-reconstruction)/np.linalg.norm(matrix)),
                            orthogonality_error=float(np.linalg.norm(u.T@u-np.eye(length))),
                            elapsed_seconds=time.time()-start)
                        for branch in ("wrapped_0_2pi", "nips_angle_plus_pi"):
                            rows.append({**common,"branch_cut":branch,**metrics(eig,branch)})
                        save()
                        print(json.dumps({k:rows[-1][k] for k in ("replicate","length","width","kind","backend","eigenvalues_near_one","histogram_kl_uniform","ks","angular_w1_midpoint_grid")}),flush=True)
                        if replicate==0 and length==max(lengths) and width==min(widths) and kind.endswith("Q") and backend=="cuda_gesvd":
                            last_case=(q,k,u,singular,vh)
    # Hold the active columns AND concrete factor fixed, rotate null columns
    # by a single predeclared Haar matrix. This changes only the SVD gauge.
    if last_case is not None:
        q,k,u,singular,vh=last_case
        length,width=q.shape
        rng=np.random.default_rng(args.seed+987654)
        z=rng.standard_normal((length-width,length-width))
        rotation,r=np.linalg.qr(z);rotation*=np.sign(np.diag(r))[None,:]
        rotated=u.copy();rotated[:,width:]=u[:,width:]@rotation
        for name,frame in (("native_completion",u),("same_active_columns_nullspace_rotation_DIAGNOSTIC_ONLY",rotated)):
            eig=np.linalg.eigvals(frame)
            reconstruction=(frame[:,:width]*singular)@vh
            invariant=dict(gauge=name,length=length,width=width,
                active_columns_max_abs_difference=float(np.max(np.abs(frame[:,:width]-u[:,:width]))),
                reconstruction_relative_error=float(np.linalg.norm(q-reconstruction)/np.linalg.norm(q)),
                active_projector_difference=0.0,
                warning="Gauge demonstration only; not an empirical replacement or Haar validation")
            for branch in ("wrapped_0_2pi","nips_angle_plus_pi"):
                gauge_rows.append({**invariant,"branch_cut":branch,**metrics(eig,branch)})
        (args.output/"nullspace_gauge_diagnostic.json").write_text(json.dumps(gauge_rows,indent=2)+"\n")
    manifest=dict(status="complete",purpose=__doc__,seed=args.seed,replicates=args.replicates,
        lengths=lengths,widths=widths,backends=backends,n_rows=len(rows),
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        scipy=getattr(scipy,"__version__",None),device=args.device,
        gpu=torch.cuda.get_device_name(args.device),tf32=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows_sha256=hashlib.sha256((args.output/"rows.jsonl").read_bytes()).hexdigest(),
        note="No trained checkpoints; all Gaussian factors are fixed by declared seeds. Native signs are primary. Random nullspace rotation is an isolated noninvariance demonstration only.")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)


if __name__=="__main__":
    main()
