#!/usr/bin/env python3
"""Small, predetermined SVD precision audit; never changes production results.

Compare CUDA's default FP32 SVD with QR-based gesvd and FP64 gesvd on the
same operator entries. RetNet additionally forms the operator in FP64 from
the same native captured Q/K factors; casting an FP32-formed matrix to FP64
does NOT undo matrix-formation rounding. Independent entry-derived Frobenius
norms and locality fractions accompany every SVD measurement.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))
SCHEMA_VERSION = 1


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def ratio(a: float, b: float):
    return a / b if b > 0 else None


def parse_heads(value: str, total: int) -> list[int]:
    heads = [int(x) for x in value.split(",")]
    if len(set(heads)) != len(heads) or any(x < 0 or x >= total for x in heads):
        raise ValueError(f"Invalid predetermined heads {heads} for {total} heads")
    return heads


def documents(args) -> list[dict]:
    longest = {}
    for line in Path(args.manifest).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model", row.get("checkpoint")) != args.model:
            continue
        if row.get("input_condition", "native-prefix") != "native-prefix":
            continue
        key = str(row["document_id"])
        if key not in longest or len(row["token_ids"]) > len(longest[key]["token_ids"]):
            longest[key] = row
    eligible = [x for x in longest.values() if len(x["token_ids"]) >= args.length]
    selected = eligible[args.doc_start:args.doc_stop]
    if not selected:
        raise ValueError(f"No eligible native-prefix {args.model} documents for L={args.length}")
    return selected


def spectrum_row(matrix, value, *, dtype, driver, base, rank_bound):
    work = matrix.detach().to(dtype=dtype).contiguous()
    if not bool(torch.isfinite(work).all()):
        raise ValueError("Nonfinite audit operator")
    entry_square = work.double().square()
    entry_energy = float(entry_square.sum())
    started = time.monotonic()
    kwargs = {"driver": driver} if work.device.type == "cuda" and driver else {}
    singular = torch.linalg.svdvals(work, **kwargs).double().cpu()
    elapsed = time.monotonic() - started
    singular_energy = float(singular.square().sum())
    top = float(singular[0])
    nonzero = top > 0.0
    p1 = singular / singular.sum() if nonzero else singular
    p2 = singular.square() / singular.square().sum() if nonzero else singular
    eps = torch.finfo(dtype).eps
    threshold = top * max(work.shape) * eps
    row = {
        **base, "shape": list(work.shape), "svd_dtype": str(dtype),
        "svd_driver_requested": driver or "default",
        "svd_driver_effective": (driver or "cuda_default_gesvdj_with_fallback") if work.device.type == "cuda" else "cpu_default",
        "svd_seconds": elapsed, "zero_matrix": not nonzero,
        "entry_frobenius_norm": math.sqrt(entry_energy),
        "svd_frobenius_norm": math.sqrt(singular_energy),
        "svd_frobenius_relative_error": ratio(abs(math.sqrt(singular_energy)-math.sqrt(entry_energy)), math.sqrt(entry_energy)),
        "svd_squared_energy_relative_error": ratio(singular_energy-entry_energy, entry_energy),
        "operator_norm": top, "nuclear_norm": float(singular.sum()),
        "srank_svd_energy": ratio(singular_energy, top**2) if nonzero else 0.0,
        "srank_entry_energy": ratio(entry_energy, top**2) if nonzero else 0.0,
        "erank_1": float((-(p1[p1 > 0]*p1[p1 > 0].log()).sum()).exp()) if nonzero else 0.0,
        "erank_2": float((-(p2[p2 > 0]*p2[p2 > 0].log()).sum()).exp()) if nonzero else 0.0,
        "absolute_counts": {format(t, ".12g"): int((singular > t).sum()) for t in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9)},
        "relative_counts": {format(t, ".12g"): int((singular > t*top).sum()) if nonzero else 0
                            for t in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12)},
        "machine_scaled_tolerance": threshold,
        "machine_scaled_rank": int((singular > threshold).sum()) if nonzero else 0,
        "rank_bound": rank_bound,
        "top16_singular_values": singular[:16].tolist(),
        "smallest16_singular_values": singular[-16:].tolist(),
    }
    if rank_bound is not None:
        row["singular_values_near_rank_bound"] = singular[max(0,rank_bound-4):rank_bound+8].tolist()
        row["squared_energy_beyond_rank_bound"] = float(singular[rank_bound:].square().sum())
        row["squared_energy_beyond_rank_bound_fraction"] = ratio(row["squared_energy_beyond_rank_bound"], singular_energy)
        row["algebraic_rank_note"] = "bound applies to exact RetNet factorization; finite precision formation may produce a small tail"
    if work.shape[0] == work.shape[1] and base["region"] == "full":
        n = work.shape[0]
        pos = torch.arange(n, device=work.device)
        lag = pos[:, None] - pos[None, :]
        diagonal = float(entry_square.diag().sum())
        past = float(entry_square[lag > 0].sum())
        future = float(entry_square[lag < 0].sum())
        row.update({
            "diagonal_energy_fraction_entry": ratio(diagonal, entry_energy),
            "strict_past_energy_fraction_entry": ratio(past, entry_energy),
            "future_energy_fraction_entry": ratio(future, entry_energy),
            "locality_partition_sum_entry": ratio(diagonal+past+future, entry_energy),
            "diagonal_energy_fraction_svd_frob": ratio(diagonal, singular_energy),
            "strict_past_energy_fraction_svd_frob": ratio(past, singular_energy),
            "locality_partition_sum_svd_frob": ratio(diagonal+past+future, singular_energy),
        })
    if value is not None:
        native_value = value.to(device=work.device, dtype=torch.float64)
        output = work.double() @ native_value
        input_norm = float(torch.linalg.vector_norm(native_value))
        output_norm = float(torch.linalg.vector_norm(output))
        row.update(value_available=True, value_input_frobenius_norm=input_norm,
                   value_output_frobenius_norm=output_norm,
                   value_output_to_input_gain=ratio(output_norm,input_norm),
                   value_diagnostic_dtype="float64 multiplication of the audited operator entries and native captured values")
    else:
        row["value_available"] = False
    return row


def audit_operator(matrix, value, *, build_dtype, base, rank_bound, include_fp32):
    n = matrix.shape[0]
    cut = n//2
    rows = []
    for region, block, vals in (("full",matrix,value),
                                ("crosscut",matrix[cut:,:cut],value[:cut] if value is not None else None)):
        label = {**base,"region":region,"cut_index":cut if region=="crosscut" else None,
                 "gap":0 if region=="crosscut" else None,"operator_build_dtype":build_dtype}
        bound = rank_bound if region == "crosscut" else None
        methods = [(torch.float32,None),(torch.float32,"gesvd"),(torch.float64,"gesvd")] if include_fp32 else [(torch.float64,"gesvd")]
        for dtype,driver in methods:
            rows.append(spectrum_row(block,vals,dtype=dtype,driver=driver,base=label,rank_bound=bound))
            print(f"{base['model']} doc={base['document_id']} head={base['head']} {region} build={build_dtype} svd={dtype}/{driver or 'default'} froerr={rows[-1]['svd_frobenius_relative_error']:.3g}",flush=True)
    return rows


def save_head(args,sample,head,rows,metadata):
    key=sha([sample['document_id'],head])[:16]
    target=Path(args.output)/"shards"/f"{args.model}.{key}.json"
    atomic_json(target,{"metadata":metadata,"document_id":sample['document_id'],"head":head,"rows":rows,"status":"ok"})


def run_retnet(args,samples,metadata):
    from compare_attention_ranks_wikitext import capture_module_input
    import fla.models
    from fla.models import RetNetForCausalLM
    model=RetNetForCausalLM.from_pretrained(args.checkpoint,device_map=args.device,dtype=args.model_dtype,local_files_only=True).eval()
    layer=len(model.model.layers)-1
    module=model.model.layers[layer].attn
    heads=parse_heads(args.heads or "0,4,7",module.num_heads)
    allrows=[]
    for sample in samples:
        ids=torch.tensor([sample['token_ids'][:args.length]],dtype=torch.long)
        hidden,_=capture_module_input(model,module,{"input_ids":ids,"attention_mask":torch.ones_like(ids)})
        with torch.inference_mode():
            kv=getattr(module,'num_kv_heads',module.num_heads)
            q=module.q_proj(hidden).view(1,args.length,module.num_heads,module.head_k_dim)
            k=module.k_proj(hidden).view(1,args.length,kv,module.head_k_dim)
            v=module.v_proj(hidden).view(1,args.length,kv,module.head_v_dim)
            q,k=module.rotary(q,k,seqlen_offset=0,max_seqlen=max(args.length,2048))
            if kv!=module.num_heads:
                k=k.repeat_interleave(module.num_heads//kv,2);v=v.repeat_interleave(module.num_heads//kv,2)
            pos=torch.arange(args.length,device=q.device)
            lag=pos[:,None]-pos[None,:]
            causal=lag>=0
            for head in heads:
                query,key,value=[x[0,:,head].detach() for x in (q,k,v)]
                gamma=1.0-2.0**(-5-head)
                base={"model":args.model,"document_id":sample['document_id'],"length":args.length,"layer":layer,"head":head,
                      "native_factor_dtype":str(query.dtype),"token_ids_sha256":sha(sample['token_ids'][:args.length]),
                      "gamma":gamma,"state_width":module.head_k_dim,"run_fingerprint":metadata['run_fingerprint'],
                      "value_scope":"native projected V before retention gate/norm/output projection"}
                fp32=(query.float()@key.float().T/math.sqrt(module.head_k_dim))*torch.exp(math.log(gamma)*lag.clamp_min(0))*causal
                fp64=(query.double()@key.double().T/math.sqrt(module.head_k_dim))*torch.exp(math.log(gamma)*lag.clamp_min(0).double())*causal
                delta=fp32.double()-fp64
                base['fp32_formation_relative_frobenius_error_vs_fp64']=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(fp64))
                headrows=audit_operator(fp32,value,build_dtype='float32',base=base,rank_bound=module.head_k_dim,include_fp32=True)
                headrows.extend(audit_operator(fp64,value,build_dtype='float64_from_same_native_factors',base=base,rank_bound=module.head_k_dim,include_fp32=False))
                save_head(args,sample,head,headrows,metadata);allrows.extend(headrows)
                del fp32,fp64,delta,headrows
    return allrows


def run_pythia(args,samples,metadata):
    from transformers import AutoModelForCausalLM
    dtype='float32' if args.model_dtype=='auto' else args.model_dtype
    if dtype!='float32':
        raise ValueError('Pythia audit requires --model-dtype float32 (or auto, which resolves to float32)')
    model=AutoModelForCausalLM.from_pretrained(args.checkpoint,device_map=args.device,dtype=dtype,local_files_only=True,attn_implementation='eager').eval()
    layer=model.config.num_hidden_layers-1
    heads=parse_heads(args.heads or '0,8,15',model.config.num_attention_heads)
    allrows=[]
    for sample in samples:
        capture={}
        block=model.gpt_neox.layers[layer]
        module=getattr(block,'attention',None) or getattr(block,'self_attn',None)
        qkv=getattr(module,'query_key_value',None)
        def hook(_module,_args,output):
            out=output[0] if isinstance(output,(tuple,list)) else output
            shaped=out.view(1,args.length,model.config.num_attention_heads,-1)
            capture['value']=shaped.chunk(3,dim=-1)[2].detach().cpu()
        handle=qkv.register_forward_hook(hook) if qkv is not None else None
        ids=torch.tensor([sample['token_ids'][:args.length]],device=model.get_input_embeddings().weight.device)
        try:
            with torch.inference_mode():
                out=model.base_model(input_ids=ids,attention_mask=torch.ones_like(ids),output_attentions=True,use_cache=False,return_dict=True)
            weights=out.attentions[layer].detach().cpu()
            del out
        finally:
            if handle is not None:handle.remove()
        if weights.dtype!=torch.float32:
            raise ValueError(f'Expected native FP32 Pythia attention, got {weights.dtype}')
        for head in heads:
            matrix=weights[0,head].to(args.device)
            value=capture['value'][0,:,head].to(args.device) if 'value' in capture else None
            base={"model":args.model,"document_id":sample['document_id'],"length":args.length,"layer":layer,"head":head,
                  "native_attention_dtype":str(weights.dtype),"native_factor_dtype":'native_model_float32',
                  "token_ids_sha256":sha(sample['token_ids'][:args.length]),"run_fingerprint":metadata['run_fingerprint'],
                  "operator_definition":"native eager model.attentions, no reconstruction or normalization change",
                  "value_scope":"native projected V before output projection"}
            rows=audit_operator(matrix,value,build_dtype='native_float32_attention',base=base,rank_bound=None,include_fp32=True)
            save_head(args,sample,head,rows,metadata);allrows.extend(rows)
    return allrows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('retnet','pythia'),required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--heads',default='')
    parser.add_argument('--doc-start',type=int,default=0)
    parser.add_argument('--doc-stop',type=int,default=1)
    parser.add_argument('--length',type=int,default=2048)
    parser.add_argument('--model-dtype',choices=('auto','float32','float16','bfloat16'),default='auto')
    args=parser.parse_args()
    if args.length<2 or args.doc_start<0 or args.doc_stop<=args.doc_start:parser.error('Invalid length/document slice')
    torch.manual_seed(20261108)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    metadata={"schema_version":SCHEMA_VERSION,"args":vars(args),"manifest_sha256":file_sha(Path(args.manifest)),
              "source_sha256":file_sha(Path(__file__)),"torch_version":str(torch.__version__),"cuda_version":torch.version.cuda,
              "checkpoint_config_sha256":file_sha(Path(args.checkpoint)/'config.json'),
              "selection":"first eligible native-prefix documents in manifest order; predetermined final-layer heads",
              "warning":"FP64 SVD on FP32 entries does not undo FP32 operator formation error; no production results overwritten"}
    metadata['run_fingerprint']=sha(metadata)
    samples=documents(args)
    metadata['selected_document_ids']=[x['document_id'] for x in samples]
    atomic_json(Path(args.output)/'precision_metadata.json',metadata)
    rows=(run_retnet if args.model=='retnet' else run_pythia)(args,samples,metadata)
    atomic_json(Path(args.output)/'precision_report.json',{"metadata":metadata,"rows":rows,"status":"complete"})
    print(json.dumps({"status":"complete","rows":len(rows),"report":str(Path(args.output)/'precision_report.json')}),flush=True)


if __name__=='__main__':main()
