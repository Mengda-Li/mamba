import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class _StopAfterCapture(Exception):
    pass


@dataclass
class Mamba2Matrices:
    q: torch.Tensor
    k: torch.Tensor
    a: torch.Tensor
    l: torch.Tensor
    qk: torch.Tensor
    m: torch.Tensor
    dt: torch.Tensor
    value_dt: torch.Tensor
    effective_m: torch.Tensor


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _capture_mixer_input(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}

    capture: dict[str, torch.Tensor | None] = {}
    mixer = model.backbone.layers[layer_idx].mixer

    def hook(_module, args, kwargs):
        capture["hidden_states"] = args[0].detach()
        capture["attention_mask"] = kwargs.get("attention_mask")
        raise _StopAfterCapture

    handle = mixer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            try:
                model(**inputs, use_cache=False)
            except _StopAfterCapture:
                pass
    finally:
        handle.remove()

    if "hidden_states" not in capture:
        raise RuntimeError(f"Did not capture input to layer {layer_idx}.")

    return capture["hidden_states"], capture["attention_mask"]


def _apply_padding_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    if attention_mask.shape[1] <= 1:
        return hidden_states
    return hidden_states * attention_mask[:, :, None].to(hidden_states.device, hidden_states.dtype)


def extract_mamba2_matrices(
    mixer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    selected_heads: list[int] | torch.Tensor | None = None,
) -> Mamba2Matrices:
    """Reconstruct the SSD matrices used by a Transformers Mamba2 mixer.

    Shapes:
      q, k:           [batch, heads, seq_len, state_size]
      a:              [batch, heads, seq_len]
      l, qk, m:       [batch, heads, seq_len, seq_len]
      dt, value_dt:   [batch, heads, seq_len]

    `m` is the paper's matrix L * (Q K^T). `effective_m` additionally
    multiplies each source column by dt_j, matching the matrix applied to the
    original pre-discretized value stream in the HF implementation.
    """
    if mixer.__class__.__name__ != "Mamba2Mixer":
        raise TypeError(f"Expected a Mamba2Mixer, got {mixer.__class__.__name__}.")

    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    hidden_states = _apply_padding_mask(hidden_states, attention_mask)
    projected_states = mixer.in_proj(hidden_states)

    groups_state = mixer.n_groups * mixer.ssm_state_size
    d_mlp = (
        projected_states.shape[-1]
        - 2 * mixer.intermediate_size
        - 2 * groups_state
        - mixer.num_heads
    ) // 2

    _, _, _, hidden_states_b_c, dt_raw = projected_states.split(
        [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads],
        dim=-1,
    )

    conv_in = hidden_states_b_c.transpose(1, 2)
    conv_out = mixer.conv1d(conv_in)[..., :seq_len].transpose(1, 2)
    hidden_states_b_c = mixer.act(conv_out)
    hidden_states_b_c = _apply_padding_mask(hidden_states_b_c, attention_mask)

    _, b, c = torch.split(
        hidden_states_b_c,
        [mixer.intermediate_size, groups_state, groups_state],
        dim=-1,
    )

    b = b.reshape(batch_size, seq_len, mixer.n_groups, mixer.ssm_state_size).float()
    c = c.reshape(batch_size, seq_len, mixer.n_groups, mixer.ssm_state_size).float()
    repeat = mixer.num_heads // mixer.n_groups
    b = b.repeat_interleave(repeat, dim=2, output_size=mixer.num_heads)
    c = c.repeat_interleave(repeat, dim=2, output_size=mixer.num_heads)

    dt = F.softplus(dt_raw + mixer.dt_bias.to(dt_raw.device, dt_raw.dtype))
    dt = torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])

    a_log = -torch.exp(mixer.A_log.float()).to(device)[None, None, :] * dt.float()

    q = c.permute(0, 2, 1, 3).contiguous()
    k = b.permute(0, 2, 1, 3).contiguous()
    value_dt = dt.permute(0, 2, 1).contiguous().float()
    if selected_heads is not None:
        head_idx = torch.as_tensor(selected_heads, device=device, dtype=torch.long)
        q = q.index_select(1, head_idx)
        k = k.index_select(1, head_idx)
        value_dt = value_dt.index_select(1, head_idx)
        a_log = a_log.index_select(2, head_idx)

    # a[:, :, t] is the scalar transition from token t - 1 to token t.
    a = torch.exp(a_log).permute(0, 2, 1).contiguous()

    prefix = torch.cumsum(a_log, dim=1)
    log_l = prefix[:, :, None, :] - prefix[:, None, :, :]
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    l = torch.exp(log_l).permute(0, 3, 1, 2).contiguous()
    l = l.masked_fill(~causal[None, None, :, :], 0.0)

    qk = torch.einsum("bhid,bhjd->bhij", q, k)
    m = l * qk
    effective_m = m * value_dt[:, :, None, :]

    return Mamba2Matrices(
        q=q,
        k=k,
        a=a,
        l=l,
        qk=qk,
        m=m,
        dt=dt,
        value_dt=value_dt,
        effective_m=effective_m,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Q, K, L, and M from a HF Mamba2 layer.")
    model_root = Path(os.environ.get("MODEL_ROOT", "models"))
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MAMBA2_MODEL_PATH",
            str(model_root / "Mamba-Codestral-7B-v0.1"),
        ),
    )
    parser.add_argument("--prompt", default="Plants create energy through a process known as")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        dtype="auto",
        local_files_only=True,
    )
    model.eval()

    hidden_states, attention_mask = _capture_mixer_input(model, tokenizer, args.prompt, args.layer)
    matrices = extract_mamba2_matrices(
        model.backbone.layers[args.layer].mixer,
        hidden_states,
        attention_mask=attention_mask,
    )

    bundle = {
        "q": matrices.q.cpu(),
        "k": matrices.k.cpu(),
        "a": matrices.a.cpu(),
        "l": matrices.l.cpu(),
        "qk": matrices.qk.cpu(),
        "m": matrices.m.cpu(),
        "dt": matrices.dt.cpu(),
        "value_dt": matrices.value_dt.cpu(),
        "effective_m": matrices.effective_m.cpu(),
    }

    for name, tensor in bundle.items():
        print(f"{name:12s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    if args.save:
        torch.save(bundle, args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
