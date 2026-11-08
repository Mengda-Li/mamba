"""Two predetermined native heads on the first submitted document (no selection by outcome)."""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(REPO_ROOT))


def main():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from gpt2_experiments import load_gpt2, capture_all_layers, reconstruct_native_attention
    from compare_attention_ranks_wikitext import capture_module_input
    from extract_mamba_matrices import extract_mamba2_matrices
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

    p = argparse.ArgumentParser()
    for name in ['manifest', 'gpt2', 'mamba', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:3')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    records = [json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    document = sorted({x['document_id'] for x in records})[0]
    length = 128
    def ids(model):
        rows = [x for x in records if x['document_id'] == document and x['checkpoint'] == model
                and x['input_condition'] == 'native-prefix' and x['length'] >= length]
        row = min(rows, key=lambda x: x['length'])
        return torch.tensor([row['token_ids'][:length]], dtype=torch.long)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        model, _ = load_gpt2(a.gpt2, device_map=a.device, dtype='float32')
        bundle = capture_all_layers(model, ids('gpt2'), layer_indices=[11], capture_native_core=True)
        _, weights, _ = reconstruct_native_attention(model.transformer.h[11].attn, bundle.layers[11], 11)
        attention = weights[0, 0].float().cpu().numpy()
        del bundle, model, weights
        gc.collect()
        torch.cuda.empty_cache()
        model = Mamba2ForCausalLM.from_pretrained(a.mamba, dtype='auto', device_map=a.device, local_files_only=True).eval()
        for block in model.backbone.layers:
            block.mixer.chunk_size = 64
        module = model.backbone.layers[63].mixer
        tokens = ids('mamba2')
        hidden, mask = capture_module_input(model, module, {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens)})
        matrices = extract_mamba2_matrices(module, hidden, attention_mask=mask, selected_heads=[0])
        ssd = matrices.m[0, 0].float().cpu().numpy()
    metadata = {'document_id': document, 'length': length, 'input_condition': 'native-prefix',
                'selection': 'first submitted document; head 0 at last layer of each checkpoint; fixed before examining heatmaps',
                'gpt2': {'layer': 11, 'head': 0}, 'mamba2': {'layer': 63, 'head': 0},
                'operator_boundary': 'GPT2 causal softmax A; Mamba canonical M=L*(CB^T), excludes dt and D',
                'distinct_tokenization': True, 'normalization': 'raw entries; independent labeled color scales'}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    for ax, matrix, label, cmap in zip(axes, [attention, ssd], ['GPT-2, layer 11 / head 0', 'Mamba-2, layer 63 / head 0'], ['viridis', 'RdBu_r']):
        if cmap == 'RdBu_r':
            vmax = float(np.abs(matrix).max())
            image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest', aspect='equal')
        else:
            image = ax.imshow(matrix, cmap=cmap, vmin=0, interpolation='nearest', aspect='equal')
        energy = float(np.square(matrix).sum())
        diagonal = float(np.square(np.diag(matrix)).sum() / energy)
        sink = float(np.square(matrix[:, 0]).sum() / energy)
        ax.set_title(f'{label}\ndiagonal energy={diagonal:.3f}; column-0 energy={sink:.3f}', fontsize=10)
        ax.set_xlabel('Source token j')
        ax.set_ylabel('Output token i')
        fig.colorbar(image, ax=ax, shrink=.85, label='Raw operator entry')
        metadata['gpt2' if label.startswith('GPT') else 'mamba2'].update(diagonal_energy_fraction=diagonal, first_column_energy_fraction=sink)
    fig.savefig(out / 'native_mixer_heatmaps.pdf')
    fig.savefig(out / 'native_mixer_heatmaps.png', dpi=180)
    np.savez_compressed(out / 'native_mixer_heatmaps.npz', gpt2=attention, mamba2=ssd)
    (out / 'native_mixer_heatmaps.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
