"""Compare patched/unpatched full-checkpoint computation before resuming a run."""

from repo_paths import REPO_ROOT, ARTIFACT_ROOT
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(REPO_ROOT))


def main():
    import torch
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
    from compare_attention_ranks_wikitext import capture_module_input
    from mamba_memory_patch_exact import install_patch
    p = argparse.ArgumentParser()
    for key in ['manifest', 'checkpoint', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:3')
    p.add_argument('--length', type=int, default=128)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    records = [json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    sample = next(x for x in records if x['checkpoint'] == 'mamba2' and x['length'] == a.length
                  and x['input_condition'] == 'native-prefix')
    model = Mamba2ForCausalLM.from_pretrained(a.checkpoint, dtype='auto', device_map=a.device,
                                             local_files_only=True).eval()
    for block in model.backbone.layers:
        block.mixer.chunk_size = 64
    layer = len(model.backbone.layers) - 1
    module = model.backbone.layers[layer].mixer
    ids = torch.tensor([sample['token_ids']])
    inputs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    def compare(old, new):
        return {'relative_error': float(torch.linalg.vector_norm((old.float()-new.float()).double()) /
                                        torch.linalg.vector_norm(old.double()).clamp_min(1e-30)),
                'max_abs_error': float((old.float()-new.float()).abs().max()),
                'bitwise_equal': torch.equal(old, new)}
    with torch.inference_mode():
        old, mask = capture_module_input(model, module, inputs)
        old = old.detach().clone()
        output_old = module(old, attention_mask=mask).detach().clone()
        patch = install_patch(head_chunk_size=8)
        output_same_input = module(old, attention_mask=mask)
        new, new_mask = capture_module_input(model, module, inputs)
        output_new = module(new, attention_mask=new_mask)
        checks = {'same_input_mixer_output': compare(output_old, output_same_input),
                  'final_mixer_input': compare(old, new),
                  'complete_final_mixer_output': compare(output_old, output_new)}
    result = {'document_id': sample['document_id'], 'length': a.length, 'layer': layer,
              'checks': checks, 'tolerance': .005, 'patch': patch,
              'passed': all(v['relative_error'] < .005 for v in checks.values())}
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'patch'}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
