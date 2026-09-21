#!/usr/bin/env python3
"""Bounded GPU parity check; collect and replay in separate processes.

uv run python scripts/check_vllm_steering_parity.py collect --out /tmp/steering-parity.json
uv run python scripts/check_vllm_steering_parity.py replay --out /tmp/steering-parity.json

vLLM generates greedy traces and token log-probabilities. Transformers replays
those exact tokens, avoiding divergence from sampling; both apply the same edit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from reasoning_attention.config import MODEL_ID, NLAConfig
from reasoning_attention.tokenview import _chat_header


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['collect', 'replay'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=96)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    boundaries = [i for i in range(len(tok)) if tok.convert_ids_to_tokens(i).count('Ċ') >= 2]
    boundary_set = set(boundaries)
    close_id = tok.convert_tokens_to_ids('</think>')
    unit = torch.load('data/pool/dir_A_1trace.pt', map_location='cpu', weights_only=False)['unit'].float()
    if args.phase == 'collect':
        from vllm import SamplingParams
        from reasoning_attention.serving.vllm_steering import build_steering_llm
        rows = [json.loads(x) for x in Path('data/prefix_online/prefixes.jsonl').read_text().splitlines()][:3]
        prompts = [tok(_chat_header(tok, r['question']) + r['prefix'], add_special_tokens=True)['input_ids'] for r in rows]
        llm = build_steering_llm(MODEL_ID, NLAConfig().extraction_layer, boundaries, close_id,
                                 max(map(len, prompts)) + args.tokens, 2, .65, 256)
        records = []
        traces = {}
        for arm, direction, alpha in [('base', None, 0.), ('zero', unit.tolist(), 0.), ('N', unit.tolist(), 1.)]:
            llm.collective_rpc('configure_reasoning_steering', args=(direction, alpha, True))
            params = [SamplingParams(temperature=0, max_tokens=args.tokens, logprobs=1,
                                     extra_args={'steering_audit_id': str(i)}) for i in range(len(prompts))]
            outputs = llm.generate([{'prompt_token_ids': p} for p in prompts], params, use_tqdm=False)
            stats = llm.collective_rpc('reasoning_steering_stats')[0]
            assert stats['calls'] > 0
            traces[arm] = [list(o.outputs[0].token_ids) for o in outputs]
            for i, out in enumerate(outputs):
                seq = out.outputs[0]
                ids = list(seq.token_ids)
                close = ids.index(close_id) if close_id in ids else len(ids)
                expected = [len(prompts[i])+j for j, t in enumerate(ids[:-1]) if j < close and t in boundary_set]
                audit = stats['requests'][str(i)]
                assert audit['sites'] == expected, (audit, expected)
                assert audit['injections'] == (expected if arm == 'N' else []), audit
                if arm == 'zero':
                    continue
                records.append({'arm': arm, 'prompt': prompts[i], 'ids': ids,
                                'logprobs': [lp[t].logprob for lp, t in zip(seq.logprobs, ids)],
                                'audit': audit})
        assert traces['base'] == traces['zero'], 'alpha=0 changed greedy outputs'
        assert any(r['audit']['injections'] for r in records if r['arm'] == 'N'), 'No injection exercised'
        assert traces['base'] != traces['N'], 'Steering produced no observable change'
        args.out.write_text(json.dumps(records))
        print(f'PASS: zero-alpha exact, sites exact, chunked prefill and 3 requests / 2 slots; saved {args.out}')
    else:
        from transformers import AutoModelForCausalLM
        from reasoning_attention.nla.arch import inner_transformer
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='cuda').eval()
        unit = unit.to('cuda')
        state = {'inject': False}
        def hook(_module, _inputs, output):
            if not state['inject']:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            raw = h[:, -1, :]
            h[:, -1, :] = raw + (raw.float().norm(dim=-1, keepdim=True) * unit).to(h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h
        handle = inner_transformer(model).layers[NLAConfig().extraction_layer].register_forward_hook(hook)
        records = json.loads(args.out.read_text())
        report = {}
        for arm in ['base', 'N']:
            diffs, agree = [], []
            for row in [r for r in records if r['arm'] == arm]:
                ids = torch.tensor([row['prompt']], device='cuda')
                past = None
                thinking = True
                state['inject'] = False
                for step, target in enumerate(row['ids']):
                    with torch.no_grad():
                        res = model(input_ids=ids, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = res.past_key_values
                    logits = res.logits[0, -1].float()
                    lp = torch.log_softmax(logits, -1)[target].item()
                    diffs.append(abs(lp - row['logprobs'][step]))
                    agree.append(int(logits.argmax().item() == target))
                    if target == close_id:
                        thinking = False
                    state['inject'] = arm == 'N' and thinking and target in boundary_set
                    ids = torch.tensor([[target]], device='cuda')
            report[arm] = {'mean_abs_logprob': float(np.mean(diffs)),
                           'max_abs_logprob': float(np.max(diffs)),
                           'greedy_agreement': float(np.mean(agree)), 'tokens': len(diffs)}
            assert np.mean(diffs) < .12 and np.max(diffs) < .6 and np.mean(agree) >= .85, report[arm]
        handle.remove()
        args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print('PASS: fixed-token HF/vLLM parity within declared BF16 tolerances')


if __name__ == '__main__':
    main()
