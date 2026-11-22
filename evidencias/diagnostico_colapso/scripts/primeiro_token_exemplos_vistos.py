"""Diagnóstico do colapso do LoRA-02. Rodar de src/ com PYTHONPATH=. (usa artifacts/lora_piloto).
1) quais exemplos de treino foram consumidos nos 40 passos;
2) distribuição do primeiro token da resposta (B1 vs LoRA-02 vs checkpoint 20) em holdout e treino visto;
3) log-prob da resposta correta vs '[]' sob cada modelo."""
import json
import sys

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import load

sys.path.insert(0, '.')
from motor.data import read_jsonl  # noqa: E402
from motor.model import MODEL_ID, MODEL_REVISION, build_messages, configure_tokenizer  # noqa: E402

path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
pilot = read_jsonl('data/pilot/train.jsonl')
hold = read_jsonl('data/prepared/holdout_rows.jsonl')

# ---- 1) exemplos consumidos (reconstrução do iterate_batches do mlx-lm) ----
from transformers import AutoTokenizer  # noqa: E402
hf = configure_tokenizer(AutoTokenizer.from_pretrained(path))
lens = [len(hf.apply_chat_template(r['messages'], return_dict=False)) for r in pilot]
idx = sorted(range(len(pilot)), key=lambda i: lens[i])
np.random.seed(42)
perm = np.random.permutation(len(idx))
seen = [idx[i] for i in perm[:40]]
seen_empty = sum(json.loads(pilot[i]['messages'][-1]['content']) == [] for i in seen)
print(f'[1] exemplos vistos: {len(set(seen))} distintos; alvo vazio entre eles: {seen_empty}/40')


def first_token_report(model, tokenizer, label, rows, as_msgs):
    empty_id = tokenizer.encode('[]', add_special_tokens=False)
    full_id = tokenizer.encode('[{"', add_special_tokens=False)
    res = []
    for r in rows:
        msgs = r['messages'][:-1] if as_msgs else build_messages(r['text'], r['history'])
        gold = r['messages'][-1]['content'] if as_msgs else json.dumps(r['acts'], ensure_ascii=False)
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)
        def seq_logprob(target):
            t = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids('<|im_end|>')]
            x = mx.array([prompt + t])
            logits = model(x[:, :-1]).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = x[0, len(prompt):]
            return mx.take_along_axis(lp[0, len(prompt) - 1:], tgt[:, None], axis=-1).sum().item(), lp[0, len(prompt) - 1]
        g_lp, first = seq_logprob(gold)
        e_lp, _ = seq_logprob('[]')
        p = mx.exp(first)
        res.append({'p_empty_tok': p[empty_id[0]].item(), 'p_full_tok': p[full_id[0]].item(),
                    'gold_lp': g_lp, 'empty_lp': e_lp, 'gold_empty': gold == '[]'})
    pos = [x for x in res if not x['gold_empty']]
    print(f'[2/3] {label}: P(1º token "[]") média={np.mean([x["p_empty_tok"] for x in pos]):.3f} '
          f'P(1º token \'[{{"\')={np.mean([x["p_full_tok"] for x in pos]):.3f} | '
          f'resposta certa mais provável que "[]" em {sum(x["gold_lp"] > x["empty_lp"] for x in pos)}/{len(pos)} positivos')
    return res


print('tokens:', hf.tokenize('[]'), hf.tokenize('[{"slot"'))
hold_pos = hold
seen_rows = [pilot[i] for i in seen]
for label, adapter in [('B1 (sem adaptador)', None), ('LoRA-02 ckpt 20', 'ck20'), ('LoRA-02 final (40)', 'artifacts/lora_piloto')]:
    if adapter == 'ck20':
        import shutil, tempfile, os  # noqa: E401
        d = tempfile.mkdtemp()
        shutil.copy('artifacts/lora_piloto/adapter_config.json', d)
        shutil.copy('artifacts/lora_piloto/0000020_adapters.safetensors', os.path.join(d, 'adapters.safetensors'))
        adapter = d
    model, tokenizer = load(path, adapter_path=adapter)
    configure_tokenizer(tokenizer)
    first_token_report(model, tokenizer, label + ' · holdout', hold_pos, False)
    first_token_report(model, tokenizer, label + ' · treino visto', seen_rows, True)
    del model
