"""P(1º token '[]') por tipo de fala, para vários adaptadores. Uso (de src/): python primeiro_token_por_tipo.py rotulo=pasta_do_adaptador ... (none = sem adaptador)"""
import sys

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import load

sys.path.insert(0, '.')
from motor.data import read_jsonl  # noqa: E402
from motor.model import MODEL_ID, MODEL_REVISION, build_messages, configure_tokenizer  # noqa: E402

path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
valid = read_jsonl('data/prepared/valid_rows.jsonl')
rng = np.random.default_rng(0)
sample = [valid[i] for i in rng.choice(len(valid), 120, replace=False)]
for spec in sys.argv[1:]:
    label, adapter = spec.split('=', 1)
    model, tok = load(path, adapter_path=None if adapter == 'none' else adapter)
    configure_tokenizer(tok)
    empty_id = tok.encode('[]', add_special_tokens=False)[0]
    out = {True: [], False: []}
    for r in sample:
        prompt = tok.apply_chat_template(build_messages(r['text'], r['history']), add_generation_prompt=True,
                                         return_dict=False)
        logits = model(mx.array([prompt]))[0, -1].astype(mx.float32)
        out[not r['acts']].append(mx.softmax(logits)[empty_id].item())
    e, p = np.array(out[True]), np.array(out[False])
    # AUC: chance de uma fala sem pergunta ter P('[]') maior que uma fala com pergunta
    auc = np.mean([[a > b for b in p] for a in e])
    print(f'{label:28s} P("[]") sem pergunta={e.mean():.3f} (n={len(e)}) | com pergunta={p.mean():.3f} '
          f'(n={len(p)}) | AUC separação={auc:.2f} | greedy "[]" em positivos={np.mean(p > .5):.0%}', flush=True)
    del model
