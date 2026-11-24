"""Treinos de diagnóstico do LoRA-02 (A: reprodução; B: --sem-vazias; C: iters=400) com o mesmo worker do projeto, mudando uma variável por vez.
Uso (de src/): python ablacao.py NOME chave=valor ... [--sem-vazias]"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

S = Path('artifacts/diagnostico')  # local, ignorado pelo Git
name, *opts = sys.argv[1:]
run = S / name
(run / 'data').mkdir(parents=True, exist_ok=True)
base = yaml.safe_load(Path('artifacts/lora_runtime.yaml').read_text())
for split in ('train', 'valid'):
    rows = [json.loads(x) for x in Path('data/pilot', split + '.jsonl').read_text().splitlines()]
    if '--sem-vazias' in opts:
        rows = [r for r in rows if r['messages'][-1]['content'] != '[]']
    (run / 'data' / (split + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
from huggingface_hub import snapshot_download
base.update(data=str(run / 'data'), adapter_path=str(run / 'adapter'),
            model=snapshot_download('mlx-community/Qwen3-4B-Instruct-2507-4bit', revision='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'))
for opt in opts:
    if '=' in opt:
        key, value = opt.split('=', 1)
        base[key] = yaml.safe_load(value)
(run / 'runtime.yaml').write_text(yaml.safe_dump(base))
log = (run / 'train.log').open('w')
code = subprocess.call([sys.executable, '-u', '-m', 'motor.train', '--worker', str(run / 'runtime.yaml'),
                        '--evidence', str(run)], stdout=log, stderr=subprocess.STDOUT)
print(name, 'exit', code)
