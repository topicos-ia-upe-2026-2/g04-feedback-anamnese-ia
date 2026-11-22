"""Escolhe o checkpoint de uma rodada pela extração na validação (não pelo teste nem só pela perda).

Uso: python -m motor.select --run lora03 [--n 80]
Gera em evidencias/<run>/selecao_checkpoint.json as métricas de cada checkpoint numa amostra fixa da validação
e copia o vencedor para artifacts/<run>/melhor/.
"""
import argparse
import json
import random
import shutil
from datetime import datetime, timezone

from motor.data import CODE_ROOT, EVIDENCE, read_jsonl, sha256, write_json, write_jsonl
from motor.evaluate import score_predictions
from motor.model import MLXExtractor, parse_output


def validation_sample(run, n, seed=42):
    """Amostra fixa da validação, com a distribuição natural, excluindo as falas usadas na perda de validação."""
    rows = read_jsonl(CODE_ROOT / 'data' / 'prepared' / 'valid_rows.jsonl')
    used = set(json.loads((EVIDENCE / run / 'manifesto_piloto.json').read_text())['valid']['sample_ids'])
    pool = [row for row in rows if row['dialogue_id'] + ':' + row['uid'] not in used]
    sample = random.Random(seed).sample(pool, n)
    path = CODE_ROOT / 'data' / run / 'valid_selecao_rows.jsonl'
    write_jsonl(path, sample)
    return sample, path


def generate(adapter_dir, rows):
    backend = MLXExtractor(adapter_path=str(adapter_dir))
    predictions = []
    for row in rows:
        try:
            predictions.append(parse_output(backend.generate_raw(row['text'], row['history'])))
        except ValueError:
            predictions.append(None)
    return predictions


def select(run, n=80):
    adapters = CODE_ROOT / 'artifacts' / run
    sample, path = validation_sample(run, n)
    gold = [row['acts'] for row in sample]
    results = []
    for checkpoint in sorted(adapters.glob('*_adapters.safetensors')):
        candidate = adapters / ('ckpt_' + checkpoint.name.split('_')[0])
        candidate.mkdir(exist_ok=True)
        shutil.copy(adapters / 'adapter_config.json', candidate)
        shutil.copy(checkpoint, candidate / 'adapters.safetensors')
        predictions = generate(candidate, sample)
        score = score_predictions(gold, predictions)
        results.append({'checkpoint': checkpoint.name, 'iteration': int(checkpoint.name.split('_')[0]),
                        'sha256': sha256(checkpoint),
                        'slot_value_f1': score['slot_value_micro']['f1'], 'act_f1': score['act_micro']['f1'],
                        'attribute_f1': score['attribute_micro']['f1'],
                        'json_schema_valid_rate': score['json_schema_valid_rate'],
                        'exact_turn_accuracy': score['exact_turn_accuracy'],
                        'empty_predictions': sum(p == [] for p in predictions)})
        print(json.dumps(results[-1]), flush=True)
    best = max(results, key=lambda r: (r['slot_value_f1'], r['act_f1'], r['json_schema_valid_rate'], -r['iteration']))
    winner = adapters / 'melhor'
    winner.mkdir(exist_ok=True)
    shutil.copy(adapters / 'adapter_config.json', winner)
    shutil.copy(adapters / best['checkpoint'], winner / 'adapters.safetensors')
    write_json(EVIDENCE / run / 'selecao_checkpoint.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'run': run,
        'criterion': 'max slot_value micro F1 on fixed validation sample; ties: act F1, JSON validity, earlier step',
        'validation_sample': {'n': n, 'seed': 42, 'sha256': sha256(path),
                              'empty_gold': sum(not g for g in gold),
                              'sample_ids': [r['dialogue_id'] + ':' + r['uid'] for r in sample]},
        'checkpoints': results, 'selected': best})
    print('selecionado:', best['checkpoint'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--n', type=int, default=80)
    a = parser.parse_args()
    select(a.run, a.n)
