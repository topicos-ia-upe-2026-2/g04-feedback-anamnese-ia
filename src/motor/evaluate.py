"""Frozen holdout comparison; exact set matching, not the paper's NLU metric."""
import argparse
import json
import re
import resource
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from motor.data import CODE_ROOT, EVIDENCE, read_jsonl, sha256, write_json
from motor.model import MLXExtractor, parse_output


def canonical(text):
    return ' '.join(text.casefold().split())


def act_set(acts, mode='act'):
    output = set()
    for act in acts:
        slot, value = canonical(act['slot']), canonical(act['value'])
        attrs = tuple(sorted(set(act['attributes'])))
        if mode == 'slot_value':
            output.add((slot, value))
        elif mode == 'attribute':
            output.update((slot, value, attr) for attr in attrs)
        else:
            output.add((slot, value, attrs))
    return output


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def score_predictions(gold, predictions):
    if len(gold) != len(predictions) or not gold:
        raise ValueError('Gold and predictions must have equal, nonzero lengths')
    result = {'n': len(gold), 'invalid_outputs': sum(p is None for p in predictions),
              'json_schema_valid_rate': sum(p is not None for p in predictions) / len(gold)}
    by_slot = {}
    for mode in ('act', 'slot_value', 'attribute'):
        tp = fp = fn = 0
        for expected, predicted in zip(gold, predictions):
            # Invalid prediction stays explicit as None, and contributes all its gold false negatives.
            g, p = act_set(expected, mode), act_set(predicted or [], mode)
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
            if mode == 'slot_value':
                for slot in {entry[0] for entry in g | p}:
                    sg, sp = {v for v in g if v[0] == slot}, {v for v in p if v[0] == slot}
                    counts = by_slot.setdefault(slot, Counter())
                    counts.update(tp=len(sg & sp), fp=len(sp - sg), fn=len(sg - sp))
        result[mode + '_micro'] = prf(tp, fp, fn)
    result['per_slot'] = {slot: prf(**counts) for slot, counts in sorted(by_slot.items())}
    result['macro_slots'] = sorted({act['slot'] for row in gold for act in row})
    result['macro_convention'] = 'Mean over gold-supported slots, fixed across systems; zero_division=0'
    result['slot_value_macro_f1'] = (
        sum(result['per_slot'][slot]['f1'] for slot in result['macro_slots'])
        / len(result['macro_slots']) if result['macro_slots'] else 0)
    result['exact_turn_accuracy'] = sum(
        p is not None and act_set(g) == act_set(p) for g, p in zip(gold, predictions)) / len(gold)
    return result


class KeywordBaseline:
    """B0 dictionary built from training labels only; exact phrases in current question."""

    def __init__(self, train_rows):
        self.vocabulary = sorted({(act['slot'], canonical(act['value']))
                                  for row in train_rows for act in row['acts']
                                  if len(act['value'].strip()) >= 3 and ',' not in act['value']})

    def __call__(self, text, history=None):
        lowered = canonical(text)
        if '?' not in text and not re.search(r'\b(tell me|can you|do you|have you|how|what|when|where)\b', lowered):
            return []
        attrs = []
        for pattern, attr in [
            (r'\b(when|start|began)\b', 'when check'),
            (r'\b(where|location)\b', 'location check'),
            (r'\b(how often|frequency)\b', 'frequency check'),
            (r'\b(better|worse|relieve)\b', 'ana factors check'),
            (r'\b(describe|what.*like)\b', 'characteristics check'),
        ]:
            if re.search(pattern, lowered):
                attrs.append(attr)
        return [{'slot': slot, 'value': value, 'attributes': sorted(attrs)}
                for slot, value in self.vocabulary
                if re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', lowered)]


def evaluate(variant, adapter=None, rows_file=None, evidence_dir=None, tag=None):
    """Sem opções, reproduz a avaliação do holdout; opções permitem outra rodada ou outro conjunto."""
    holdout_path = CODE_ROOT / 'data' / 'prepared' / 'holdout_rows.jsonl'
    rows_path = CODE_ROOT / rows_file if rows_file else holdout_path
    rows = read_jsonl(rows_path)
    data_manifest = json.loads((EVIDENCE / 'manifesto_dados.json').read_text())
    if rows_path == holdout_path and sha256(holdout_path) != data_manifest['holdout']['sha256']:
        raise ValueError('Holdout SHA-256 differs from frozen manifest')
    evidence_dir = Path(evidence_dir) if evidence_dir else EVIDENCE
    evidence_dir.mkdir(parents=True, exist_ok=True)
    name = tag or variant
    output_path = CODE_ROOT / 'artifacts' / (evidence_dir.name if evidence_dir != EVIDENCE else '') / ('predictions_' + name + '.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if variant == 'b0':
        backend = KeywordBaseline(read_jsonl(CODE_ROOT / 'data' / 'prepared' / 'train_rows.jsonl'))
    else:
        adapter = adapter or (str(CODE_ROOT / 'artifacts' / 'lora_piloto') if variant == 'lora' else None)
        backend = MLXExtractor(adapter_path=adapter)
    load_seconds = time.perf_counter() - started
    predictions, records = [], []
    with output_path.open('w') as stream:
        for row in rows:
            tick = time.perf_counter()
            error, raw, predicted = None, None, None
            try:
                if variant == 'b0':
                    predicted = backend(row['text'], row['history'])
                    raw = json.dumps(predicted)
                else:
                    raw = backend.generate_raw(row['text'], row['history'])
                    predicted = parse_output(raw)
            except ValueError as exc:
                error = str(exc)
            duration = time.perf_counter() - tick
            record = {'sample_id': row['dialogue_id'] + ':' + row['uid'],
                      'gold': row['acts'], 'prediction': predicted, 'raw_output': raw,
                      'error': error, 'seconds': duration}
            predictions.append(predicted)
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{variant} {len(predictions)}/{len(rows)} valid={error is None} {duration:.2f}s', flush=True)
    result = score_predictions([row['acts'] for row in rows], predictions)
    result.update(variant=variant, tag=name, adapter=adapter, rows_file=str(rows_path.relative_to(CODE_ROOT)),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  holdout_sha256=sha256(holdout_path) if rows_path == holdout_path else None,
                  rows_sha256=sha256(rows_path), predictions_sha256=sha256(output_path),
                  empty_predictions=sum(p == [] for p in predictions),
                  load_seconds=load_seconds, total_seconds=time.perf_counter() - started,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  latency_seconds={'mean': sum(r['seconds'] for r in records) / len(records),
                                   'min': min(r['seconds'] for r in records),
                                   'max': max(r['seconds'] for r in records)},
                  decoding={'temperature': 0, 'max_tokens': 192, 'seed': 42},
                  metric_definition='Exact micro sets per turn: act=(slot,value,sorted attributes); '
                                    'slot_value=(slot,value); attribute=(slot,value,attribute). '
                                    'Casefold/whitespace only. Invalid JSON contributes gold FN. '
                                    'Not the original paper metric.',
                  label_budget='B0 vocabulary from all training rows; LoRA pilot subset256/40steps. '
                               'This comparison does not equalize annotation budget.',
                  per_sample=[{'sample_id': r['sample_id'], 'valid': r['error'] is None,
                               'error': r['error'], 'seconds': r['seconds']} for r in records])
    if variant != 'b0':
        import mlx.core as mx
        result['peak_mlx_allocated_bytes'] = mx.get_peak_memory()
    write_json(evidence_dir / ('metricas_' + name + '.json'), result)
    print(json.dumps({k: result[k] for k in ('variant', 'n', 'json_schema_valid_rate', 'act_micro', 'slot_value_micro')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('b0', 'b1', 'lora'))
    parser.add_argument('--adapter', help='Pasta do adaptador (padrão: artifacts/lora_piloto)')
    parser.add_argument('--rows', help='Arquivo *_rows.jsonl relativo a src/ (padrão: holdout)')
    parser.add_argument('--evidence', help='Pasta de evidências da rodada (padrão: evidencias/)')
    parser.add_argument('--tag', help='Nome dos arquivos de saída (padrão: a variante)')
    a = parser.parse_args()
    evaluate(a.variant, a.adapter, a.rows, a.evidence, a.tag)
