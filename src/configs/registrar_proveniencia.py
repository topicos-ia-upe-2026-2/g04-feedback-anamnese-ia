"""Attach observed provenance after all three runs; never modify predictions."""
import json
from datetime import datetime, timezone

from motor.data import CODE_ROOT, EVIDENCE, sha256, write_json


def main():
    training = json.loads((EVIDENCE / 'execucao_treino.json').read_text())
    data = json.loads((EVIDENCE / 'manifesto_dados.json').read_text())
    if training['exit_code'] != 0:
        raise ValueError('Training did not succeed')
    adapter = 'artifacts/lora_piloto/adapters.safetensors'
    adapter_hash = sha256(CODE_ROOT / adapter)
    if adapter_hash != training['artifacts'][adapter]['sha256']:
        raise ValueError('Adapter changed since training')
    evaluator_hash = sha256(CODE_ROOT / 'motor/evaluate.py')
    if evaluator_hash != training['source_files_sha256']['evaluate.py']:
        raise ValueError('Evaluator changed since the training manifest was captured')
    for variant in ('b0', 'b1', 'lora'):
        path = EVIDENCE / ('metricas_' + variant + '.json')
        result = json.loads(path.read_text())
        if result['holdout_sha256'] != data['holdout']['sha256']:
            raise ValueError('Holdout changed')
        if result['predictions_sha256'] != sha256(CODE_ROOT / 'artifacts' / ('predictions_' + variant + '.jsonl')):
            raise ValueError('Prediction artifact changed')
        result.update(
            model_id=training['model_id'] if variant != 'b0' else None,
            model_revision=training['model_revision'] if variant != 'b0' else None,
            model_sha256=training['model_files']['model.safetensors'] if variant != 'b0' else None,
            adapter_sha256=adapter_hash if variant == 'lora' else None,
            chat_template_sha256=training['template_source_sha256'] if variant != 'b0' else None,
            holdout_status='Diagnostic: inspected to debug experiment 01 template; not a blind test.',
            source_evaluate_sha256=evaluator_hash,
            provenance_observed_utc=datetime.now(timezone.utc).isoformat(),
            provenance_note='Metadata appended after execution, with source and artifact hash checks. '
                            'The adapter matches the saved training checkpoint and the evaluator matches '
                            'the training source manifest. Base weight hash comes from that manifest. '
                            'Prediction files are unchanged.',
        )
        write_json(path, result)
    print('Provenance attached; prediction file hashes unchanged.')


if __name__ == '__main__':
    main()
