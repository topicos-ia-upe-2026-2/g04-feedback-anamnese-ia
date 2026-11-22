"""Regressions for label alignment, leakage and explicit model failures."""
import importlib
import json

import pytest


def api(module, name):
    try:
        loaded = importlib.import_module('motor.' + module)
    except ModuleNotFoundError:
        pytest.fail('Implementation missing: motor.' + module)
    assert hasattr(loaded, name), 'Missing production function: ' + name
    return getattr(loaded, name)


def test_uid_alignment_uses_id_not_annotation_order_and_ignores_patient():
    build = api('data', 'build_samples')
    dialogs = {'a': {'utterances': [
        {'uttr_id': 10, 'speaker': 'doctor', 'text': 'When did the cough start?'},
        {'uttr_id': 11, 'speaker': 'patient', 'text': 'Yesterday'},
        {'uttr_id': 12, 'speaker': 'doctor', 'text': 'Thank you'},
    ]}}
    labels = {'a': [[{'uid': '11', 'intent': 'inform', 'slot': 'symptom', 'value': 'cough'}],
                    [{'uid': '12', 'intent': 'salutations'}],
                    [{'uid': '10', 'intent': 'inquire', 'slot': 'symptom',
                      'value': 'cough', 'when check': True, 'frquency check': True}]]}
    rows = build(dialogs, labels)
    assert len(rows) == 2
    assert rows[0]['acts'] == [{'slot': 'symptom', 'value': 'cough',
                               'attributes': ['frequency check', 'when check']}]
    assert rows[1]['acts'] == []
    assert rows[0]['history'] == []
    assert rows[1]['history'][-1]['text'] == 'Yesterday'


def test_missing_or_unknown_uid_is_not_silently_dropped():
    build = api('data', 'build_samples')
    dialogs = {'a': {'utterances': [{'uttr_id': 0, 'speaker': 'doctor', 'text': 'Hi'}]}}
    with pytest.raises(ValueError, match='uid'):
        build(dialogs, {'a': [[{'uid': '9', 'intent': 'salutations'}]]})
    with pytest.raises(ValueError, match='uid'):
        build(dialogs, {'a': []})


def test_splits_are_disjoint_complete_and_independent_of_input_order():
    split = api('data', 'split_dialogues')
    ids = [str(i) for i in range(231)]
    result = split(ids, seed=42)
    assert result == split(list(reversed(ids)), seed=42)
    assert {key: len(value) for key, value in result.items()} == {'train': 185, 'valid': 23, 'test': 23}
    assert not set(result['train']) & set(result['valid'])
    assert not set(result['train']) & set(result['test'])
    assert set().union(*map(set, result.values())) == set(ids)


def test_identical_dialogues_cannot_cross_split_even_when_ids_differ():
    split = api('data', 'split_dialogue_groups')
    dialogs = {str(i): {'utterances': [
        {'speaker': 'doctor', 'text': 'case ' + str(i // 2), 'uttr_id': 0}
    ]} for i in range(80)}
    result, groups = split(dialogs, seed=42)
    assert len(groups) == 40
    membership = {key: name for name, ids in result.items() for key in ids}
    assert len(membership) == 80
    for i in range(0, 80, 2):
        assert membership[str(i)] == membership[str(i + 1)]


def test_modified_raw_file_is_rejected(tmp_path):
    verify = api('data', 'verify_source')
    path = tmp_path / 'dialogs.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='SHA-256'):
        verify(path, 'expectedhash')


@pytest.mark.parametrize('raw', ['not JSON', '{}', '[{"slot":"symptom"}]',
                                '[{"slot":"symptom","slot":"habit","value":"smoke","attributes":[]}]',
                                '[{"slot":"invented","value":"x","attributes":[]}]',
                                '[{"slot":"symptom","value":"x","attributes":["invented"]}]',
                                '[{"intent":"inform","slot":"symptom","value":"x","attributes":[]}]'])
def test_invalid_generation_is_explicit_error(raw):
    parse = api('model', 'parse_output')
    with pytest.raises(ValueError):
        parse(raw)


def test_valid_json_preserves_empty_output_and_rejects_markdown_repair():
    parse = api('model', 'parse_output')
    assert parse('[]') == []
    item = {'slot': 'symptom', 'value': 'cough', 'attributes': ['when check']}
    assert parse(json.dumps([item])) == [item]
    with pytest.raises(ValueError):
        parse('```json\n[]\n```')


def test_metrics_keep_invalid_predictions_in_recall_denominator():
    score = api('evaluate', 'score_predictions')
    act = {'slot': 'symptom', 'value': 'cough', 'attributes': ['when check']}
    result = score([[act], [act], []], [[act], None, []])
    assert result['json_schema_valid_rate'] == pytest.approx(2 / 3)
    assert result['act_micro']['precision'] == 1.0
    assert result['act_micro']['recall'] == 0.5
    assert result['act_micro']['f1'] == pytest.approx(2 / 3)
    assert result['act_micro']['fn'] == 1
    assert result['attribute_micro']['recall'] == 0.5
    with pytest.raises(ValueError):
        score([[act]], [])


def test_macro_uses_same_gold_supported_slots_for_all_systems():
    score = api('evaluate', 'score_predictions')
    symptom = {'slot': 'symptom', 'value': 'cough', 'attributes': []}
    habit = {'slot': 'habit', 'value': 'smoking', 'attributes': []}
    clean = score([[symptom]], [[symptom]])
    hallucinated = score([[symptom]], [[symptom, habit]])
    assert clean['macro_slots'] == hallucinated['macro_slots'] == ['symptom']
    assert hallucinated['slot_value_macro_f1'] == 1.0
    assert hallucinated['slot_value_micro']['precision'] == .5
    assert hallucinated['per_slot']['habit']['fp'] == 1


def test_training_metrics_are_written_without_external_telemetry(tmp_path):
    recorder = api('train', 'TrainingRecorder')(tmp_path / 'metrics.jsonl')
    recorder.on_train_loss_report({'iteration': 5, 'train_loss': 1.2, 'peak_memory': 3.4})
    recorder.on_val_loss_report({'iteration': 0, 'val_loss': 2.3})
    events = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert events == [{'phase': 'train', 'iteration': 5, 'train_loss': 1.2, 'peak_memory': 3.4},
                      {'phase': 'validation', 'iteration': 0, 'val_loss': 2.3}]


def test_real_tokenizer_supervised_target_is_json_without_hidden_reasoning():
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    model = importlib.import_module('motor.model')
    cached = try_to_load_from_cache(model.MODEL_ID, 'tokenizer_config.json', revision=model.MODEL_REVISION)
    if not isinstance(cached, str):
        pytest.skip('Integration test requires the pinned tokenizer in local HF cache')
    from pathlib import Path
    tokenizer = AutoTokenizer.from_pretrained(str(Path(cached).parent), local_files_only=True)
    messages = model.build_messages('Do you smoke?')
    original_prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    if hasattr(model, 'configure_tokenizer'):
        model.configure_tokenizer(tokenizer)
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': '[]'}], return_dict=False)
    assert prefix == original_prefix
    assert full[:len(prefix)] == prefix
    continuation = tokenizer.decode(full[len(prefix):])
    assert '<think>' not in continuation
    assert continuation == '[]<|im_end|>\n'


def test_prompt_contains_only_past_context_and_fixed_inquire_contract():
    messages = api('model', 'build_messages')('Do you smoke?', [{'speaker': 'patient', 'text': 'I cough'}])
    assert messages[0]['role'] == 'system'
    assert 'inquire' in messages[0]['content']
    payload = json.loads(messages[1]['content'])
    assert payload['current_doctor_turn'] == 'Do you smoke?'
    assert payload['history'] == [{'speaker': 'patient', 'text': 'I cough'}]
