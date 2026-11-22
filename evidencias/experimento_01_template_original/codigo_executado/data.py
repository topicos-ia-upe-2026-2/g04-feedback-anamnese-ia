"""Prepare MediTOD inquiry data locally; no redistribution rights are asserted."""
import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from motor.model import ATTRIBUTES, SLOTS, build_messages

SOURCE_COMMIT = '0b7a87c3553c9056bc8371b0809811003f94b261'
SOURCE_HASHES = {
    'dialogs.json': '9d79ecaa10e8da630f9accdb52a1be9a7b9bd812b9e3d19b77579e320a4123b6',
    'annotations.json': '828a802da0c5d1c42e92ac15b0fb79b2d34c20e78c9d1c70e1b4423424982c59',
}
CODE_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = CODE_ROOT.parent / 'evidencias'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError('SHA-256 mismatch: ' + str(path))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_dialogues(ids, seed=42):
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    n_test = int(len(ids) * .1)
    n_valid = int(len(ids) * .1)
    return {'train': sorted(ids[n_test + n_valid:]),
            'valid': sorted(ids[n_test:n_test + n_valid]), 'test': sorted(ids[:n_test])}


def split_dialogue_groups(dialogs, seed=42):
    groups = {}
    for key, dialog in sorted(dialogs.items()):
        content = [(turn['speaker'], turn['text']) for turn in dialog['utterances']]
        digest = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
        groups.setdefault(digest, []).append(key)
    split_hashes = split_dialogues(groups, seed)
    splits = {name: sorted(key for digest in hashes for key in groups[digest])
              for name, hashes in split_hashes.items()}
    return splits, groups


def normalize_act(act):
    attrs = sorted({'frequency check' if key == 'frquency check' else key
                    for key, value in act.items() if key.endswith(' check') and value is True})
    if act['slot'] not in SLOTS or not set(attrs) <= set(ATTRIBUTES):
        raise ValueError('Unknown CMAS category')
    value = act.get('value', '')
    if not isinstance(value, str):
        raise TypeError('Non-string CMAS value')
    return {'slot': act['slot'], 'value': ' '.join(value.split()), 'attributes': attrs}


def build_samples(dialogs, annotations):
    if set(dialogs) != set(annotations):
        raise ValueError('Dialogue IDs differ between text and annotation files')
    samples = []
    for dialogue_id in sorted(dialogs):
        turns = dialogs[dialogue_id]['utterances']
        ids = [str(turn['uttr_id']) for turn in turns]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate utterance uid')
        by_uid = {}
        for group in annotations[dialogue_id]:
            for act in group:
                uid = str(act.get('uid'))
                if uid not in ids:
                    raise ValueError('Unknown annotation uid: ' + uid)
                by_uid.setdefault(uid, []).append(act)
        if set(ids) != set(by_uid):
            raise ValueError('Missing annotation uid')
        history = []
        for turn in turns:
            uid = str(turn['uttr_id'])
            if turn['speaker'] == 'doctor':
                acts = [normalize_act(act) for act in by_uid[uid] if act['intent'] == 'inquire']
                samples.append({'dialogue_id': dialogue_id, 'uid': uid, 'text': turn['text'],
                                'history': history[-2:], 'acts': acts})
            if turn['speaker'] in ('doctor', 'patient'):
                history.append({'speaker': turn['speaker'], 'text': turn['text']})
    return samples


def supervised(row):
    return {'messages': build_messages(row['text'], row['history']) + [
        {'role': 'assistant', 'content': json.dumps(row['acts'], ensure_ascii=False)}]}


def select_holdout(rows, seed=42):
    # Fixed diagnostic strata, selected before model execution: 4 empty / 8 attribute / 12 other.
    rng = random.Random(seed)
    pools = {'empty': [], 'attribute': [], 'other': []}
    for row in rows:
        category = 'empty' if not row['acts'] else (
            'attribute' if any(act['attributes'] for act in row['acts']) else 'other')
        pools[category].append(row)
    result = []
    for category, count in [('empty', 4), ('attribute', 8), ('other', 12)]:
        rng.shuffle(pools[category])
        if len(pools[category]) < count:
            raise ValueError('Insufficient holdout stratum: ' + category)
        result.extend(pools[category][:count])
    rng.shuffle(result)
    return result


def prepare():
    raw = CODE_ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ('dialogs.json', 'annotations.json'):
        url = f'https://raw.githubusercontent.com/dair-iitd/MediTOD/{SOURCE_COMMIT}/raw_data/{name}'
        path = raw / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        verify_source(path, SOURCE_HASHES[name])
        sources[name] = {'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size}
    dialogs = json.loads((raw / 'dialogs.json').read_text())
    annotations = json.loads((raw / 'annotations.json').read_text())
    rows = build_samples(dialogs, annotations)
    splits, groups = split_dialogue_groups(dialogs)
    write_json(EVIDENCE / 'splits_dialogos.json', splits)
    sets = {}
    for name, ids in splits.items():
        subset = [row for row in rows if row['dialogue_id'] in ids]
        sets[name] = subset
        write_jsonl(CODE_ROOT / 'data' / 'prepared' / (name + '_rows.jsonl'), subset)
        path = CODE_ROOT / 'data' / 'prepared' / (name + '.jsonl')
        write_jsonl(path, [supervised(row) for row in subset])
    holdout = select_holdout(sets['test'])
    write_jsonl(CODE_ROOT / 'data' / 'prepared' / 'holdout_rows.jsonl', holdout)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': SOURCE_COMMIT,
        'license_status': 'No dataset LICENSE found; no redistribution authorization asserted. '
                          'Raw/prepared data and adapters remain local and gitignored.',
        'task': 'Retrospective extraction of doctor inquire acts, not paper patient inform NLU.',
        'language': 'English; no translation or human review performed in this pilot.',
        'seed': 42, 'history_turns': 2, 'sources': sources,
        'dialogue_counts': {key: len(value) for key, value in splits.items()},
        'split_method': 'seed42 80/10/10 by SHA-256 groups of full ordered (speaker,text) sequence',
        'unique_content_groups': len(groups),
        'duplicate_dialogue_groups': [ids for ids in groups.values() if len(ids) > 1],
        'group_counts': {name: sum(bool(set(ids) & set(members)) for ids in groups.values())
                         for name, members in splits.items()},
        'sample_counts': {key: len(value) for key, value in sets.items()},
        'total_doctor_turns': len(rows),
        'total_inquire_acts': sum(len(row['acts']) for row in rows),
        'source_inquire_acts_all_speakers': sum(act['intent'] == 'inquire'
            for groups in annotations.values() for group in groups for act in group),
        'inquire_excluded_non_doctor': sum(act['intent'] == 'inquire'
            for groups in annotations.values() for group in groups for act in group)
            - sum(len(row['acts']) for row in rows),
        'slot_counts': dict(Counter(act['slot'] for row in rows for act in row['acts'])),
        'normalization': {'frquency check': 'frequency check', 'missing_value': '',
                          'value_whitespace': 'collapsed', 'compound_values': 'preserved unsplit',
                          'only_attributes': 'Boolean True keys ending in check'},
        'all_annotation_uids_aligned': True, 'dialogue_split_intersections': 0,
        'full_dialogue_content_intersections': 0,
        'holdout': {'n': len(holdout), 'selection': 'seed42 stratified: 4 empty, 8 with attributes, 12 other',
                    'sample_ids': [row['dialogue_id'] + ':' + row['uid'] for row in holdout],
                    'sha256': sha256(CODE_ROOT / 'data' / 'prepared' / 'holdout_rows.jsonl')},
        'processed_files': {path.name: sha256(path) for path in
                            sorted((CODE_ROOT / 'data' / 'prepared').glob('*.jsonl'))},
    }
    write_json(EVIDENCE / 'manifesto_dados.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('dialogue_counts', 'sample_counts', 'total_inquire_acts')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
