"""Reproducible MLX pilot with length filtering, process logs and weight hashes."""
import argparse
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from motor.data import (
    CODE_ROOT,
    EVIDENCE,
    read_jsonl,
    sha256,
    supervised,
    write_json,
    write_jsonl,
)
from motor.model import MODEL_ID, MODEL_REVISION, configure_tokenizer


class TrainingRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.write_text('')

    def record(self, phase, info):
        with self.path.open('a') as stream:
            stream.write(json.dumps({'phase': phase, **info}) + '\n')

    def on_train_loss_report(self, info):
        self.record('train', info)

    def on_val_loss_report(self, info):
        self.record('validation', info)


def worker(config_path):
    import mlx.core as mx
    import numpy as np
    import yaml
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset

    config = dict(CONFIG_DEFAULTS)
    config.update(yaml.safe_load(Path(config_path).read_text()))
    args = SimpleNamespace(**config)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)  # train_model also resets this before constructing LoRA layers.
    model, tokenizer = load(args.model)
    configure_tokenizer(tokenizer)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    recorder = TrainingRecorder(EVIDENCE / 'treinamento_metricas.jsonl')
    train_model(args, model, train_set, valid_set, training_callback=recorder)


def prepare_pilot(tokenizer, max_length=512):
    configure_tokenizer(tokenizer)
    selected = {}
    manifest = {'max_tokens': max_length, 'seed': 42, 'selection': 'seeded shuffle after full-message length filter'}
    fixed_path = CODE_ROOT / 'configs/piloto_ids.json'
    fixed_ids = json.loads(fixed_path.read_text()) if fixed_path.exists() else None
    if fixed_ids:
        manifest['selection'] = 'IDs fixed from experiment 01 to isolate the chat-template software correction'
        manifest['fixed_ids_sha256'] = sha256(fixed_path)
    for split, limit in [('train', 256), ('valid', 16)]:
        rows = read_jsonl(CODE_ROOT / 'data' / 'prepared' / (split + '_rows.jsonl'))
        rng = random.Random(42)
        eligible = []
        for row in rows:
            message = supervised(row)
            length = len(tokenizer.apply_chat_template(message['messages'], return_dict=False))
            if length <= max_length:
                eligible.append((row, message, length))
        rng.shuffle(eligible)
        if fixed_ids:
            by_id = {item[0]['dialogue_id'] + ':' + item[0]['uid']: item for item in eligible}
            chosen = [by_id[key] for key in fixed_ids[split]]
        else:
            chosen = eligible[:limit]
        if len(chosen) < limit:
            raise ValueError('Insufficient short samples for ' + split)
        selected[split] = [item[1] for item in chosen]
        path = CODE_ROOT / 'data' / 'pilot' / (split + '.jsonl')
        write_jsonl(path, selected[split])
        manifest[split] = {'all_rows': len(rows), 'eligible': len(eligible), 'selected': len(chosen),
                           'excluded_over_length': len(rows) - len(eligible),
                           'max_selected_tokens': max(item[2] for item in chosen),
                           'min_selected_tokens': min(item[2] for item in chosen),
                           'sample_ids': [item[0]['dialogue_id'] + ':' + item[0]['uid'] for item in chosen],
                           'sha256': sha256(path)}
    write_json(EVIDENCE / 'manifesto_piloto.json', manifest)
    return manifest


def train():
    import psutil
    import yaml
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config_path = CODE_ROOT / 'configs' / 'lora_piloto.yaml'
    config = yaml.safe_load(config_path.read_text())
    prepare_pilot(tokenizer, config['max_seq_length'])
    config.update(model=model_path, data=str(CODE_ROOT / 'data' / 'pilot'),
                  adapter_path=str(CODE_ROOT / 'artifacts' / 'lora_piloto'))
    runtime_config = CODE_ROOT / 'artifacts' / 'lora_runtime.yaml'
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(yaml.safe_dump(config))
    command = [sys.executable, '-u', '-m', 'motor.train', '--worker', str(runtime_config)]
    started = time.perf_counter()
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                'config': config, 'config_source_sha256': sha256(config_path),
                'data_manifest_sha256': sha256(EVIDENCE / 'manifesto_dados.json'),
                'pilot_manifest_sha256': sha256(EVIDENCE / 'manifesto_piloto.json'),
                'template_source_sha256': sha256(CODE_ROOT / 'configs/chat_template_oficial.jinja'),
                'source_files_sha256': {path.name: sha256(path) for path in
                                        (CODE_ROOT / 'motor').glob('*.py')},
                'python': sys.version, 'platform': platform.platform(),
                'machine': platform.machine(), 'physical_memory_bytes': psutil.virtual_memory().total,
                'versions': {name: importlib.metadata.version(name) for name in
                             ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'huggingface-hub', 'numpy')},
                'model_files': {path.name: sha256(path) for path in
                                Path(model_path).iterdir() if path.is_file()}}
    peak_rss = [0]
    stopped = threading.Event()
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    with (EVIDENCE / 'treinamento.log').open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=environment)

        def monitor():
            child = psutil.Process(process.pid)
            while not stopped.wait(.2):
                try:
                    peak_rss[0] = max(peak_rss[0], child.memory_info().rss)
                except psutil.NoSuchProcess:
                    return

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
        stopped.set()
        watcher.join()
    manifest.update(exit_code=code, wall_seconds=time.perf_counter() - started,
                    peak_child_rss_bytes=peak_rss[0],
                    artifacts={str(path.relative_to(CODE_ROOT)): {'sha256': sha256(path), 'bytes': path.stat().st_size}
                               for path in (CODE_ROOT / 'artifacts' / 'lora_piloto').glob('*') if path.is_file()},
                    log_sha256=sha256(EVIDENCE / 'treinamento.log'))
    events_path = EVIDENCE / 'treinamento_metricas.jsonl'
    if events_path.exists():
        events = read_jsonl(events_path)
        manifest['training_events'] = events
        manifest['peak_mlx_allocated_gb'] = max((e.get('peak_memory', 0) for e in events), default=0)
        manifest['training_events_sha256'] = sha256(events_path)
    write_json(EVIDENCE / 'execucao_treino.json', manifest)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.worker)
    else:
        train()
