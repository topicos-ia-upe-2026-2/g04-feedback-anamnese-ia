"""Lazy local MLX extractor. Invalid model output is always an explicit error."""
import json
import os
from functools import lru_cache
from threading import RLock

MODEL_ID = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
MODEL_REVISION = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
SLOTS = ('symptom', 'medical history', 'habit', 'medication', 'exposure',
         'family history', 'residence', 'travel', 'symptom (dermatology)',
         'medical test', 'occupation', 'other', 'basic information')
ATTRIBUTES = ('when check', 'location check', 'ana factors check',
              'characteristics check', 'progression check', 'frequency check',
              'other check', 'status check', 'response to check', 'value check',
              'color check', 'where check', 'duration check', 'volume check',
              'impact check', 'since when check', 'how check',
              'destination check', 'exposure check')
SYSTEM_PROMPT = (
    'Extract only inquire acts from current_doctor_turn. History resolves references; '
    'never credit history as a current question. Return ONLY a JSON array '
    'of {"slot":string,"value":string,"attributes":[string]}. '
    'Use [] when no inquiry. Value is the queried clinical concept in English; '
    'use "" if unspecified. No answers, explanations or markdown. '
    'Slots: ' + ', '.join(SLOTS) + '. Attributes (only what is asked): '
    + ', '.join(ATTRIBUTES) + '.'
)


class ModelOutputError(ValueError):
    """Generation cannot be trusted as a valid list of inquiry acts."""


def build_messages(text, history=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a nonempty string')
    history = [] if history is None else history
    if not isinstance(history, list):
        raise TypeError('history must be a list')
    for turn in history:
        if not isinstance(turn, dict) or not isinstance(turn.get('text'), str):
            raise TypeError('Invalid history turn')
        if turn.get('speaker') not in ('doctor', 'student', 'patient'):
            raise ValueError('Invalid history speaker')
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': json.dumps(
            {'history': history[-2:], 'current_doctor_turn': text}, ensure_ascii=False)},
    ]


def parse_output(raw):
    def strict_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ModelOutputError('Repeated JSON key: ' + key)
            output[key] = value
        return output

    try:
        acts = json.loads(raw, object_pairs_hook=strict_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelOutputError('Model output is not strict JSON') from exc
    if not isinstance(acts, list):
        raise ModelOutputError('Expected JSON array')
    for act in acts:
        if not isinstance(act, dict) or set(act) != {'slot', 'value', 'attributes'}:
            raise ModelOutputError('Each act must contain exactly slot, value, attributes')
        if act['slot'] not in SLOTS or not isinstance(act['value'], str):
            raise ModelOutputError('Unknown slot or invalid value')
        if not isinstance(act['attributes'], list) or any(
            not isinstance(attr, str) or attr not in ATTRIBUTES for attr in act['attributes']
        ):
            raise ModelOutputError('Unknown attribute or invalid attributes list')
    return acts


class MLXExtractor:
    """Callable backend; loads weights once and serializes GPU access."""

    def __init__(self, adapter_path=None, model_path=None, max_tokens=192):
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        self.lock = RLock()
        self.max_tokens = max_tokens
        path = model_path or os.environ.get('CMAS_MODEL_PATH') or snapshot_download(
            MODEL_ID, revision=MODEL_REVISION)
        self.model, self.tokenizer = load(path, adapter_path=adapter_path)

    def generate_raw(self, text, history=None):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        with self.lock:
            mx.random.seed(42)
            prompt = self.tokenizer.apply_chat_template(
                build_messages(text, history), tokenize=False, add_generation_prompt=True)
            return generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=self.max_tokens, sampler=make_sampler(temp=0),
                            verbose=False)

    def __call__(self, text, history=None):
        return parse_output(self.generate_raw(text, history))


@lru_cache(maxsize=1)
def get_extractor(adapter_path=None):
    return MLXExtractor(adapter_path=adapter_path)


def extract_turn(text, history=None, adapter_path=None):
    return get_extractor(adapter_path)(text, history)
