"""Fetch the explicitly requested original Qwen VL diagnostic weights to scratch.

Run through run_logged; model loading and weight hashing remain in GPU jobs.
The Hub revision is resolved once and pinned throughout each download.
"""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import traceback

MODELS = {s: 'Qwen/Qwen2.5-VL-' + s.upper() + '-Instruct' for s in ('7b', '32b')}
PATTERNS = ('*.json', '*.safetensors', '*.model', '*.txt', '*.md', 'LICENSE', 'chat_template*')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--model-size', choices=MODELS, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not re.fullmatch(r'diag_diag02_model_[A-Za-z0-9_]+', args.name):
        parser.error('Require an isolated DIAG02 model download name')
    root = Path(__file__).resolve().parents[2]
    output = root / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError(output)
    scratch = Path(os.environ['CVPR_SCRATCH']).resolve()
    cache = Path(os.environ['HF_HUB_CACHE']).resolve()
    if not cache.is_relative_to(scratch):
        raise ValueError('All model files must stay under scratch')
    metrics = {'model': MODELS[args.model_size], 'completed': False,
               'snapshot_path': None, 'revision': None, 'errors': [], 'gpus': 0}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Diagnostic model download time limit reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout_seconds)
    code = 0
    try:
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(metrics['model'], files_metadata=True)
        if not re.fullmatch(r'[0-9a-f]{40}', info.sha):
            raise ValueError('Hub did not return a complete pinned revision')
        files = []
        for item in info.siblings:
            if '/' not in item.rfilename and any(fnmatch.fnmatch(item.rfilename, pat) for pat in PATTERNS):
                if not isinstance(item.size, int) or item.size < 0:
                    raise ValueError('Missing official file size: ' + item.rfilename)
                lfs = getattr(item, 'lfs', None)
                files.append({'path': item.rfilename, 'bytes': item.size,
                    'hub_lfs_sha256': getattr(lfs, 'sha256', None), 'hub_blob_id': item.blob_id})
        if not files or not any(f['path'].endswith('.safetensors') for f in files):
            raise ValueError('No official safetensors weights selected')
        metrics.update(revision=info.sha, expected_bytes=sum(f['bytes'] for f in files), files=files)
        print('MODEL_DOWNLOAD_MANIFEST_JSON: ' + json.dumps(metrics, sort_keys=True), flush=True)
        path = Path(snapshot_download(metrics['model'], revision=info.sha,
                    cache_dir=str(cache), allow_patterns=[f['path'] for f in files], max_workers=4)).resolve()
        if not path.is_relative_to(cache) or path.name != info.sha:
            raise ValueError('Unexpected downloaded snapshot path')
        for f in files:
            p = path / f['path']
            if not p.is_file() or not p.resolve().is_relative_to(cache) or p.stat().st_size != f['bytes']:
                raise ValueError('Incomplete model file: ' + f['path'])
        config_path = path / 'config.json'
        config = json.loads(config_path.read_text())
        if config.get('model_type') != 'qwen2_5_vl':
            raise ValueError('Wrong original model architecture')
        metrics.update(completed=True, snapshot_path=str(path),
            model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            local_weight_full_sha_check='deferred to native model_reference in GPU job; download sizes checked')
    except Exception as error:
        code = 1
        metrics['errors'].append({'type': type(error).__name__, 'message': str(error)})
        traceback.print_exc()
    finally:
        signal.alarm(0)
        metrics['elapsed_seconds'] = time.monotonic() - started
        config = {**vars(args), 'cache': str(cache), 'scratch': str(scratch),
                  'scope': 'Human-authorized DIAG02 original model re-download; no GPU or training'}
        output.write_text(json.dumps({'metrics': metrics, 'config': config}, indent=2) + '\n')
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
