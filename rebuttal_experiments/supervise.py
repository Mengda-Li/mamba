"""Run independent GPU queues and persist process/exit status; no polling service."""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def queue(jobs, root, working_directory):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for job in jobs:
        status = {'name': job['name'], 'gpu': job.get('gpu'), 'command': job['command'],
                  'state': 'waiting', 'created_utc': now()}
        status_path = root / 'status' / (job['name'] + '.json')
        write(status_path, status)
        deadline = time.monotonic() + job.get('wait_timeout_seconds', 1800)
        while not all(Path(p).is_file() for p in job.get('requires_files', [])):
            if time.monotonic() > deadline:
                status.update(state='prerequisite_timeout', ended_utc=now())
                write(status_path, status)
                return False
            time.sleep(30)
        log_path = root / 'logs' / (job['name'] + '.log')
        with log_path.open('a', buffering=1) as log:
            process = subprocess.Popen(job['command'], cwd=working_directory,
                                       env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
            status.update(state='running', pid=process.pid, started_utc=now(), log=str(log_path))
            write(status_path, status)
            code = process.wait()
        status.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_utc=now())
        write(status_path, status)
        if code:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan['root'])
    working_directory = Path(plan.get('working_directory', REPO_ROOT)).resolve()
    if not working_directory.is_dir():
        raise FileNotFoundError(working_directory)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    # Register all queued work before any child starts, so status is inspectable.
    for jobs in plan['queues']:
        for job in jobs:
            write(root / 'status' / (job['name'] + '.json'),
                  {'name': job['name'], 'gpu': job.get('gpu'), 'state': 'queued',
                   'command': job['command']})
    with ThreadPoolExecutor(max_workers=len(plan['queues'])) as pool:
        futures = [pool.submit(queue, jobs, root, working_directory) for jobs in plan['queues']]
        outcomes = [f.result() for f in futures]
    write(root / 'batch_status.json', {'queue_success': outcomes, 'ended_utc': now()})
    if plan.get('analysis_command') and all(outcomes):
        with (root / 'logs' / 'analysis.log').open('a') as log:
            code = subprocess.call(plan['analysis_command'], cwd=working_directory,
                                   stdout=log, stderr=subprocess.STDOUT)
        write(root / 'analysis_status.json', {'exit_code': code, 'ended_utc': now()})


if __name__ == '__main__':
    main()
