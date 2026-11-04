"""Use a locally validated Python runtime for this machine's affected environment.

The manifest is machine-local, not an assertion that all Python 3.11.11 builds
are faulty. Model math, CUDA precision and checkpoint formats are unchanged.
"""
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = PROJECT_ROOT / '.runtime/training-runtime.json'


def maybe_reexec():
    if not RUNTIME_CONFIG.exists():
        return
    config = json.loads(RUNTIME_CONFIG.read_text())
    source = Path(config['source_executable']).resolve()
    if Path(sys.executable).resolve() != source:
        return  # Includes the new interpreter, preventing re-exec loops.
    executable = PROJECT_ROOT / config['runtime_python']
    packages = Path(config['source_site_packages'])
    if not executable.is_file() or not packages.is_dir():
        raise RuntimeError(f'Configured training runtime is incomplete: {executable}; see motion_generation/README.md')
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    # Reuse the verified CPython 3.11 PyTorch/NumPy wheels from the source env.
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT_ROOT), str(packages), env.get('PYTHONPATH', '')])
    env['PYTHONNOUSERSITE'] = '1'
    print(f'[training runtime] Switching from {sys.executable} to {executable}', flush=True)
    os.execve(str(executable), [str(executable), '-u', '-m', 'motion_generation.train', *sys.argv[1:]], env)
