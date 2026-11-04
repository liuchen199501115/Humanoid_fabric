"""Build one archive while preserving source clip boundaries and validation split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .data import joint_spec, load_clip, load_contacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--with-contacts', action='store_true')
    p.add_argument('--fps', type=float, default=50.)
    p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-files', type=int, default=0)
    args = p.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0 or not 0 < args.validation_fraction < 1 or args.max_files < 0:
        p.error('Require positive fps, validation fraction in (0,1), nonnegative max-files')
    names, limits = joint_spec()
    files = sorted(args.input.rglob('*.pkl'))
    if args.max_files:
        files = files[:args.max_files]
    clips, paths, hashes, rejected = [], [], set(), []
    contacts = []
    for i, path in enumerate(files):
        try:
            x = load_clip(path, args.fps, names)
            if len(x) < 27:
                raise ValueError('Fewer than 27 resampled frames')
            digest = hashlib.sha256(x.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError('Duplicate resampled motion')
            if args.with_contacts:
                contact = load_contacts(path, args.fps, len(x))
                contacts.append(contact)
            hashes.add(digest)
            clips.append(x)
            paths.append(str(path.resolve()))
        except (ValueError, KeyError, OSError, EOFError, TypeError) as exc:
            rejected.append({'path': str(path), 'reason': str(exc)})
        if (i+1) % 100 == 0:
            print(f'Processed {i+1}/{len(files)}, accepted {len(clips)}', flush=True)
    if len(clips) < 2:
        raise ValueError('Need at least two distinct valid clips for train/validation')
    split = np.zeros(len(clips), dtype=np.int8)
    rng = np.random.default_rng(args.seed)
    nval = min(len(clips)-1, max(1, round(len(clips)*args.validation_fraction)))
    split[rng.permutation(len(clips))[:nval]] = 1
    offsets = np.cumsum([0]+[len(x) for x in clips])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    states = np.concatenate(clips)
    np.savez_compressed(args.output, states=states, offsets=offsets, split=split,
                        paths=np.asarray(paths), fps=args.fps, dof_names=np.asarray(names),
                        joint_limits=limits, format_version=2 if args.with_contacts else 1,
                        **({'contacts': np.concatenate(contacts)} if args.with_contacts else {}))
    report = {'accepted_clips': len(clips), 'train_clips': int((split == 0).sum()),
              'validation_clips': int(split.sum()), 'frames': len(states),
              'hours': len(states)/args.fps/3600, 'fps': args.fps,
              'joint_limit_violation_fraction': float(((states[:, 9:] < limits[:, 0]) | (states[:, 9:] > limits[:, 1])).mean()),
              'rejected': rejected}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'rejected'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
