"""Verify and restore the frozen scientific records shipped with this repository.

Python 3.9+; standard library only. This script does not start experiments.
  python restore_materials.py --verify-only
  python restore_materials.py --output restored
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_join(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or '\\' in relative:
        raise RuntimeError('Unsafe path in manifest: ' + relative)
    result = (root / Path(*part.parts)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise RuntimeError('Path escapes output directory: ' + relative)
    return result

def checked_file(root, item):
    path = safe_join(root, item['path'])
    data = path.read_bytes()
    if len(data) != item['bytes'] or digest(data) != item['sha256']:
        raise RuntimeError('Size or SHA-256 mismatch: ' + item['path'])
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--verify-only', action='store_true')
    choice.add_argument('--output', type=Path, help='New directory; must not already exist')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'integrity_manifest.json').read_text(encoding='utf-8'))
    target = args.output.resolve() if args.output else None
    if target and target.exists():
        raise RuntimeError('Output already exists; choose a new directory: ' + str(target))
    checked = 0
    jobs = {key:0 for key in manifest['merge_order']}
    buffers = {}
    for item in manifest['archives']:
        data = b''.join(checked_file(root, part) for part in item['parts'])
        if len(data) != item['bytes'] or digest(data) != item['sha256']:
            raise RuntimeError('Reassembled archive mismatch: ' + item['archive_name'])
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            actual_names = z.namelist()
            expected_names = [member['path'] for member in item['members']]
            if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(expected_names):
                raise RuntimeError('Archive member list mismatch: ' + item['archive_name'])
            for member in item['members']:
                content = z.read(member['path'])
                if len(content) != member['bytes'] or digest(content) != member['sha256']:
                    raise RuntimeError('Member mismatch: ' + item['id'] + '/' + member['path'])
                if member['path'].startswith(('results/jobs/', 'results/jobs_extra/')) and member['path'].endswith('.json'):
                    jobs[item['id']] += 1
                checked += 1
        buffers[item['id']] = data
        print('Verified', item['id'], item['member_count'], 'files;', jobs[item['id']], 'completed job records')
    expected_jobs = {key:value for key,value in manifest['completed_analysis_jobs'].items() if key != 'total'}
    if jobs != expected_jobs or sum(jobs.values()) != manifest['completed_analysis_jobs']['total']:
        raise RuntimeError('Job count mismatch')
    source_item = manifest['source_tree']
    source_data = checked_file(root, source_item)
    with zipfile.ZipFile(io.BytesIO(source_data)) as z:
        if set(z.namelist()) != {x['path'] for x in manifest['source_overlay']}:
            raise RuntimeError('Source snapshot member mismatch')
        for source in manifest['source_overlay']:
            data = z.read(source['path'])
            if len(data) != source['bytes'] or digest(data) != source['sha256']:
                raise RuntimeError('Source snapshot content mismatch: ' + source['path'])
    for source in manifest['browseable_sources']:
        checked_file(root, source)
    print('Verified source snapshot and', len(manifest['browseable_sources']), 'browseable files')
    if target:
        target.mkdir(parents=True, exist_ok=False)
        for item in manifest['archives']:
            data = buffers[item['id']]
            (target / item['archive_name']).write_bytes(data)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for member in item['members']:
                    content = z.read(member['path'])
                    for out_root in [target / item['id'], target / 'project']:
                        destination = safe_join(out_root, member['path'])
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(content)
        for source in manifest['source_overlay']:
            checked_file(target / 'project', source)
        print('Restored archive snapshots and merged project to', target)
        print('Merge order:', ' -> '.join(manifest['merge_order']))
    print('PASS:', checked, 'scientific members;', sum(jobs.values()), 'completed experimental jobs')

if __name__ == '__main__':
    main()
