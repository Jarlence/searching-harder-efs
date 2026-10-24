"""Verify every tracked delivery file against SHA256SUMS.txt; standard library only."""
import hashlib
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parent
    checked = 0
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        parts = PurePosixPath(relative)
        if parts.is_absolute() or '..' in parts.parts or '\\' in relative:
            raise RuntimeError('Unsafe manifest path: ' + relative)
        path = root.joinpath(*parts.parts)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('SHA-256 mismatch: ' + relative)
        checked += 1
    print('PASS:', checked, 'repository files match SHA256SUMS.txt')


if __name__ == '__main__':
    main()
