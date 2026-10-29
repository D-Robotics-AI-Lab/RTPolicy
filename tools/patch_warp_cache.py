"""Enable Warp 1.6.2 JIT caches on filesystems without directory rename.

Run with the isolated simulation environment's Python and --apply after an
EXDEV error. Existing POSIX directory-rename behavior is preserved.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    import warp
    if warp.__version__ != '1.6.2':
        raise RuntimeError('Install the declared Warp 1.6.2 simulation dependency first')
    source = Path(warp.__file__).resolve().parent / 'context.py'
    if not source.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Warp must be installed in the current isolated environment')
    text = source.read_text()
    if 'RTPOLICY_EFS_WARP_CACHE' in text:
        print(json.dumps({'status':'already_patched', 'source':str(source)}))
        return
    start = text.index('                def safe_rename(')
    end = text.index('                # try to move process outputs to cache', start)
    block = text[start:end]
    original = '                            os.rename(src, dst)\n                            return\n'
    replacement = '''                            # RTPOLICY_EFS_WARP_CACHE: publish complete files atomically.
                            try:
                                os.rename(src, dst)
                            except OSError as cache_error:
                                if cache_error.errno != errno.EXDEV:
                                    raise
                                import shutil
                                os.makedirs(dst, exist_ok=True)
                                # A binary is the cache-readiness marker: commit metadata first.
                                names = sorted(os.listdir(src), key=lambda name: name.endswith((".ptx", ".cubin", ".o")))
                                for name in names:
                                    source_file = os.path.join(src, name)
                                    if not os.path.isfile(source_file):
                                        raise cache_error
                                    target_file = os.path.join(dst, name)
                                    pending_file = target_file + f".{os.getpid()}.tmp"
                                    shutil.copyfile(source_file, pending_file)
                                    os.replace(pending_file, target_file)
                                shutil.rmtree(src)
                            return
'''
    if block.count(original) != 1:
        raise RuntimeError('Unexpected Warp cache implementation; no files changed')
    patched = text[:start] + block.replace(original, replacement, 1) + text[end:]
    ast.parse(patched)
    backup = source.with_name('context.py.rtpolicy-before-efs-cache')
    if args.apply:
        if not backup.exists():
            shutil.copy2(source, backup)
        source.write_text(patched)
    print(json.dumps({'status':'patched' if args.apply else 'patch_available',
                      'source':str(source), 'backup':str(backup) if args.apply else None,
                      'before_sha256':hashlib.sha256(text.encode()).hexdigest(),
                      'after_sha256':hashlib.sha256(patched.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
