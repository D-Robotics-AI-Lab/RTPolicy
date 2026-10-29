"""Convert an original RoboTwin evaluation log into a JSON summary."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def summarize(path):
    text = path.read_text()
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    total = re.search(r'Success Rate:\s*([0-9.]+)\s*\((\d+)/(\d+)\)', text)
    if total is None:
        raise ValueError('The original log has no completed success-rate summary')
    rate, successes, episodes = float(total[1]), int(total[2]), int(total[3])
    records = []
    for line in text.splitlines():
        match = re.search(r'【(\d+)】.*?seed:\s*(\d+).*?result:\s*(Success|Fail)', line)
        if match is None:
            continue
        record = {'episode':int(match[1]), 'seed':int(match[2]), 'success':match[3]=='Success'}
        for name in ('action_cnt','limit','infer_cnt'):
            value = re.search(rf'\b{name}:\s*(\d+)', line)
            if value:
                record[name] = int(value[1])
        records.append(record)
    records.sort(key=lambda value:value['episode'])
    if len(records) != episodes or [r['episode'] for r in records] != list(range(episodes)):
        raise ValueError('The original log does not contain every completed episode')
    if sum(r['success'] for r in records) != successes:
        raise ValueError('Per-episode outcomes do not match the original summary')
    if len({r['seed'] for r in records}) != episodes:
        raise ValueError('The original log contains duplicate episode seeds')
    return {'source_log':str(path.resolve()), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'successes':successes, 'episodes':episodes, 'success_rate':rate,
            'seeds':[r['seed'] for r in records], 'episode_results':records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = summarize(args.log)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Original evaluator: {result['successes']}/{result['episodes']}")


if __name__ == '__main__':
    main()
