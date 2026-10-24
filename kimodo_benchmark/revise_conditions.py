"""Create a new anchored-condition protocol without changing the task panel."""

import argparse
import copy
from pathlib import Path
import shutil
import tempfile

from .protocol import ANCHORED_ADAPTER, identity, load_protocol, validate_protocol, write_json


def revise_conditions(source, output):
    original = load_protocol(source)
    if original['schema_version'] != 2:
        raise ValueError('Revision requires the original current-main v2 protocol')
    if output.exists():
        raise FileExistsError('Use a new output directory')
    revised = copy.deepcopy(original)
    revised.pop('protocol_id')
    revised['schema_version'] = 3
    revised['adapter'] = copy.deepcopy(ANCHORED_ADAPTER)
    revised['condition_revision'] = dict(parent_protocol_id=original['protocol_id'],
        change='anchor relative quantities at the start; align wrist/ankle targets with scored joints; squat five-frame hold')
    revised['protocol_id'] = identity(revised)
    validate_protocol(revised)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.revise-', dir=output.parent))
    try:
        shutil.copytree(source.parent/'scoring_source', temporary/'scoring_source', ignore=shutil.ignore_patterns('__pycache__'))
        write_json(temporary/'parent_protocol.json', original)
        write_json(temporary/'protocol.json', revised)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = revise_conditions(args.protocol, args.output)
    print(protocol['protocol_id'])


if __name__ == '__main__':
    main()
