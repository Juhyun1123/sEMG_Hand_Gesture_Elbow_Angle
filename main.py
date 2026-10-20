"""프로젝트 위치나 현재 작업 폴더에 관계없이 실험 실행."""

import runpy
import sys
from pathlib import Path


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent / 'code'
    sys.path.insert(0, str(code_dir))
    runpy.run_path(str(code_dir / 'main.py'), run_name='__main__')
