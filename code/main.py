"""Step 2 전체 실험 실행: 1~2주차 데이터셋 생성 후 3주차 모델 비교."""

import argparse
import importlib
import json
import platform
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from data import DataPipeline, HOP, LEN, WIN, label_of
from evaluation import (evaluate_holdout, evaluate_trial_vote,
                        paired_tests, plot_confusion)
from model import (cross_validate, load_dataset, make_models,
                   measure_compute_cost, train_models)


class Main:
    def __init__(self, data_dir=None, output=None, trial_vote=False):
        self.root = Path(__file__).resolve().parents[1]
        self.data = DataPipeline(self.root, output=output, data_dir=data_dir)
        self.output = self.data.output
        self.trial_vote = trial_vote

    def environment(self):
        versions = {}
        for name in ['numpy', 'scipy', 'sklearn', 'xgboost', 'pandas', 'matplotlib']:
            try:
                module = importlib.import_module(name)
                print(name.ljust(10), module.__version__)
                versions[name] = module.__version__
            except ImportError:
                raise ImportError(f'{name} 미설치 (pip install -r requirements.txt)')

        metadata = {'python': sys.version, 'platform': platform.platform(),
                    'processor': platform.processor(), 'packages': versions,
                    'run_time': datetime.now(timezone(timedelta(hours=9))).isoformat()}
        (self.output / 'environment.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    def run(self):
        self.environment()
        files = self.data.find_files()
        self.data.run_week1(files)
        trials = self.data.find_trials(files)
        self.data.run_week2(trials)

        print('\n[3주차: 기존 3개 + 추가 2개 모델 학습]')
        Xtr, ytr, Xte, yte, le = load_dataset(self.output / 'dataset.npz')
        models = train_models(make_models(), Xtr, ytr)

        print('\n[3주차: Hold-out 성능]')
        evaluate_holdout(models, Xte, yte,
                                   self.output / 'week3_holdout_metrics.csv')
        plot_confusion(models, Xte, yte, le, self.output / 'confusion8.png')
        if self.trial_vote:
            print('\n[추가 분석: 시행별 최빈 라벨 투표]')
            evaluate_trial_vote(self.output, (LEN - WIN) // HOP + 1)

        print('\n[3주차: 5-fold 교차검증]')
        labels = [label_of(f) for f in trials]
        fold_acc = cross_validate(models, trials, labels, self.data.load_signal, le,
                                  self.output / 'week3_cv_scores.csv')

        print('\n[3주차: 대응표본 t-검정]')
        paired_tests(fold_acc, self.output / 'week3_ttests.csv')

        print('\n[3주차: 계산 비용]')
        measure_compute_cost(models, Xtr, ytr, Xte,
                                     self.output / 'week3_compute_times.csv')



if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='sEMG 1~3주차 재현 및 5모델 비교')
    parser.add_argument('--data-dir', type=Path, help='CSV 폴더 (기본: 프로젝트/data)')
    parser.add_argument('--output-dir', type=Path, help='결과 폴더 (기본: 프로젝트/outputs)')
    parser.add_argument('--trial-vote', action='store_true',
                        help='창별 예측을 시행별 최빈 라벨로 집계해 성능 추가 출력')
    args = parser.parse_args()
    Main(args.data_dir, args.output_dir, args.trial_vote).run()
