"""별도 프로세스/가상환경에서 원자료부터 재현하고 평가 단위 차이를 진단."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from data import (DataPipeline, START, LEN, WIN, HOP, FS, build, label_of,
                  make_dataset, make_windows, preprocess, split_trials, td_features)
from evaluation import evaluate_holdout
from model import cross_validate, load_dataset, make_models, train_models


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataset_digest(path):
    h = hashlib.sha256()
    with np.load(path) as arrays:
        for key in sorted(arrays.files):
            value = arrays[key]
            h.update(key.encode('utf-8'))
            h.update(str(value.dtype).encode('ascii'))
            h.update(str(value.shape).encode('ascii'))
            h.update(value.tobytes(order='C'))
    return h.hexdigest()


def check_reference(root, input_manifest, output):
    # main.py 재실행으로 outputs가 바뀌어도 제출 시점 기준값은 별도로 보존.
    path = root / 'reproduction_reference.json'
    if not path.exists():
        return {'published_reference_present': False}
    reference = json.loads(path.read_text(encoding='utf-8'))
    current_inputs = {r['file']: r['sha256'] for r in input_manifest}
    checks = {'published_inputs_match': current_inputs == reference['input_sha256'],
              'published_dataset_match': dataset_digest(output / 'dataset.npz') == reference['dataset_digest']}
    for model, expected in reference['prediction_sha256'].items():
        pred = pd.read_csv(output / f'predictions_{model}.csv').predicted
        digest = hashlib.sha256('\n'.join(pred).encode('utf-8')).hexdigest()
        checks[f'published_predictions_{model}_match'] = digest == expected
    scores = pd.read_csv(output / 'cv_scores.csv').sort_values(['fold', 'model'])
    expected = pd.DataFrame(reference['cv_scores']).sort_values(['fold', 'model'])
    checks['published_cv_scores_match'] = bool(
        scores[['fold', 'model']].reset_index(drop=True).equals(
            expected[['fold', 'model']].reset_index(drop=True)) and
        np.allclose(scores.acc, expected.acc, atol=1e-12, rtol=0))
    return checks


def diagnose_units(models, Xtr, ytr, Xte, yte, le, output):
    # 순서는 시행별 19창. 성능을 보고 채택하는 튜닝이 아닌 별도 진단.
    n_windows = (LEN - WIN) // HOP + 1
    true = yte.reshape(-1, n_windows)
    assert np.all(true == true[:, :1])
    rows, position_rows = [], []
    for name, model in models.items():
        pred = model.predict(Xte).reshape(-1, n_windows)
        # 동률이면 LabelEncoder 순서에서 첫 라벨로 결정.
        votes = np.asarray([np.bincount(p, minlength=len(le.classes_)).argmax()
                            for p in pred])
        for unit, y, p in [('window', yte, pred.ravel()),
                           ('trial_majority', true[:, 0], votes)]:
            rows.append({'model': name, 'unit': unit, 'n_test': len(y),
                         'acc': accuracy_score(y, p),
                         'macro_f1': f1_score(y, p, average='macro', zero_division=0)})
        pd.DataFrame({'true': le.inverse_transform(true[:, 0]),
                      'predicted': le.inverse_transform(votes)}).to_csv(
            output / f'trial_predictions_{name}.csv', index=False, encoding='utf-8-sig')
        for i in range(n_windows):
            position_rows.append({'model': name, 'window': i,
                                  'start_seconds_from_mark': (START + i * HOP) / FS,
                                  'acc': accuracy_score(true[:, i], pred[:, i])})
    # 19창의 특징을 평균해 시행당 25D 한 벡터로 학습하는 별도 구성.
    Xa, Xb = Xtr.reshape(-1, n_windows, 25).mean(1), Xte.reshape(-1, n_windows, 25).mean(1)
    ya = ytr.reshape(-1, n_windows)[:, 0]
    sc = StandardScaler(); Xa = sc.fit_transform(Xa); Xb = sc.transform(Xb)
    for name in ['SVM', 'RF', 'XGB']:
        model = clone(models[name]); model.fit(Xa, ya)
        pred = model.predict(Xb)
        rows.append({'model': name, 'unit': 'trial_mean_features', 'n_test': len(pred),
                     'acc': accuracy_score(true[:, 0], pred),
                     'macro_f1': f1_score(true[:, 0], pred, average='macro', zero_division=0)})
    pd.DataFrame(rows).to_csv(output / 'evaluation_unit_diagnostic.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(position_rows).to_csv(output / 'window_position_accuracy.csv', index=False, encoding='utf-8-sig')
    print('평가 단위 진단 완료', flush=True)


def diagnose_segments(pipeline, trials, labels, output):
    # 사전에 정한 시작점들. 테스트 결과로 최적 구간을 선정하거나 기본값을 바꾸지 않음.
    tr, _, te, _ = split_trials(trials, labels)
    index = {trial: i for i, trial in enumerate(trials)}
    tr_i, te_i = [index[t] for t in tr], [index[t] for t in te]
    offsets = [0, 1000, 1500, 2000, 2500]
    all_features = {start: [] for start in offsets}
    for trial in trials:
        filtered = preprocess(pipeline.load_signal(trial))
        for start in offsets:
            w = make_windows(filtered[start:start + LEN])
            all_features[start].append(np.stack([td_features(x) for x in w]))
    rows = []
    for start in offsets:
        values = np.stack(all_features[start])
        Xa = values[tr_i].reshape(-1, 25); Xb = values[te_i].reshape(-1, 25)
        ya = np.repeat([label_of(t) for t in tr], 19)
        yb = np.repeat([label_of(t) for t in te], 19)
        sc = StandardScaler(); Xa = sc.fit_transform(Xa); Xb = sc.transform(Xb)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(Xa, ya); pred = model.predict(Xb)
        rows.append({'model': 'RF', 'segment_start_ms': start, 'segment_length_ms': LEN,
                     'acc': accuracy_score(yb, pred),
                     'macro_f1': f1_score(yb, pred, average='macro', zero_division=0)})
        print(f'구간 진단: {start}ms {rows[-1]["acc"]:.6f}', flush=True)
    pd.DataFrame(rows).to_csv(output / 'segment_sensitivity.csv', index=False, encoding='utf-8-sig')


def run_audit(root, baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    pipeline = DataPipeline(root, output=output)
    files = pipeline.find_files()
    input_manifest = [{'file': Path(p).relative_to(pipeline.data_dir).as_posix(),
                       'bytes': Path(p).stat().st_size, 'sha256': sha256(p)} for p in files]
    pd.DataFrame(input_manifest).to_csv(output / 'input_sha256.csv', index=False, encoding='utf-8-sig')
    trials = pipeline.find_trials(files)
    labels = [label_of(t) for t in trials]
    make_dataset(trials, labels, pipeline.load_signal, output=output / 'dataset.npz')
    original = np.load(baseline / 'dataset.npz')
    rebuilt = np.load(output / 'dataset.npz')
    checks = {'dataset_' + k + '_exact': bool(np.array_equal(original[k], rebuilt[k]))
              for k in ['Xtr', 'ytr', 'Xte', 'yte']}
    checks['trial_manifest_exact'] = pd.read_csv(baseline / 'trial_manifest.csv').equals(
        pd.read_csv(output / 'trial_manifest.csv'))
    print('원자료 재계산 비교:', checks, flush=True)
    Xtr, ytr, Xte, yte, le = load_dataset(output / 'dataset.npz')
    models = train_models(make_models(), Xtr, ytr)
    metrics = evaluate_holdout(models, Xte, yte, output / 'holdout_metrics.csv')
    old_metrics = pd.read_csv(baseline / 'week3_holdout_metrics.csv')
    checks['holdout_metrics_match_1e-12'] = bool(np.allclose(
        old_metrics.iloc[:, 1:], metrics.iloc[:, 1:], atol=1e-12, rtol=0))
    mismatches = {}
    for name, model in models.items():
        prediction = model.predict(Xte)
        old_prediction = pd.read_csv(baseline / f'predictions_{name}.csv').predicted.to_numpy()
        mismatches[name] = int(np.count_nonzero(le.inverse_transform(prediction) != old_prediction))
        pd.DataFrame({'true': le.inverse_transform(yte),
                      'predicted': le.inverse_transform(prediction)}).to_csv(
            output / f'predictions_{name}.csv', index=False, encoding='utf-8-sig')
    checks['all_predictions_exact'] = not any(mismatches.values())
    folds = cross_validate(models, trials, labels, pipeline.load_signal, le, output / 'cv_scores.csv')
    old_cv = pd.read_csv(baseline / 'week3_cv_scores.csv')
    new_cv = pd.read_csv(output / 'cv_scores.csv')
    checks['all_25_cv_scores_match_1e-12'] = bool(np.allclose(old_cv.acc, new_cv.acc, atol=1e-12, rtol=0))
    checks['cv_split_exact'] = pd.read_csv(baseline / 'week3_cv_split.csv').equals(
        pd.read_csv(output / 'week3_cv_split.csv'))
    checks.update(check_reference(root, input_manifest, output))
    # 비교 근거를 진단 이전에 저장
    result = {'scope': 'same Windows machine and Python, fresh isolated venv; not cross-platform test',
              'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'checks': checks, 'prediction_mismatches': mismatches,
              'max_cv_absolute_delta': float(np.max(np.abs(old_cv.acc - new_cv.acc))),
              'baseline_dataset_digest': dataset_digest(baseline / 'dataset.npz'),
              'source_sha256': {p.relative_to(root).as_posix(): sha256(p)
                                for p in [root / 'main.py', root / 'requirements.txt', *sorted((root / 'code').glob('*.py'))]},
              'model_parameters': {name: m.get_params(deep=False) for name,m in models.items()}}
    (output / 'reproducibility.json').write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False),encoding='utf-8')
    print('재현 검사:', checks, flush=True)
    diagnose_units(models, Xtr, ytr, Xte, yte, le, output)
    diagnose_segments(pipeline, trials, labels, output)
    if not all(checks.values()):
        raise RuntimeError('일부 재현 검사 불일치: reproducibility.json을 확인하세요.')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='원자료 재현 및 평가 단위·구간 진단')
    parser.add_argument('--baseline-dir', type=Path, default=root / 'outputs')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs' / 'audit')
    args = parser.parse_args()
    run_audit(root, args.baseline_dir.resolve(), args.output_dir.resolve())
