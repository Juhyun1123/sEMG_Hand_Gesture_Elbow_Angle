"""3주차 실습 ① 모델 학습, ④ 교차검증, ⑥ 계산 비용."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from data import build, trial_id


def load_dataset(path):
    # 강의 9페이지: dataset.npz에서 2주차 결과 로드
    d = np.load(path, allow_pickle=True)
    Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
    le = LabelEncoder(); ytr = le.fit_transform(ytr)
    yte = le.transform(yte)
    return Xtr, ytr, Xte, yte, le


def make_models():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError('xgboost가 필요합니다: pip install -r requirements.txt') from exc

    # 강의 9페이지의 모델 설정을 그대로 사용
    models = {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale',
                   decision_function_shape='ovr'),
        'RF': RandomForestClassifier(n_estimators=100,
                   random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(n_estimators=400, max_depth=6,
                   learning_rate=0.05, subsample=0.9,
                   colsample_bytree=0.8, objective='multi:softprob',
                   reg_lambda=1.0, tree_method='hist', random_state=42),
        # 추가 비교: 동일한 25D 표준화 특징, 테스트 결과를 보고 튜닝하지 않음
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(solver='svd'),
    }
    return models


def train_models(models, Xtr, ytr):
    # 강의 9페이지 실습 단계 1
    for k, m in models.items():
        m.fit(Xtr, ytr); print(k, '학습 완료')
    return models


def cross_validate(models, files, labels, load_signal, le, output):
    # 강의 12페이지 실습 단계 4: 시행 단위 층화 5-fold
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_acc = {k: [] for k in models}
    rows, split_rows = [], []
    for fold, (tr_i, va_i) in enumerate(skf.split(files, labels), start=1):
        assert not set(tr_i) & set(va_i)
        for split, indices in [('train', tr_i), ('validation', va_i)]:
            split_rows.extend({'fold': fold, 'split': split,
                               'trial': trial_id(files[i]), 'label': labels[i]}
                              for i in indices)
        Xa, ya = build([files[i] for i in tr_i], [labels[i] for i in tr_i], load_signal)
        Xb, yb = build([files[i] for i in va_i], [labels[i] for i in va_i], load_signal)
        sc = StandardScaler(); Xa = sc.fit_transform(Xa)
        Xb = sc.transform(Xb); ya, yb = le.transform(ya), le.transform(yb)
        for k, proto in models.items():
            m = clone(proto); m.fit(Xa, ya)
            acc = m.score(Xb, yb)
            fold_acc[k].append(acc)
            rows.append({'fold': fold, 'model': k, 'acc': acc})
        print(f'5-fold: {fold}/5 완료')
    for k, v in fold_acc.items():
        print(f'{k}: {np.mean(v)*100:.2f} ± {np.std(v)*100:.2f} %')

    pd.DataFrame(rows).to_csv(output, index=False, encoding='utf-8-sig')
    pd.DataFrame(split_rows).to_csv(output.with_name('week3_cv_split.csv'),
                                    index=False, encoding='utf-8-sig')
    return fold_acc


def measure_compute_cost(models, Xtr, ytr, Xte, output):
    # 강의 14페이지 실습 단계 6: 학습 및 1개 샘플 추론 100회
    rows = []
    for k, m in models.items():
        t0 = time.perf_counter(); m.fit(Xtr, ytr)
        train_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(100): m.predict(Xte[:1])
        infer_ms = (time.perf_counter() - t0) * 10

        print(f'{k}: 학습 {train_t:.3f}s, '
              f'추론 {infer_ms:.3f} ms/sample')
        import pickle
        size_mb = len(pickle.dumps(m, protocol=5)) / 1024**2
        rows.append({'model': k, 'train_s': train_t, 'infer_ms_per_sample': infer_ms,
                     'model_mb': size_mb})
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return df
