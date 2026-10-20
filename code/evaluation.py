"""3주차 실습 ② 성능 평가, ③ 혼동행렬, ⑤ 대응표본 t-검정."""

from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def evaluate_holdout(models, Xte, yte, output):
    # 강의 10페이지 실습 단계 2
    rows = []
    for k, m in models.items():
        p = m.predict(Xte)
        rows.append([k,
            accuracy_score(yte, p),
            precision_score(yte, p, average='macro', zero_division=0),
            recall_score(yte, p, average='macro', zero_division=0),
            f1_score(yte, p, average='macro', zero_division=0)])

    df = pd.DataFrame(rows, columns=
        ['model', 'acc', 'precision', 'recall', 'f1'])
    print(df.round(4))
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return df


def plot_confusion(models, Xte, yte, le, output):
    # 강의 11페이지의 RF 그림 형식을 모든 모델에 적용
    matrices = {}
    L = le.classes_
    for k, m in models.items():
        p = m.predict(Xte)
        cm = confusion_matrix(yte, p, labels=range(len(L)))
        matrices[k] = cm
        pd.DataFrame(cm, index=L, columns=L).to_csv(
            output.with_name(f'confusion_{k}.csv'), encoding='utf-8-sig')
        pd.DataFrame({'true': le.inverse_transform(yte),
                      'predicted': le.inverse_transform(p)}).to_csv(
            output.with_name(f'predictions_{k}.csv'), index=False, encoding='utf-8-sig')
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(L)), L, rotation=45, ha='right')
        ax.set_yticks(range(len(L)), L)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(k)
        for i in range(len(L)):
            for j in range(len(L)):
                if cm[i, j]:
                    color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color=color)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(output.with_name(f'confusion_{k}.png'), dpi=150)
        if k == 'RF':
            fig.savefig(output, dpi=150)  # 기존 강의 제출 파일명 유지
        plt.close(fig)
    return matrices


def paired_tests(fold_acc, output):
    # 강의 13페이지 실습 단계 5
    rows = []
    for a, b in combinations(fold_acc.keys(), 2):
        t, p = ttest_rel(fold_acc[a], fold_acc[b])
        if not np.isfinite(p):
            print(a, b, '검정 불가')
            rows.append({'model_a': a, 'model_b': b, 't': np.nan, 'p': np.nan,
                         'verdict': '검정 불가'})
            continue
        verdict = '유의함' if p < 0.05 else '유의하지 않음'
        print(f'{a} vs {b}: p = {p:.4f}  ({verdict})')
        rows.append({'model_a': a, 'model_b': b, 't': t, 'p': p,
                     'verdict': verdict})
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return df
