"""3주차 실습 ② 성능 평가, ③ 혼동행렬, ⑤ 대응표본 t-검정, 보고서."""

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


def write_week3_report(metrics, fold_acc, tests, costs, cm, class_names, output):
    lines = [
        '[3주차 모델 비교 실습]',
        '', '[1] Hold-out 성능 (macro 정밀도/재현율/F1)',
        metrics.round(4).to_string(index=False),
        '', '[2] 교차검증 정확도 (5-fold, 마커 시행 단위)',
    ]
    for model, scores in fold_acc.items():
        line = ', '.join(f'{value * 100:.2f}%' for value in scores)
        lines.append(f'{model}: 폴드 [{line}], 평균 {np.mean(scores)*100:.2f}%, '
                     f'표준편차 {np.std(scores)*100:.2f}%')
    lines += ['', '[3] 통계 검정 (같은 fold 간 대응표본 t-검정, α=0.05)']
    for row in tests.itertuples():
        result = ('검정 불가' if not np.isfinite(row.p)
                  else f'p={row.p:.4f}: ' +
                  ('유의한 차이가 검출됨' if row.p < 0.05 else '유의한 차이가 검출되지 않음'))
        lines.append(f'{row.model_a} vs {row.model_b}: {result}')
    lines += [
        'p ≥ 0.05는 모델이 동등하다는 증거가 아닙니다.',
        f'비교가 {len(tests)}쌍이며 폴드가 5개뿐이므로 검정력과 다중비교를 고려해야 합니다.',
        '', '[4] 계산 비용 (현재 실행 환경에서 측정)',
        costs.round(4).to_string(index=False),
        '', '[5] RF 혼동행렬의 주요 오류',
    ]
    mistakes = sorted(
        ((int(cm[i, j]), str(class_names[i]), str(class_names[j]))
         for i in range(len(class_names)) for j in range(len(class_names))
         if i != j and cm[i, j] > 0), reverse=True)
    if not mistakes:
        lines.append('관찰된 오분류가 없습니다.')
    for count, true, predicted in mistakes[:5]:
        kind = ('같은 동작·다른 각도' if true.split('_', 1)[-1] ==
                predicted.split('_', 1)[-1] else '서로 다른 동작')
        lines.append(f'정답 {true} → 예측 {predicted}: {count}개 ({kind})')
    lines += [
        '', '[6] 해석의 한계]',
        '1. CSV mark 기준 trial-wise 분할입니다. 새로운 피험자에 대한 일반화 성능(LOSO)을 검증하지 않았습니다.',
        '2. 같은 기록 파일 안의 다른 시행이 양쪽 분할에 포함될 수 있습니다.',
        '3. 겹치는 윈도우의 독립성은 제한적이며, 같은 시행의 윈도우가 한 분할 안에 모이게 했습니다.',
        '4. 5개 교차검증 폴드의 점수는 학습 집합이 겹치므로 대응표본 t-검정의 독립성 가정에 주의해야 합니다.',
        '5. SVM/RF와 XGBoost의 하이퍼파라미터 탐색 정도가 동일하지 않습니다.',
        '6. 1~2주차 전처리 정의는 제공된 PPT의 실습 코드를 따르며, 하이퍼파라미터는 실습 설정입니다.',
        '', '[7] 출력 파일]',
        'week3_holdout_metrics.csv / confusion8.png / week3_cv_scores.csv',
        'week3_ttests.csv / week3_compute_times.csv / week3_summary.txt',
    ]
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    return output
