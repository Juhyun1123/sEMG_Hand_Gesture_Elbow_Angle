"""Step 2: 1주차 데이터 확인과 2주차 PPT의 전처리·특징 추출."""

import csv
import glob
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 2주차 PPT 실습 순서와 특징 정의를 유지합니다.
FS = 1000
START = 1500  # 강의 실습 가정: 실제 동작 구간에 맞춰 조정 가능
LEN = 2000
WIN = 200
HOP = 100
FEATURE_NAMES = ('RMS', 'MAV', 'WL', 'ZC', 'SSC')
CHANNEL_NAMES = ['ECR', 'ED', 'FDS', 'Biceps', 'Triceps']


@dataclass(frozen=True)
class Trial:
    path: str
    index: int
    start: int
    stop: int

    @property
    def name(self):
        return f'{Path(self.path).stem}_trial{self.index:02d}'


def trial_id(path):
    return path.name if isinstance(path, Trial) else Path(path).name


def label_of(path):
    if isinstance(path, Trial):
        path = path.path
    # 예: a_30 fist.csv -> 30_fist
    return Path(path).stem.split('_', 1)[1].replace(' ', '_')


def preprocess(x, fs=FS):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError('입력은 (샘플 수, 5채널) 배열이어야 합니다.')
    if not np.isfinite(x).all():
        raise ValueError('신호에 NaN 또는 무한대가 포함되어 있습니다.')
    if fs <= 900:
        raise ValueError('20~450Hz 필터에는 샘플링 주파수 fs > 900Hz가 필요합니다.')
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20/(fs/2), 450/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)


def cut(x, start=START, n=LEN):
    if start < 0 or n <= 0 or start + n > len(x):
        raise ValueError(f'요청한 구간 [{start}:{start+n}]이 길이 {len(x)}을 벗어납니다.')
    return np.asarray(x)[start:start+n]


def make_windows(x, win=WIN, hop=HOP):
    x = np.asarray(x)
    if x.ndim != 2 or win <= 0 or hop <= 0 or len(x) < win:
        raise ValueError('윈도우 길이·간격 및 입력 배열을 확인해 주세요.')
    return np.stack([x[i:i+win] for i in range(0, len(x)-win+1, hop)])


def td_features(w, th=0.01):
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or len(w) < 3:
        raise ValueError('윈도우는 최소 3샘플의 2차원 배열이어야 합니다.')
    rms = np.sqrt(np.mean(w**2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1]*w[1:]) < 0) & (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1]*d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])


def extract_trial_features(x, fs=FS, start=START, n=LEN, win=WIN, hop=HOP, th=0.01):
    """5채널 시행 -> (19, 25) 특징 행렬 (기본 설정)."""
    windows = make_windows(cut(preprocess(x, fs), start, n), win, hop)
    return np.stack([td_features(w, th) for w in windows])


def split_trials(files, labels, test_size=0.2, random_state=42):
    """윈도우 추출 이전에 파일 단위 분할: 겹치는 윈도우 간 누수 방지."""
    if not files or len(files) != len(labels):
        raise ValueError('files/labels 길이가 일치하지 않거나 비어 있습니다.')
    train_idx, test_idx = train_test_split(
        np.arange(len(files)), test_size=test_size,
        stratify=labels, random_state=random_state)
    return ([files[i] for i in train_idx], [labels[i] for i in train_idx],
            [files[i] for i in test_idx], [labels[i] for i in test_idx])


@lru_cache(maxsize=4096)
def cached_features(path, load_signal, **feature_kwargs):
    # 특징은 시행 내부에서만 계산하므로 재사용 가능. scaler는 각 학습 분할에서 fit.
    return extract_trial_features(load_signal(path), **feature_kwargs)


def build(files, labels, load_signal, **feature_kwargs):
    """시행별 특징을 연결. load_signal은 신호 5열의 ndarray를 반환."""
    X, y = [], []
    for path, label in zip(files, labels):
        f = cached_features(path, load_signal, **feature_kwargs)
        X.append(f)
        y.extend([label] * len(f))
    if not X:
        raise ValueError('특징을 생성할 파일이 없습니다.')
    return np.concatenate(X, axis=0), np.asarray(y)


def standardize(Xtr, Xte):
    scaler = StandardScaler()
    return scaler.fit_transform(Xtr), scaler.transform(Xte), scaler


def save_dataset(path, Xtr, ytr, Xte, yte):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return path


def make_dataset(files, labels, load_signal, output='dataset.npz',
                 test_size=0.2, random_state=42, **feature_kwargs):
    tr_f, tr_l, te_f, te_l = split_trials(files, labels, test_size, random_state)
    Xtr, ytr = build(tr_f, tr_l, load_signal, **feature_kwargs)
    Xte, yte = build(te_f, te_l, load_signal, **feature_kwargs)
    Xtr, Xte, scaler = standardize(Xtr, Xte)
    save_dataset(output, Xtr, ytr, Xte, yte)
    return Xtr, ytr, Xte, yte, scaler


class DataPipeline:
    def __init__(self, root=None, output=None, data_dir=None):
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self.data_dir = Path(data_dir or self.root / 'data').resolve()
        self.output = Path(output or self.root / 'outputs').resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=64)
    def load(self, path):
        # 원본 8열: sEMG 5채널 + time, onset, mark
        raw = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        if raw.shape[1] < 8:
            raise ValueError(f'8열 CSV를 기대했지만 {raw.shape[1]}열입니다: {path}')
        if not np.isfinite(raw).all():
            raise ValueError(f'NaN/무한값: {path}')
        if not np.allclose(np.diff(raw[:, 5]), 1 / FS, atol=1e-6):
            raise ValueError(f'1kHz 시간 간격을 확인하세요: {path}')
        x = raw[:, :5]
        t = raw[:, 5] - raw[0, 5]
        return raw, x, t

    def load_signal(self, path):
        """2주차 전처리에 전달할 5채널 신호만 반환."""
        if isinstance(path, Trial):
            _, x, _ = self.load(path.path)
            return x[path.start:path.stop]
        _, x, _ = self.load(path)
        return x

    def plot_example(self, path):
        _, x, t = self.load(path)
        raw, _, _ = self.load(path)
        marks = np.flatnonzero(raw[:, 7] != 0)
        if len(marks):
            start = int(marks[0]); x = x[start:start + 5000]
            t = np.arange(len(x)) / FS
        fig, ax = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
        for ch in range(5):
            ax[ch].plot(t, x[:, ch], lw=0.5)
            ax[ch].set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
        if len(x) == 5000 and np.isclose(t[-1], 4.999, atol=0.01):
            for axis in ax:
                axis.axvline(2.0, color='r', ls='--')
                axis.axvline(3.0, color='r', ls='--')
        ax[4].set_xlabel('time (s)')
        fig.suptitle(os.path.basename(path))
        plt.tight_layout()
        fig.savefig(self.output / 'ch5_example.png', dpi=120)
        plt.close(fig)

    def plot_comparison(self, files):
        observations = []
        subject = Path(files[0]).stem.split('_', 1)[0]
        for gesture in ['extor', 'fist', 'open', 'pro']:
            pair = [self.data_dir / f'{subject}_{angle} {gesture}.csv'
                    for angle in [30, 60]]
            if not all(path.exists() for path in pair):
                continue
            rms_by_angle = []
            fig, ax = plt.subplots(5, 2, figsize=(14, 7), sharex='col', sharey='row')
            for col, path in enumerate(pair):
                raw, x, t = self.load(path)
                start = int(np.flatnonzero(raw[:, 7] != 0)[0])
                x = x[start:start + 5000]; t = np.arange(len(x)) / FS
                rms_by_angle.append(np.sqrt(np.mean(cut(x)**2, axis=0)))
                for ch in range(5):
                    ax[ch, col].plot(t, x[:, ch], lw=0.5)
                    ax[ch, col].set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
                ax[0, col].set_title(path.stem)
                ax[4, col].set_xlabel('time (s)')
            plt.tight_layout()
            fig.savefig(self.output / f'compare_{gesture}_30_60.png', dpi=120)
            plt.close(fig)
            for ch, name in enumerate(CHANNEL_NAMES):
                a, b = rms_by_angle[0][ch], rms_by_angle[1][ch]
                observations.append([subject, gesture, name, a, b, b / a if a else np.nan])
        with (self.output / 'week1_angle_observations.csv').open(
                'w', newline='', encoding='utf-8-sig') as fp:
            writer = csv.writer(fp)
            writer.writerow(['subject', 'gesture', 'channel', 'rms30', 'rms60', 'ratio60_30'])
            writer.writerows(observations)

    def find_files(self):
        files = sorted(glob.glob(str(self.data_dir / '**' / '*.csv'), recursive=True))
        if not files:
            raise FileNotFoundError('data/ 폴더에 CSV 데이터를 넣어 주세요.')
        return files

    def find_trials(self, files):
        # 제공 CSV는 여러 시행을 포함. mark를 시행 시작으로 해석하는 재현 가정.
        trials, rows = [], []
        for path in files:
            raw, _, _ = self.load(path)
            starts = np.flatnonzero(raw[:, 7] != 0)
            if not len(starts):
                raise ValueError(f'mark가 없습니다: {path}')
            for index, start in enumerate(starts, start=1):
                stop = min(int(start) + 5 * FS,
                           int(starts[index]) if index < len(starts) else len(raw))
                if stop - start < START + LEN:
                    raise ValueError(f'분석 구간보다 짧은 시행: {path}, {index}')
                trial = Trial(str(path), index, int(start), stop)
                trials.append(trial)
                rows.append([trial.name, Path(path).name, Path(path).stem.split('_')[0],
                             label_of(path), index, int(start), stop,
                             int(start) + START, int(start) + START + LEN])
        with (self.output / 'trial_manifest.csv').open('w', newline='', encoding='utf-8-sig') as fp:
            writer = csv.writer(fp)
            writer.writerow(['trial', 'file', 'subject', 'label', 'trial_index',
                             'start_sample', 'stop_sample', 'segment_start', 'segment_stop'])
            writer.writerows(rows)
        print('원본 기록:', len(files), '/ 마커 기준 시행:', len(trials))
        print('시행 클래스 분포:', Counter(label_of(t) for t in trials))
        return trials

    def run_week1(self, files):
        print('파일 수:', len(files))
        print('예시:', files[0])
        labels = [label_of(f) for f in files]
        print('클래스 분포:', Counter(labels))
        _, x, _ = self.load(files[0])
        print('배열 모양:', x.shape)
        print('채널 수:', x.shape[1])
        rows = ['file,label,samples,channels,duration_seconds,mark_count']
        for path in files:
            raw, signal, t = self.load(path)
            rows.append(f'{Path(path).name},{label_of(path)},{len(signal)},'
                        f'{signal.shape[1]},{t[-1]:.3f},{np.count_nonzero(raw[:, 7])}')
        (self.output / 'data_summary.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8-sig')
        example = self.data_dir / 'a_30 fist.csv'
        self.plot_example(example if example.exists() else files[0])
        self.plot_comparison(files)

    def plot_filter_spectrum(self, before, after, filename='week2_filter_spectrum.png'):
        """제출 ①: 같은 세그먼트의 필터 전/후 PSD (5채널)."""
        before_seg = cut(before)
        after_seg = cut(after)
        freq, original = welch(before_seg, fs=FS, nperseg=1000, axis=0)
        _, filtered = welch(after_seg, fs=FS, nperseg=1000, axis=0)
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(freq, 10 * np.log10(np.maximum(original[:, i], 1e-20)),
                    label='Before', lw=0.9)
            ax.plot(freq, 10 * np.log10(np.maximum(filtered[:, i], 1e-20)),
                    label='After', lw=0.9)
            ax.axvline(60, color='gray', ls='--', lw=0.7, alpha=0.7)
            ax.set_ylabel(f'{CHANNEL_NAMES[i]}\nPSD (dB/Hz)', fontsize=8)
            ax.grid(alpha=0.2)
        axes[0].legend()
        axes[-1].set_xlabel('Frequency (Hz)')
        axes[-1].set_xlim(0, FS / 2)
        fig.suptitle('sEMG Spectrum: Before / After Filtering (Welch PSD)')
        fig.tight_layout()
        path = self.output / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
        t = (START + np.arange(LEN)) / FS
        for i, ax in enumerate(axes):
            ax.plot(t, before_seg[:, i], label='Before', lw=0.6, alpha=0.65)
            ax.plot(t, after_seg[:, i], label='After', lw=0.6)
            ax.set_ylabel(CHANNEL_NAMES[i]); ax.grid(alpha=0.2)
        axes[0].legend(); axes[-1].set_xlabel('Time from mark (s)')
        fig.suptitle('Selected 1.5-3.5 s segment: Before / After Filtering')
        fig.tight_layout()
        fig.savefig(self.output / 'week2_filter_waveform.png', dpi=150)
        plt.close(fig)
        return path

    def save_split_tables(self, files, labels, ytr, yte):
        """시행 분할 명세와 클래스 분포를 CSV로 저장."""
        train_files, train_labels, test_files, test_labels = split_trials(
            files, labels, test_size=0.2, random_state=42)

        if set(map(trial_id, train_files)) & set(map(trial_id, test_files)):
            raise AssertionError('같은 시행 ID이 학습/테스트에 중복되었습니다.')
        split_path = self.output / 'week2_trial_split.csv'
        with split_path.open('w', newline='', encoding='utf-8-sig') as fp:
            writer = csv.writer(fp)
            writer.writerow(['split', 'file', 'label'])
            for split, split_files, split_labels in (
                ('train', train_files, train_labels),
                ('test', test_files, test_labels)):
                for path, label in zip(split_files, split_labels):
                    writer.writerow([split, trial_id(path), label])

        counts_train = Counter(ytr)
        counts_test = Counter(yte)
        trials_train = Counter(train_labels)
        trials_test = Counter(test_labels)
        dist_path = self.output / 'week2_class_distribution.csv'
        with dist_path.open('w', newline='', encoding='utf-8-sig') as fp:
            writer = csv.writer(fp)
            writer.writerow(['class', 'train_trials', 'test_trials',
                             'train_windows', 'test_windows'])
            for label in sorted(set(labels)):
                writer.writerow([label, trials_train[label], trials_test[label],
                                 counts_train[label], counts_test[label]])

        return split_path, dist_path

    def run_week2(self, files):
        labels = [label_of(f) for f in files]
        x = self.load_signal(files[0])
        xf = preprocess(x)
        seg = cut(xf)
        windows = make_windows(seg)
        first_features = td_features(windows[0])
        print('\n[2주차 중간 결과]')
        print('필터 전 표준편차:', x.std(axis=0).round(4))
        print('필터 후 표준편차:', xf.std(axis=0).round(4))
        print('세그먼트:', seg.shape)
        print('윈도우:', windows.shape)
        print('특징:', first_features.shape)
        spectrum_path = self.plot_filter_spectrum(x, xf)
        print('필터 스펙트럼:', spectrum_path)

        Xtr, ytr, Xte, yte, scaler = make_dataset(
            files, labels, load_signal=self.load_signal,
            output=self.output / 'dataset.npz', test_size=0.2, random_state=42)
        print('\n[2주차 데이터셋]')
        print('학습:', Xtr.shape, Counter(ytr))
        print('테스트:', Xte.shape, Counter(yte))
        print('학습 평균:', Xtr.mean(axis=0).round(6))
        print('학습 표준편차:', Xtr.std(axis=0).round(6))
        print('저장:', self.output / 'dataset.npz')
        split_path, dist_path = self.save_split_tables(files, labels, ytr, yte)
        print('분할 명세:', split_path)
        print('클래스 분포:', dist_path)
        return Xtr, ytr, Xte, yte, scaler
