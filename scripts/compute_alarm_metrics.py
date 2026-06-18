"""
Real shot-level alarm metrics using t_cut disruption timestamps.

All 50 test shots disrupted (finite t_cut). We compute:
    - TPR: fraction of shots where model raises alarm before t_cut
    - Mean Warning Time: mean(t_cut - t_alarm) across shots where alarm fired
    - FAR: not computable — no non-disruptive shots in test set

Alarm definition: predicted plasma current drops below threshold fraction
of the shot's mean predicted current, sustained for N consecutive windows.

Threshold is swept and best result reported.
"""

import sys, os, json, pickle
import numpy as np
import torch

sys.path.insert(0, '/workspace/tokamark/src')
sys.path.insert(0, '/workspace/fusion_research/scripts')

from train_lstm import PlasmaLSTM, nrmse
from train_transformer import PlasmaTransformer
from train_cnn_baseline import PlasmaCNN
import pickle as pkl

DATA_DIR   = '/workspace/fusion_research/data'
CKPT_DIR   = '/workspace/fusion_research/checkpoints'
RESULTS_DIR = '/workspace/fusion_research/results'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(name):
    ckpt_path = os.path.join(CKPT_DIR, f'{name}_clean.pt')
    if not os.path.exists(ckpt_path):
        print(f'  WARNING: {ckpt_path} not found')
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if name == 'lstm':
        model = PlasmaLSTM(input_size=ckpt['n_features']).to(DEVICE)
    elif name == 'transformer':
        model = PlasmaTransformer(input_size=ckpt['n_features']).to(DEVICE)
    elif name == 'cnn':
        model = PlasmaCNN(
            n_channels=ckpt['n_channels'],
            input_len=ckpt['input_len'],
            backbone_hidden=ckpt.get('backbone_hidden', 64)
        ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'  Loaded {name}')
    return model


def predict_batch(model, X):
    """Run inference on (N, T, F) array, return (N,) predictions."""
    preds = []
    BATCH = 128
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            b = torch.tensor(
                X[i:i+BATCH], dtype=torch.float32).to(DEVICE)
            preds.extend(model(b).cpu().numpy())
    return np.array(preds)


def compute_metrics_for_model(model_name, model, samples,
                               X_test_ts, y_test):
    """
    Group windows by shot, run predictions, sweep alarm threshold,
    compute TPR and mean warning time.
    """
    # Group by shot
    shot_data = {}
    for i, s in enumerate(samples):
        sid = s['shot_id']
        if sid not in shot_data:
            shot_data[sid] = {
                'windows':      [],
                'window_times': [],  # end time of each window
                't_cut':        s['t_cut'],
                'X_indices':    [],
            }
        # Get end time of this window from first signal's time array
        first_sig = list(s['input'].keys())[0]
        t_arr = s['input'][first_sig]['time']
        window_end = float(t_arr[-1])

        shot_data[sid]['window_times'].append(window_end)
        shot_data[sid]['X_indices'].append(i)

    shot_ids = list(shot_data.keys())
    print(f'  Shots: {len(shot_ids)}, all disruptive (finite t_cut)')

    # Run predictions per shot
    for sid in shot_ids:
        d = shot_data[sid]
        idx = d['X_indices']
        X   = X_test_ts[idx]
        if model_name == 'xgboost':
            X_feat = np.load(f'{DATA_DIR}/test_X_feat.npy')[idx]
            d['preds'] = model.predict(X_feat)
        else:
            d['preds'] = predict_batch(model, X)

        # Sort by window time
        order = np.argsort(d['window_times'])
        d['preds']        = np.array(d['preds'])[order]
        d['window_times'] = np.array(d['window_times'])[order]

    # Sweep alarm threshold
    best = {'tpr': 0, 'mwt': np.nan, 'threshold': 0}
    thresholds = np.linspace(0.05, 0.60, 30)

    for thresh in thresholds:
        tprs, warning_times = [], []

        for sid in shot_ids:
            d      = shot_data[sid]
            preds  = d['preds']
            times  = d['window_times']
            t_cut  = d['t_cut']

            shot_mean = np.abs(preds).mean()
            if shot_mean < 1e-6:
                continue

            # Alarm fires when prediction drops thresh fraction below mean
            alarm_mask = preds < (1 - thresh) * shot_mean

            # Find first alarm before t_cut
            pre_disruption = times < t_cut
            alarm_before   = alarm_mask & pre_disruption

            if alarm_before.any():
                first_alarm_time = times[alarm_before][0]
                warning_time = (t_cut - first_alarm_time) * 1000  # ms
                if warning_time > 0:
                    tprs.append(1)
                    warning_times.append(warning_time)
                else:
                    tprs.append(0)
            else:
                tprs.append(0)

        tpr = np.mean(tprs)
        mwt = np.mean(warning_times) if warning_times else np.nan

        if tpr > best['tpr'] or (
                tpr == best['tpr'] and not np.isnan(mwt)
                and (np.isnan(best['mwt']) or mwt > best['mwt'])):
            best = {'tpr': tpr, 'mwt': mwt, 'threshold': thresh}

    return best


if __name__ == '__main__':
    print('Loading test data...')
    with open(f'{DATA_DIR}/test_raw_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    X_test_ts = np.load(f'{DATA_DIR}/test_X_ts.npy')
    y_test    = np.load(f'{DATA_DIR}/test_y.npy')
    print(f'Samples: {len(samples)}, X_ts: {X_test_ts.shape}')

    all_metrics = {}

    # Deep learning models
    for name in ['lstm', 'transformer', 'cnn']:
        print(f'\nEvaluating {name}...')
        model = load_model(name)
        if model is None:
            continue
        metrics = compute_metrics_for_model(
            name, model, samples, X_test_ts, y_test)
        all_metrics[name] = metrics
        print(f'  TPR: {metrics["tpr"]:.3f}  '
              f'MWT: {metrics["mwt"]:.1f}ms  '
              f'Threshold: {metrics["threshold"]:.2f}')

    # XGBoost
    print('\nEvaluating xgboost...')
    xgb_path = os.path.join(CKPT_DIR, 'xgboost_clean.pkl')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            xgb_model = pkl.load(f)
        metrics = compute_metrics_for_model(
            'xgboost', xgb_model, samples, X_test_ts, y_test)
        all_metrics['xgboost'] = metrics
        print(f'  TPR: {metrics["tpr"]:.3f}  '
              f'MWT: {metrics["mwt"]:.1f}ms  '
              f'Threshold: {metrics["threshold"]:.2f}')
    # Save
    out = os.path.join(RESULTS_DIR, 'shot_level_metrics.json')
    with open(out, 'w') as f:
        json.dump({
            k: {
                'tpr':       float(v['tpr']),
                'mean_warning_time_ms': float(v['mwt'])
                    if not np.isnan(v['mwt']) else None,
                'threshold': float(v['threshold']),
                'note':      'FAR not computable — all test shots disruptive'
            }
            for k, v in all_metrics.items()
        }, f, indent=2)
    print(f'\nSaved to {out}')

    print('\n' + '='*55)
    print('SHOT-LEVEL ALARM METRICS (real t_cut disruption times)')
    print('='*55)
    print(f'{"Model":<15} {"TPR":>8} {"MWT (ms)":>12} {"Threshold":>10}')
    print('-'*55)
    for name, m in all_metrics.items():
        mwt = f'{m["mwt"]:.1f}' if not np.isnan(m['mwt']) else 'N/A'
        print(f'  {name:<13} {m["tpr"]:>8.3f} {mwt:>12} '
              f'{m["threshold"]:>10.2f}')
    print('\nNote: FAR not computable — all 50 test shots disrupted.')
    print('TPR = fraction of shots with alarm before t_cut.')
    print('MWT = mean warning time in ms before disruption.')
