"""Shared, deterministic experiment helpers; never mutate the deployed release."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, fbeta_score, confusion_matrix, brier_score_loss)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KAMP = ROOT / '준비패키지' / 'kamp'
SEED = 20261101

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save_json(path, data):
    def convert(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, (Path, pd.Timestamp)): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=convert, allow_nan=False), encoding='utf-8')

def metrics(y, p, threshold=.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # Stable ties preserve chronology; constant baselines have no useful ranking.
    order = np.argsort(-p, kind='stable')
    out = dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
        threshold=float(threshold), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
        f2=fbeta_score(y, pred, beta=2, zero_division=0),
        roc_auc=roc_auc_score(y, p) if len(np.unique(y))==2 else None,
        average_precision=average_precision_score(y, p) if y.sum() else None,
        brier=brier_score_loss(y, p), alert_rate=float(pred.mean()),
        accuracy=float(np.mean(pred==y)))
    for fraction in [.2, .3]:
        k = math.ceil(len(y)*fraction)
        out[f'precision_at_{int(fraction*100)}pct'] = float(y[order[:k]].mean())
        out[f'k_{int(fraction*100)}pct'] = k
    return out

def choose_threshold(y, p, cap=.4):
    # A predeclared demonstration review budget, not a factory requirement.
    choices = [metrics(y, p, t) for t in np.r_[np.arange(.05, .951, .025), 1.000001]]
    eligible = [m for m in choices if m['alert_rate'] <= cap]
    return max(eligible, key=lambda m: (m['f2'], m['precision'], m['threshold']))['threshold']

def calibration(y, p):
    y, p = np.asarray(y), np.asarray(p)
    bins=[]
    for i,(lo,hi) in enumerate(zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001])):
        mask=(p>=lo)&(p<hi)
        bins.append(dict(low=lo, high=min(hi,1), n=int(mask.sum()),
            mean_score=float(p[mask].mean()) if mask.any() else None,
            observed_rate=float(y[mask].mean()) if mask.any() else None))
    return bins

class StableColumns(BaseEstimator, TransformerMixin):
    """Fit-only missingness, variance and correlation pruning with stable ordering."""
    def __init__(self, max_missing=.4, max_correlation=.98):
        self.max_missing=max_missing
        self.max_correlation=max_correlation
    def fit(self, X, y=None):
        a=pd.DataFrame(X).replace([np.inf,-np.inf],np.nan)
        eligible=[i for i in range(a.shape[1]) if a.iloc[:,i].isna().mean()<=self.max_missing
                  and a.iloc[:,i].nunique(dropna=True)>1]
        corr=a.iloc[:,eligible].corr().abs().fillna(0).to_numpy()
        kept=[]
        for j in range(len(eligible)):
            if not kept or np.all(corr[j,kept]<self.max_correlation): kept.append(j)
        self.indices_=np.array([eligible[j] for j in kept])
        if not len(self.indices_): raise ValueError('No usable training features')
        return self
    def transform(self, X):
        return np.asarray(X, dtype=float)[:,self.indices_]

class TopFeatures(BaseEstimator, TransformerMixin):
    """Train-only ANOVA ranking, with k bounded by available features."""
    def __init__(self, k=8): self.k=k
    def fit(self,X,y):
        from sklearn.feature_selection import f_classif
        scores,_=f_classif(X,y)
        scores=np.nan_to_num(scores,nan=-1,posinf=1e100,neginf=-1)
        self.indices_=np.argsort(-scores,kind='stable')[:min(self.k,len(scores))]
        return self
    def transform(self,X): return np.asarray(X)[:,self.indices_]

def compact_features(data):
    """Input-only relations calculable from the application's original 36 fields."""
    groups={'quench':[f'소입로 온도 {i} Zone' for i in range(1,5)],
            'salt':[f'솔트조 온도 {i} Zone' for i in [1,2]],
            'dryer':[f'건조로 온도 {i} Zone' for i in [1,2]],
            'conveyor':[f'솔트 컨베이어 온도 {i} Zone' for i in [1,2]]}
    out=pd.DataFrame(index=data.index)
    for name,cols in groups.items():
        avg=data[[c+'_Avg' for c in cols]]
        out[name+'_mean']=avg.mean(axis=1)
        out[name+'_zone_spread']=avg.max(axis=1)-avg.min(axis=1)
        out[name+'_variability']=data[[c+'_Std' for c in cols]].mean(axis=1)
    out['cp_mean']=data['소입로 CP 값_Avg']
    out['cp_std']=data['소입로 CP 값_Std']
    out['quench_salt_gap']=out.quench_mean-out.salt_mean
    out['salt_conveyor_gap']=out.salt_mean-out.conveyor_mean
    return out
