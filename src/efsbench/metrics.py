"""Classification metrics (minority class = 1).  Pure numpy / scipy."""
import numpy as np
from scipy.stats import rankdata


def confusion(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    return tp, fn, tn, fp


def recalls(y, pred):
    tp, fn, tn, fp = confusion(y, pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return tpr, tnr


def ba(y, pred):
    tpr, tnr = recalls(y, pred)
    return 0.5 * (tpr + tnr)


def gmean(y, pred):
    tpr, tnr = recalls(y, pred)
    return float(np.sqrt(tpr * tnr))


def acc(y, pred):
    return float((np.asarray(y) == np.asarray(pred)).mean())


def f1(y, pred):
    tp, fn, tn, fp = confusion(y, pred)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def mcc(y, pred):
    tp, fn, tn, fp = confusion(y, pred)
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else 0.0


def auc(y, score):
    """Mann-Whitney AUC with average ranks for ties; 0.5 if a class is missing."""
    y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def all_metrics(y, pred, score):
    tpr, tnr = recalls(y, pred)
    return dict(ba=0.5 * (tpr + tnr), gm=float(np.sqrt(tpr * tnr)), acc=acc(y, pred), f1=f1(y, pred),
                mcc=mcc(y, pred), auc=auc(y, score), tpr=tpr, tnr=tnr)
