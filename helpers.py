import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(hist, bins, err=None, ax=None, has_overflow=True, no_plot=False, **kwds):
    ax = plt.gca() if ax is None and not no_plot else ax
    if has_overflow:
        hist, err = _remove_overflow(hist, err)

    left, right = bins[:-1], bins[1:]
    midpoints = (left + right) / 2.0
    x = np.array([left, right]).T.flatten()
    y = np.array([hist, hist]).T.flatten()

    if not no_plot:
        line, = ax.plot(x, y, **kwds)
        if err is not None:
            if 'color' in kwds:
                del kwds['color']
            ax.errorbar(midpoints, hist, yerr=err, fmt='none',
                        color=line.get_color(), **kwds)
        ax.set_xlim(bins[0], bins[-1])
    return x, y, err
def _remove_overflow(hist, err):
    hist = hist[1:-1]
    if err is not None:
        if len(err) == len(hist) + 2: # +2 since we removed the overflow
            err = err[1:-1]
        elif len(err) == 2: # For asymmetric errors
            err = [err[0][1:-1], err[1][1:-1]]
        #TODO: What if len(hist) == 2 ?
        else:
            raise ValueError('err has wrong shape')
    return hist, err


def percentage_difference(a, b):
    return [((i-j)/i)*100 for i, j in zip(a,b)]

def folding(truth, migration, efficiency, acceptance):
    reco_folded = []
    for j in range(0,2):
        summation = 0;
        for i,t in enumerate(truth):
            summation += migration[i,j] * efficiency[i] * t
        reco_folded.append(summation * 1/acceptance[j])

    return reco_folded

def compare_method(truth_1, M_1, truth_2, M_2, acceptance_1=None, efficiency_1=None,
                   acceptance_2=None, efficiency_2=None):
    if not acceptance_1:
        reco_1 = np.matmul(M_1, truth_1)
        reco_2 = np.matmul(M_2, truth_2)
        reco_2_folded = np.matmul(M_1, truth_2)
    else:
        reco_1 = folding(truth_1, M_1, efficiency_1, acceptance_1)
        reco_2 = folding(truth_2, M_2, efficiency_2, acceptance_2)
        reco_2_folded = folding(truth_2, M_1, efficiency_1, acceptance_1)
    TF_1 = reco_1/truth_1
    print("Our truth_1 = %s and our reco_1 = %s" % (str(truth_1), str(reco_1)))
    print("The migration matrix M_1 = ")
    print(np.matrix(np.fliplr(M_1)))
    print("The transfer function TF_1 = %s" % str(TF_1))
    TF_2 = reco_2/truth_2
    print("Our truth_2 = %s and our reco_2 = %s" % (str(truth_2), str(reco_2)))
    print("The migration matrix M_2 = ")
    print(np.matrix(np.fliplr(M_2)))
    print("The transfer function TF_2 = %s" % str(TF_2))
    reco_2_transfer = truth_2 * TF_1
    print(reco_2_transfer)
    print("Difference between folded and actual reco_2 = %s %%" % str(percentage_difference(reco_2, reco_2_folded)))
    print("Difference between transfer function and actual reco_2 = %s %%" % str(percentage_difference(reco_2, reco_2_transfer)))
    
def unfold(data, migration, acceptance, efficiency):
    migration_inv = np.linalg.inv(migration)
    
    unfolded = []
    for i in range(0, len(data)):
        summation = 0;
        for j,d in enumerate(data):
            summation += migration_inv[i,j] * acceptance[j] * d
        unfolded.append(summation * 1/efficiency[i])

    return unfolded
