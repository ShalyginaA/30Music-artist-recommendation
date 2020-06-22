def MAPk(y_true, y_pred, k=20):
    """
    average precision at k with k=20
    """
    actual = set(y_true)
    
    # precision at i is a percentage of correct 
    # items among first i recommendations; the
    # correct count will be summed up by n_hit
    n_hit = 0
    precision = 0
    for i, p in enumerate(y_pred, 1):
        if p in actual:
            n_hit += 1
            precision += n_hit / i

    avg_precision = precision / min(len(actual), k)
    return avg_precision


def precision(y_true, y_pred):
    if len(y_pred) == 0:
        return 0
    else:
        actual = set(y_true)
        n_hit = 0
        for p in y_pred:
            if p in actual:
                n_hit += 1
        return n_hit/len(y_pred)
    
    
def recall(y_true, y_pred):
    if len(y_pred) == 0:
        return 0
    else:
        actual = set(y_true)
        n_hit = 0
        for p in y_pred:
            if p in actual:
                n_hit += 1
        return n_hit/len(actual)