
import numpy as np
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score

# Dados dos conjuntos de etiquetas del mismo tamaño genera su matriz de confusión
def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

# Genera el histograma (número de apariciones de cada etiqueta)
def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0]*num_ratings
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

# Dada la matriz de confusión devuelve los conjuntos de etiquetas
def genera_ys(m):
    s = len(m)
    y_real = []
    y_pred = []
 
    for i in range(s):
        for j in range(s):
            for _ in range(m[i][j]):
                y_real.append(i)
                y_pred.append(j)
        
    return y_real,y_pred

# Dada la matriz de confusión, devuelve la matriz de confusión penalizada
def modify(m):
    m2 = deepcopy(m)
    s = len(m)
    for i in range(s):
        for j in range(s):
            if i!=j:    
                m2[i][j] = (abs(i-j))*m2[i][j]
    return m2

# Calcula kappa con pesos cuadráticos
def quadratic_weighted_kappa(y, y_pred):

    rater_a = np.array(y, dtype=int)
    rater_b = np.array(y_pred, dtype=int)
    assert(len(rater_a) == len(rater_b))
 
    min_rating = min(min(rater_a), min(rater_b))    
    max_rating = max(max(rater_a), max(rater_b))

    conf_mat = Cmatrix(rater_a, rater_b, min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0) # (i-j)^2/ (k-1)^2
            
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    
    return (1.0 - numerator / denominator)

# Calcula las métricas kappa, kappa con pesos cuadráticos y kappa penalizada
def calcula_kappas(y_real,y_pred):

    kappa = cohen_kappa_score(y_real,y_pred)

    m = Cmatrix(y_real, y_pred)
    m_penalizada = modify(m)
    y_real2,y_pred2 = genera_ys(m_penalizada)
    pkappa = cohen_kappa_score(y_real2, y_pred2)
    
    qwkappa = quadratic_weighted_kappa(y_real, y_pred)

    return kappa, qwkappa, pkappa