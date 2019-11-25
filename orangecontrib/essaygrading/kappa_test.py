predictions= [8.76407,
9.60021,
9.17138,
9.1217,
8.73354,
9.2746,
8.48546,
9.13203,
8.11089,
8.22397,
7.29219,
9.28695,
7.51251,
10.8571,
7.3521,
7.84028,
10.0114,
8.28402,
8.24251,
7.34995,
8.70673,
9.64424,
9.05137,
8.38002,
7.85402,
8.22861,
5.49924,
7.84853,
8.86126,
9.52118,
6.76757,
11.3471,
8.39659,
9.36693,
7.27831,
10.094,
8.55437,
9.94694,
8.23599,
7.92601,
10.2808,
8.54997,
5.08308,
8.821,
8.8923,
8.02803,
7.861,
8.8950,
8.66688,
10.2574,
8.26426,
9.36797,
8.69643,
8.59088,
9.76509,
5.00565,
9.34433,
9.66703,
8.89375,
8.34055,
8.84892,
6.85809,
7.18286,
3.5327,
6.81307,
]

actual = [9,
10,
9,
9,
10,
9,
8,
10,
9,
8,
8,
9,
7,
11,
8,
7,
10,
8,
8,
6,
9,
11,
10,
10,
8,
9,
6,
8,
8,
9,
8,
10,
8,
10,
8,
10,
10,
12,
8,
8,
12,
8,
5,
8,
9,
8,
9,
10,
8,
10,
8,
10,
9,
9,
11,
5,
10,
9,
10,
8,
8,
7,
8,
2,
7,
]



# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
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


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))

    print(rater_a)
    print(rater_b)
    print(min_rating)
    print(max_rating)
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
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
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

import numpy as np

predictions = [round(x)-0 for x in predictions]
actual = [x-0 for x in actual]

actual = np.array(actual)
predictions = np.array(predictions)


qwk = quadratic_weighted_kappa(actual, predictions)
print(qwk)



