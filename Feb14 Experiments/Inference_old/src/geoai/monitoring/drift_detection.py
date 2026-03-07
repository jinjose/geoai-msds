import numpy as np

def population_stability_index(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum((expected_perc - actual_perc) * 
                 np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi
