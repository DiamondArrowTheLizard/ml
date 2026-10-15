from math import log2


def calc_sturges_rule(observations_count):
    return 1 + int(log2(observations_count))