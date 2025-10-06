import numpy as np


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_with_P(y_train, nb_classes, noise, random_state=0):

    if noise > 0.0:
        P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)
        keep_indices = np.arange(len(y_train))

    return y_train, P, keep_indices


def noisify_covid_asymmetric(y_train, noise, random_state=None):
    """mistakes for COVID-19 dataset:
        Covid <-> Viral_pneumonia
        Lung_opacity <-> Normal
    """
    nb_classes = 4
    P = np.eye(nb_classes)
    keep_indices = np.arange(len(y_train))
    n = noise

    if n > 0.0:
        # Covid <-> Viral_pneumonia (class 0 <-> 3)
        P[0, 0], P[0, 3] = 1. - n, n
        P[3, 3], P[3, 0] = 1. - n, n

        # Lung_opacity <-> Normal (class 1 <-> 2)
        P[1, 1], P[1, 2] = 1. - n, n
        P[2, 2], P[2, 1] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices

def noisify_binary_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 -> 0: n
        0 -> 1: .05
    """
    P = np.eye(2)
    n = noise

    keep_indices = np.arange(len(y_train))

    assert 0.0 <= n < 0.5

    if noise > 0.0:
        P[1, 1], P[1, 0] = 1.0 - n, n
        P[0, 0], P[0, 1] = 0.95, 0.05

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    keep_indices = None

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


