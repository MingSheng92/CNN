from keras import backend as K

def unhinged(y_true, y_pred):
    return K.mean(1. - y_true * y_pred, axis=-1)


def sigmoid(y_true, y_pred):
    beta = 1.0
    return K.mean(K.sigmoid(-beta * y_true * y_pred), axis=-1)


def ramp(y_true, y_pred):
    beta = 1.0
    return K.mean(K.minimum(1., K.maximum(0., 1. - beta * y_true * y_pred)),
                  axis=-1)


def savage(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(1. / K.square(1. + K.exp(2 * y_true * y_pred)),
                  axis=-1)


def boot_soft(y_true, y_pred):
    beta = 0.95

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum((beta * y_true + (1. - beta) * y_pred) *
                  K.log(y_pred), axis=-1)