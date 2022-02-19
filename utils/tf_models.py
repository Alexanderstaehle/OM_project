import tensorflow as tf
from keras import layers
from tensorflow import keras


def build_simple_cnn(x_train, dropout_prob=0.5):
    model = keras.models.Sequential()
    model.add(
        layers.Conv2D(input_shape=x_train.element_spec[0].shape[1:], filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=dropout_prob))
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=dropout_prob))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, use_bias=True, activation='softmax'))
    return model


def build_simple_dense_model(x_train):
    model = keras.models.Sequential([
        layers.Flatten(input_shape=x_train.element_spec[0].shape[1:]),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model


def build_cifar10_cnn(x_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.element_spec[0].shape[1:], activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model


# taken from https://github.com/sayakpaul/Sharpness-Aware-Minimization-TensorFlow
tf.config.run_functions_eagerly(False)


class SAMModel(tf.keras.Model):
    def __init__(self, base_model, rho=0.05, adaptive = False):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.base_model = base_model
        self.adaptive = adaptive
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(trainable_params, gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = (tf.math.pow(param, 2.0) if self.adaptive else 1.0) * grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss = self.compiled_loss(labels, predictions)

        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)

        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.base_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, params, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm((tf.math.abs(param) if self.adaptive else 1.0) * grad) for param, grad in zip(params, gradients) if grad is not None
            ])
        )
        return norm
