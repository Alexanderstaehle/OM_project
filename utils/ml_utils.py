import pickle
import time

import tensorflow as tf
from tensorflow import keras


def check_tpu_gpu():
    try:  # detect TPUs
        tpu = None
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError:  # detect GPUs
        strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines

    print("Number of accelerators: ", strategy.num_replicas_in_sync)


class ModelState:
    def __init__(
            self,
            weights=None,
            history=None,
            times=None,
    ):
        self.weights = weights
        self.history = history
        self.times = times


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train_model(model, train, validation, epochs, extra_callbacks=[], verbose=0):
    time_callback = TimeHistory()
    history = model.fit(
        train,
        epochs=epochs,
        validation_data=validation,
        callbacks=[time_callback] + extra_callbacks,
        verbose=verbose,
    )
    return get_model_state(model, history, time_callback)


def get_model_state(model, model_history, time_callback):
    model_state = ModelState()
    model_state.history = model_history.history
    model_state.times = time_callback.times
    model_state.weights = [w.value() for w in model.weights]
    return model_state


def save_model_state(model_state, filename):
    model_state_serialize = {}
    for key, state in model_state.items():
        model_state_serialize[key] = (state.weights, state.history, state.times)
    pickle.dump(model_state_serialize,
                open(path_from_filename(filename), "wb"))


def load_model_state(filename):
    model_state_serialize = pickle.load(open(path_from_filename(filename), "rb"))
    model_state_by_key = {}
    for key, state in model_state_serialize.items():
        model_state_by_key[key] = ModelState(weights=state[0], history=state[1], times=state[2])
    return model_state_by_key

def path_from_filename(filename, format_ = "pickle"):
    return f"tmp/{filename}.{format_}"

def load_model_from_filename(filename, format_ = "pickle"):
    return keras.models.load_model(path_from_filename(filename, format_))

def save_sharpnesses_dict(sharpnesses, filename = "sharpnesses"):
    with open(path_from_filename(filename), 'wb') as file:
        pickle.dump(sharpnesses, file)
    
def load_sharpnesses_dict(filename = "sharpnesses"):
    with open(path_from_filename(filename), 'rb') as file:
        sharpnesses = pickle.load(file)
    return sharpnesses