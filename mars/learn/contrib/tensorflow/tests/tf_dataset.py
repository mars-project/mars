# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import sys

import tensorflow as tf
from tensorflow.keras import layers
from mars.learn.contrib.tensorflow import gen_tensorflow_dataset
from tensorflow.python.data.ops.dataset_ops import DatasetV2


def get_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(feature_data, labels):
    data = feature_data
    labels = labels

    db_train = gen_tensorflow_dataset((data, labels))
    assert isinstance(db_train, DatasetV2)
    db_train = db_train.batch(32)

    model = get_model()
    model.fit(db_train, epochs=2)


if __name__ == "__main__":
    assert json.loads(os.environ["TF_CONFIG"])["task"]["index"] in {0, 1}
    assert len(sys.argv) == 2
    assert sys.argv[1] == "multiple"

    feature_data = globals()["feature_data"]
    labels = globals()["labels"]
    multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with multiworker_strategy.scope():
        train(feature_data, labels)
