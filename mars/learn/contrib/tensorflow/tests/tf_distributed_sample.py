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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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


assert json.loads(os.environ["TF_CONFIG"])["task"]["index"] in {0, 1}
assert len(sys.argv) == 2
assert sys.argv[1] == "multiple"

multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with multiworker_strategy.scope():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    model = get_model()
    model.fit(data, labels, epochs=2, batch_size=32)
