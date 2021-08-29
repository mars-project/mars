.. _integrate_tensorflow:

*************************
Integrate with TensorFlow
*************************

.. currentmodule:: mars.learn.contrib.tensorflow

This introduction will give a brief tour about how to integrate `TensorFlow
<https://www.tensorflow.org>`_ in Mars.

This tutorial is based on TensorFlow 2.0.

Installation
------------

If you are trying to use Mars on a single machine, e.g. on your laptop, make
sure TensorFlow is installed.

You can install TensorFlow via pip:

.. code-block:: bash

   pip install tensorflow

Visit `installation guide for TensorFlow <https://www.tensorflow.org/install>`_
for more information.

On the other hand, if you are about to use Mars on a cluster, maker sure
TensorFlow is installed on each worker.

Prepare data
------------

The dataset here we used is `ionosphere dataset
<http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/>`_, click
link to download data.

Prepare TensorFlow script
-------------------------

Now we create a Python file called ``tf_demo.py`` which contains the logic of
TensorFlow.

.. code-block:: python

    import os

    import mars.dataframe as md
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense


    def prepare_data():
        df = md.read_csv('ionosphere.data', header=None)

        # split into input and output columns
        X = df.iloc[:, :-1].to_tensor().astype('float32')
        y = df.iloc[:, -1].to_tensor()

        # convert Mars tensor to numpy ndarray
        X, y = X.to_numpy(), y.to_numpy()

        # encode string to integer
        y = LabelEncoder().fit_transform(y)

        # split into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        print(X_train.size, X_test.size, y_train.size, y_test.size)

        return X_train, X_test, y_train, y_test


    def get_model(n_features):
        model = Sequential()
        model.add(Dense(10, activation='relu', kernel_initializer='he_normal',
                        input_shape=(n_features,)))
        model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def train():
        X_train, X_test, y_train, y_test = prepare_data()

        model = get_model(X_train.shape[1])

        # fit model
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
        # evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test accuracy: %.3f' % acc)


    if __name__ == '__main__':
        if 'TF_CONFIG' in os.environ:
            # distributed TensorFlow
            multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

            with multiworker_strategy.scope():
                train()
        else:
            train()

Mars libraries including DataFrame and so forth could be used directly to
process massive data and accelerate preprocess.

Run TensorFlow script via Mars
------------------------------

The TensorFlow script can be submitted via :meth:`run_tensorflow_script` now.

.. code-block:: ipython

    In [1]: from mars.learn.contrib.tensorflow import run_tensorflow_script

    In [2]: run_tensorflow_script('tf_demo.py', n_workers=1)
    2020-04-28 15:40:38.284763: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    sh: sysctl: command not found
    2020-04-28 15:40:38.301635: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fd29699c020 executing computations on platform Host. Devices:
    2020-04-28 15:40:38.301656: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
    2020-04-28 15:40:38.303779: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:258] Initialize GrpcChannelCache for job worker -> {0 -> localhost:2221}
    2020-04-28 15:40:38.304476: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:365] Started server with target: grpc://localhost:2221
    7990 3944 235 116
    WARNING:tensorflow:`eval_fn` is not passed in. The `worker_fn` will be used if an "evaluator" task exists in the cluster.
    WARNING:tensorflow:`eval_strategy` is not passed in. No distribution strategy will be used for evaluation.
    WARNING:tensorflow:ModelCheckpoint callback is not provided. Workers will need to restart training if any fails.
    WARNING:tensorflow:`eval_fn` is not passed in. The `worker_fn` will be used if an "evaluator" task exists in the cluster.
    WARNING:tensorflow:`eval_strategy` is not passed in. No distribution strategy will be used for evaluation.
    WARNING:tensorflow:ModelCheckpoint callback is not provided. Workers will need to restart training if any fails.
    Test accuracy: 0.931
    2020-04-28 15:40:45.906407: W tensorflow/core/common_runtime/eager/context.cc:290] Unable to destroy server_ object, so releasing instead. Servers don't support clean shutdown.
    Out[2]: {'status': 'ok'}

Distributed training or inference
---------------------------------

Refer to :ref:`deploy` section for deployment, or :ref:`k8s` section for
running Mars on Kubernetes.

As you can tell from ``tf_demo.py``, Mars will set environment variable
``TF_CONFIG`` automatically.  ``TF_CONFIG`` contains cluster and task
information.  Thus you don't need to worry about the distributed setting, what
you need do is to choose a proper `distributed strategy
<https://www.tensorflow.org/guide/distributed_training#types_of_strategies>`_.

Once a cluster exists, you can either set the session as default, the training
and prediction shown above will be submitted to the cluster, or you can specify
``session=***`` explicitly as well.

.. code-block:: python

   # A cluster has been configured, and web UI is started on <web_ip>:<web_port>
   import mars
   # set the session as the default one
   sess = mars.new_session('http://<web_ip>:<web_port>')

   # submitted to cluster by default
   run_tensorflow_script('tf_demo.py', n_workers=1)

   # Or, session could be specified as well
   run_tensorflow_script('tf_demo.py', n_workers=1, session=sess)

Use ``gen_tensorflow_dataset``
---------------------------------

You can convert Mars data(:class:`mars.tensor.Tensor`, :class:`mars.dataframe.DataFrame`,
:class:`mars.dataframe.Series`) to `tf.data.Dataset <https://tensorflow.google.
cn/api_docs/python/tf/data/Dataset>`_ by :meth:`gen_tensorflow_dataset`. It also 
support :class:`numpy.ndarray`, :class:`pandas.DataFrame`, :class:`pandas.Series`.

.. code-block:: python

    In [1]: data = mt.tensor([[1, 2], [3, 4]])
    In [2]: dataset = gen_tensorflow_dataset(data)
    In [3]: list(dataset.as_numpy_iterator())
    Out[3]: [array([1, 2]), array([3, 4])]

    In [1]: data1 = mt.tensor([1, 2]); data2 = mt.tensor([3, 4]); data3 = mt.tensor([5, 6])
    In [2]: dataset = gen_tensorflow_dataset((data1, data2, data3))
    In [3]: list(dataset.as_numpy_iterator())
    Out[3]: [(1, 3, 5), (2, 4, 6)]

Now, you can preprocess the data via mars, and pass data to script.

.. code-block:: python
    
    import mars.dataframe as md
    from sklearn.preprocessing import LabelEncoder
    from mars.learn.contrib.tensorflow import run_tensorflow_script 


    df = md.read_csv('ionosphere.data', header=None)
    X = df.iloc[:, :-1].astype('float32')
    y = df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y.execute().fetch())
    X.exeute()
    X_train = X[:len(X)*0.7]
    y_train = y[:len(y)*0.7]
    X_test = X[len(X)*0.7:]
    y_test = y[len(y)*0.7:]
    
    run_tensorflow_script(
        "tf_demo.py", n_workers=2, data={'X_train': X_train, 'y_train': y_train, 
        'X_test':X_test, 'y_test': y_test}, port=9945, session=sess)

``tf_demo.py``

.. code-block:: python

    import os

    from mars.learn.contrib.tensorflow import gen_tensorflow_dataset
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense


    def get_model(n_features):
        model = Sequential()
        model.add(Dense(10, activation='relu', kernel_initializer='he_normal',
                        input_shape=(n_features,)))
        model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def train(X_train, X_test, y_train, y_test):
        model = get_model(X_train.shape[1])

        db_train = gen_tensorflow_dataset((X_train, y_train))
        db_train = db_train.batch(32)
        db_test = gen_tensorflow_dataset((X_test, y_test))
        db_test = db_test.batch(32)

        # fit model
        model.fit(db_train, epochs=150)
        # evaluate
        loss, acc = model.evaluate(db_train)
        print('Test accuracy: %.3f' % acc)


    if __name__ == '__main__':
        X_train = globals()['X_train']
        y_train = globals()['y_train']
        X_test = globals()['X_test']
        y_test = globals()['y_test']

        if 'TF_CONFIG' in os.environ:
            # distributed TensorFlow
            multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

            with multiworker_strategy.scope():
                train(X_train, X_test, y_train, y_test)
        else:
            train(X_train, X_test, y_train, y_test)

result:

.. code-block:: ipython

    Epoch 1/150
    8/Unknown - 3s 292ms/step - loss: 0.6650 - accuracy: 0.6255      8/Unknown - 3s 292ms/step - loss: 0.6650 - accura8/8 [==============================] - 3s 292ms/step - loss: 0.6650 - accuracy: 0.6255
    8/8 [==============================] - 3s 292ms/step - loss: 0.6650 - accuracy: 0.6255
    Epoch 2/150
    Epoch 2/150
    1/8 [==>...........................] - ETA: 2s - loss: 0.6864 - accuracy: 0.78121/8 [==>...........................] - E
    7/8 [=========================>....] - ETA: 0s - loss: 0.6540 - accuracy: 0.64297/8 [=========================>....] - E
    8/8 [==============================] - ETA: 0s - loss: 0.6487 - accuracy: 0.64948/8 [==============================] - E
    8/8 [==============================] - 2s 295ms/step - loss: 0.6487 - accuracy: 0.6494
    8/8 [==============================] - 2s 295ms/step - loss: 0.6487 - accuracy: 0.6494
    Epoch 3/150
    Epoch 3/150
    ...
    Epoch 150/150
    Epoch 150/150
    1/8 [==>...........................] - ETA: 3s - loss: 0.0133 - accuracy: 1.00001/8 [==>...........................] - E
    6/8 [=====================>........] - ETA: 0s - loss: 0.0484 - accuracy: 0.98446/8 [=====================>........] - E
    8/8 [==============================] - ETA: 0s - loss: 0.0406 - accuracy: 0.98808/8 [==============================] - E
    8/8 [==============================] - 3s 385ms/step - loss: 0.0406 - accuracy: 0.9880
    8/8 [==============================] - 3s 385ms/step - loss: 0.0406 - accuracy: 0.9880
    Test accuracy: 0.988
    Test accuracy: 0.988
