.. _pytorch:

*************************
Integrate with PyTorch
*************************

.. currentmodule:: mars.learn.contrib.pytorch

This introduction will give a brief tour about how to integrate `PyTorch
<https://pytorch.org/>`_ in Mars.

Installation
------------

If you are trying to use Mars on a single machine, e.g. on your laptop, make
sure PyTorch is installed.

You can install PyTorch via pip:

.. code-block:: bash

   pip3 install torch torchvision torchaudio

Visit `installation guide for PyTorch <https://pytorch.org/get-started/locally/>`_
for more information.

On the other hand, if you are about to use Mars on a cluster, maker sure
PyTorch is installed on each worker.

Prepare data
------------

The dataset here we used is `ionosphere dataset
<http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/>`_, click
link to download data.

Prepare PyTorch script
-------------------------

Now we create a Python file called ``torch_demo.py`` which contains the logic of
PyTorch.

.. code-block:: python

    import os

    import mars.dataframe as md
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.optim as optim
    import torch.utils.data
    from sklearn.preprocessing import LabelEncoder


    def prepare_data():
        df = md.read_csv('ionosphere.data', header=None)

        # split into input and output columns
        X = df.iloc[:, :-1].to_tensor().astype('float32')
        y = df.iloc[:, -1].to_tensor()

        # convert Mars tensor to numpy ndarray
        X, y = X.to_numpy(), y.to_numpy()

        # encode string to integer
        y = LabelEncoder().fit_transform(y)

        return X, y


    def get_model():
        return nn.Sequential(
            nn.Linear(34, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )


    def train():
        dist.init_process_group(backend="gloo")
        torch.manual_seed(42)

        data, labels= prepare_data()
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        train_dataset = torch.utils.data.TensorDataset(data, labels.float())
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                shuffle=False,
                                                sampler=train_sampler)


        model = nn.parallel.DistributedDataParallel(get_model())
        optimizer = optim.Adam(model.parameters(),
                            lr=0.001)
        criterion = nn.BCELoss()

        for epoch in range(150):  # 150 epochs

            for _, (batch_data, batch_labels) in enumerate(train_loader):
                
                running_loss = 0.0
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"epoch {epoch}, running_loss is {running_loss}")

        
    if __name__ == "__main__":
        train()  

Mars libraries including DataFrame and so forth could be used directly to
process massive data and accelerate preprocess.

Run PyTorch script via Mars
------------------------------

The PyTorch script can be submitted via :meth:`run_pytorch_script` now.

.. code-block:: ipython

    In [1]: from mars.learn.contrib.pytorch import run_pytorch_script

    In [2]: run_pytorch_script("torch_demo.py", n_workers=1)
    task: <Task pending coro=<Event.wait() running at ./mars-dev/lib/python3.7/asyncio/locks.py:293> wait_for=<Future pending cb=[<TaskWakeupMethWrapper object at 0x7f04c5027cd0>()]>>
    ...
    epoch 148, running_loss is 0.010707654990255833
    epoch 149, running_loss is 0.010623889975249767
    Out[4]: Object <op=RunPyTorch, key=d5c40e502b77310ef359729692233d56>

Distributed training or inference
---------------------------------

Refer to :ref:`deploy` section for deployment, or :ref:`k8s` section for
running Mars on Kubernetes.

As you can tell from ``torch_demo.py``, Mars will set environment variable 
automatically. Thus you don't need to worry about the distributed setting, what
you need do is to write a proper `distributed PyTorch script.
<https://pytorch.org/tutorials/beginner/dist_overview.html>`_.

Once a cluster exists, you can either set the session as default, the training
and prediction shown above will be submitted to the cluster, or you can specify
``session=***`` explicitly as well.

.. code-block:: python

   # A cluster has been configured, and web UI is started on <web_ip>:<web_port>
   import mars
   # set the session as the default one
   sess = mars.new_session('http://<web_ip>:<web_port>')

   # submitted to cluster by default
   run_tensorflow_script('torch_demo.py', n_workers=2)

   # Or, session could be specified as well
   run_tensorflow_script('torch_demo.py', n_workers=2, session=sess)

MarsDataset
------------

In order to use Mars to process data, we implemented a MarsDataset that can convert 
Mars object (mars.tensor, mars.dataframe, mars.series) to torch.util.data.Dataset.

.. code-block:: python

    from mars.learn.contrib.pytorch import MarsDataset

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')

    train_dataset = MarsDataset(data, labels)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                sampler=train_sampler)

Now, :meth:`run_pytorch_script` allow pass data to script. So you can preprocess data 
via mars, then pass data to script.

.. code-block:: python

    data = mt.random.rand(1000, 32, dtype='f4')
    labels = mt.random.randint(0, 2, (1000, 10), dtype='f4')

    run_pytorch_script(
        "torch_script", n_workers=2, data={'feature_data': data, 'labels': labels},
        command_argv=['multiple'], port=9945, session=sess)

``torch_script.py``

.. code-block:: python

    import sys


    def get_model():
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(),
        )


    def train(feature_data, labels):
        import torch.nn as nn
        import torch.distributed as dist
        import torch.optim as optim
        import torch.utils.data
        from mars.learn.contrib.pytorch import MarsDataset

        dist.init_process_group(backend='gloo')
        torch.manual_seed(42)

        data = feature_data
        labels = labels

        train_dataset = MarsDataset(data, labels)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                shuffle=False,
                                                sampler=train_sampler)

        model = nn.parallel.DistributedDataParallel(get_model())
        optimizer = optim.SGD(model.parameters(),
                            lr=0.01, momentum=0.5)
        criterion = nn.BCELoss()

        for _ in range(2):
            # 2 epochs
            for _, (batch_data, batch_labels) in enumerate(train_loader):
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print("Done!")


    if __name__ == "__main__":
        assert len(sys.argv) == 2
        assert sys.argv[1] == 'multiple'
        feature_data = feature_data
        labels = labels
        train(feature_data, labels)