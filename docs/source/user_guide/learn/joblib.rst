.. _joblib:

*********************
Integrate with joblib
*********************

Joblib is a library integrated with scikit-learn to make machine learning jobs
parallel.  We create a backend for joblib with :doc:`Mars remote <remote>` and
users can make their scikit-learn tasks parallel with Mars.

To enable the backend, you need to register it with the code below.

.. code-block:: python

    from mars.learn.contrib.joblib import register_mars_backend
    register_mars_backend()

After that, it is possible to create a Mars parallel backend with Mars service
endpoint or existing Mars session.  When nothing specified, default or local
session will be used.

.. code-block:: python

    import joblib
    # create with Mars endpoint
    with joblib.parallel_backend('mars', service='http://<host>:<port>'):
        # scikit-learn code
    # create with existing Mars session
    sess = new_session('http://<host>:<port>')
    with joblib.parallel_backend('mars', session=sess):
        # scikit-learn code

A simple example is shown below, where we fit a SVM classifier with randomized
search. All you need is to replace the service endpoint in
``joblib.parallel_backend`` with your own service endpoint.

.. code-block:: python

    import joblib
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVC

    from mars.learn.contrib.joblib import register_mars_backend
    register_mars_backend()

    digits = load_digits()
    param_space = {
        'C': np.logspace(-6, 6, 30),
        'gamma': np.logspace(-8, 8, 30),
        'tol': np.logspace(-4, -1, 30),
        'class_weight': [None, 'balanced'],
    }
    model = SVC(kernel='rbf')
    search = RandomizedSearchCV(model, param_space, cv=5, n_iter=10, verbose=10)

    with joblib.parallel_backend('mars', service='http://<host>:<port>'):
        search.fit(digits.data, digits.target)

Note that joblib can only be used with data small enough to be held inside a
single machine. For huge datasets, please use learning algorithms implemented
with Mars objects.
