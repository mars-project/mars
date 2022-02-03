.. _contributing:

Contributing to Mars
====================

Mars is an open-sourced project released under Apache License 2.0. We welcome
and thanks for your contributions. Here are some guidelines you may find useful
when you want to make some change of Mars.

General Guidelines
------------------
Mars hosts and maintains its code on `Github
<https://github.com/mars-project/mars>`_. We provide a `generalized guide
<https://github.com/mars-project/mars/blob/master/CONTRIBUTING.rst>`_ for
opening issues and pull requests.

Setting Up a Development Environment
------------------------------------
Unless you want to develop or debug Mars under Windows, it is strongly
recommended to develop Mars under MacOS or Linux, where you can test all
functions of Mars. The steps listed below are applicable on MacOS and Linux.

Install in Conda
````````````````
It is recommended to develop Mars with conda. When you want to install Mars for
development, use the following steps to create an environment and install Mars
inside it:

.. code-block:: bash

    git clone https://github.com/mars-project/mars.git
    cd mars
    conda create -n mars-dev --file conda-spec.txt python=3.7
    source activate mars-dev
    pip install -e ".[dev]"

Change ``3.7`` into the version of Python you want to install, and ``mars-dev``
into your preferred environment name.

Other Python Distributions
``````````````````````````
Mars has a ``dev`` option for installation. When you want to install Mars for
development, use the following steps:

.. code-block:: bash

    pip install --upgrade setuptools pip
    git clone https://github.com/mars-project/mars.git
    cd mars
    pip install -e ".[dev]"

If you are using a system-wide Python installation and you only want to install
Mars for you, you can add ``--user`` to the ``pip install`` commands.

Verification
````````````
After installing Mars, you can check if Mars is installed correctly by running

.. code-block:: bash

    python -c "import mars; print(mars.__version__)"

If this command correctly outputs your Mars version, Mars is correctly
installed.

Rebuilding Cython Code
``````````````````````
Mars uses Cython to accelerate part of its code. After you change Cython source
code, you need to compile them into binaries by executing the command below on
the root of Mars project:

.. code-block:: bash

    python setup.py build_ext -i

Rebuilding Frontend Code
````````````````````````
Mars uses `React <https://reactjs.org>`_ to build its frontend. You need to
install `nodejs <https://nodejs.org>`_ to build it from source. After all
dependencies installed, simply use command below to build your frontend code:

.. code-block:: bash

    python setup.py build_web

Running Tests
-------------
It is recommended to use ``pytest`` to run Mars tests. A simple command below
will run all the tests of Mars:

.. code-block:: bash

    pytest mars

If you want to generate a coverage report as well, you can run:

.. code-block:: bash

    pytest --cov=mars --cov-report=html mars

Coverage report will be put into the directory ``htmlcov``.

The command above does not contain coverage data for Cython files by default.
To obtain coverage data about Cython files, you can run

.. code-block:: bash

    CYTHON_TRACE=1 python setup.py build_ext -i --force

before running the pytest command mentioned above. After report is generated,
it it recommended to remove all generated C files and binaries and rebuild
without ``CYTHON_TRACE``, as this option will reduce the performance of Mars.

Check Code Styles
-----------------
Before proposing changes to Mars, you need to make sure your code style meets
our requirements. Mars uses `black
<https://black.readthedocs.io/en/stable/index.html>`_ to enforce Python code
style.  Simply run command below to format your code style automatically:

.. code-block:: bash

    black mars

We also require relative import in code for all Mars modules. Use

.. code-block:: bash

    python ./ci/importcheck.py

to check if your code meet the requirement.

.. _build_documentation:

Building Documentations
-----------------------
Mars uses ``sphinx`` to build documents. You need to install necessary packages
with the command below to install these dependencies and build your documents
into HTML.

.. code-block:: bash

    pip install -r docs/requirements-doc.txt
    cd docs
    make html

The built documents are in ``docs/build/html`` directory.

When you want to create translations of Mars documents, you may append ``-l
<locale>`` after the ``I18NSPHINXLANGS`` variable in ``Makefile``. Currently
only simplified Chinese is supported. After that, run the command below to
generate portable files (``*.po``) for the documents, which are in
``docs/source/locale/<locale>/LC_MESSAGES``:

.. code-block:: bash

    cd docs
    make gettext

After that you can translate Mars documents into your language. Note that when
you run ``make gettext`` again, translations will be broken into a fixed-width
text. For Chinese translators, you need to install ``jieba`` to get this
effect.

When you finish translation, you can run

.. code-block:: bash

    cd docs
    # change LANG into the language you want to build
    make -e SPHINXOPTS="-D language='LANG'" html

to build the document in the language you just translated into.
