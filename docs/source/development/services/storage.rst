.. _storage_service:

Storage Service
===============

Configuration
-------------

.. code-block:: yaml

    storage:
        backends: ["plasma"]
        "<storage backend name>"： "<setup params>"

APIs
----

.. currentmodule:: mars.services.storage

.. autosummary::
   :toctree: generated/

   StorageAPI
   WebStorageAPI
