.. _cluster_service:

Cluster Service
===============

Cluster service manages nodes in a Mars cluster and expose APIs for
other services to watch changes inside the cluster. The service uses different cluster backends
for different systems.

Configuration
-------------
.. code-block:: yaml

    cluster:
        backend: "<cluster backend name>"
        lookup_address: "<address of master>"
        node_timeout: "timeout seconds of nodes"
        node_check_interval: "check interval seconds for nodes"

APIs
----

.. currentmodule:: mars.services.cluster

.. autosummary::
   :toctree: generated/

   AbstractClusterBackend
   ClusterAPI
   WebClusterAPI
