.. _metrics:

Metrics
====================

Mars has a unified metrics API and three different backends.

A Unified Metrics API
------------------

Mars metrics API are in ``mars/metrics/api.py`` and there are four metric types:

* ``Counter`` is a cumulative type of data which represents a monotonically increasing number.
* ``Gauge`` is a single numerical value.
* ``Meter`` is the rate at which a set of events occur. we can use it as qps or tps.
* ``Histogram`` is a type of statistics which records the average value of a window data.

And we can use these types as follows:

.. code-block:: python

    # Four metrics have a unified parameter list:
    # 1. Declarative method: Metrics.counter(name: str, description: str = "", tag_keys: Optional[Tuple[str]] = None)
    # 2. Record method: record(value=1, tags: Optional[Dict[str, str]] = None)

    c1 = Metrics.counter('counter1', 'A counter')
    c1.record(1)

    c2 = Metrics.counter('counter2', 'A counter', ('service', 'tenant'))
    c2.record(1, {'service': 'mars', 'tenant': 'test'})

    g1 = Metrics.gauge('gauge1')
    g1.record(1)

    g2 = Metrics.gauge('gauge2', 'A gauge', ('service', 'tenant'))
    g2.record(1, {'service': 'mars', 'tenant': 'test'})

    m1 = Metrics.meter('meter1')
    m1.record(1)

    m2 = Metrics.meter('meter1', 'A meter', ('service', 'tenant'))
    m2.record(1, {'service': 'mars', 'tenant': 'test'})

    h1 = Metrics.histogram('histogram1')
    h1.record(1)

    h2 = Metrics.histogram('histogram1', 'A histogram', ('service', 'tenant')))
    h2.record(1, {'service': 'mars', 'tenant': 'test'})

**Note**: If ``tag_keys`` is declared, ``tags`` must be specified when invoking
``record`` method and tags' keys must be consistent with ``tag_keys``.

Three different Backends
------------------

Mars metrics support three different backends:

* ``console`` is used for debug and it just prints the value.
* ``prometheus`` is an open-source systems monitoring and alerting toolkit.
* ``ray`` is a metric backend which just runs on ray engine.

We can choose a metric backend by configuring ``metrics.backend`` in
``mars/deploy/oscar/base_config.yml`` or its descendant files.

Metrics Naming Convention
------------------

We propose a naming convention for metrics as follows:

``namespace.[component].metric_name[_units]``

* ``namespace`` could be ``mars``.
* ``component`` could be `supervisor`, `worker` or `band` etc, and can be omitted.
* ``units`` is the metric unit which may be seconds when recording time, or
  ``_count`` when metric type is ``Counter``, ``_number`` when metric type is
  ``Gauge`` if there is no suitable unit.
