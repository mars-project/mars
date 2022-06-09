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

Console
````````````````

The default metric backend is ``console``. It just logs the value when log level
is ``debug``.

Prometheus
````````````````

Firstly, we should download Prometheus. For details, please refer to
`Prometheus Getting Started
<https://prometheus.io/docs/prometheus/latest/getting_started/>`_.

Secondly, we can new a Mars session by configuring Prometheus backend as follows:

.. code-block:: python

    In [1]: import mars

    In [2]: session = mars.new_session(
       ...:     n_worker=1,
       ...:     n_cpu=2,
       ...:     web=True,
       ...:     config={"metrics.backend": "prometheus"}
       ...: )
    Finished startup prometheus http server and port is 15768
    Finished startup prometheus http server and port is 44303
    Finished startup prometheus http server and port is 63391
    Finished startup prometheus http server and port is 13722
    Web service started at http://0.0.0.0:15518

Thirdly, we should config Prometheus, more configurations please refer to
`Prometheus Configuration
<https://prometheus.io/docs/prometheus/latest/configuration/configuration/>`_.

.. code-block:: yaml

    scrape_configs:
      - job_name: 'mars'

        scrape_interval: 5s

        static_configs:
          - targets: ['localhost:15768', 'localhost:44303', 'localhost:63391', 'localhost:13722']


Then start Prometheus:

.. code-block:: shell

    $ prometheus --config.file=promconfig.yaml
    level=info ts=2022-06-07T13:05:01.484Z caller=main.go:296 msg="no time or size retention was set so using the default time retention" duration=15d
    level=info ts=2022-06-07T13:05:01.484Z caller=main.go:332 msg="Starting Prometheus" version="(version=2.13.1, branch=non-git, revision=non-git)"
    level=info ts=2022-06-07T13:05:01.484Z caller=main.go:333 build_context="(go=go1.13.1, user=brew@Mojave.local, date=20191018-01:13:04)"
    level=info ts=2022-06-07T13:05:01.485Z caller=main.go:334 host_details=(darwin)
    level=info ts=2022-06-07T13:05:01.485Z caller=main.go:335 fd_limits="(soft=256, hard=unlimited)"
    level=info ts=2022-06-07T13:05:01.485Z caller=main.go:336 vm_limits="(soft=unlimited, hard=unlimited)"
    level=info ts=2022-06-07T13:05:01.487Z caller=main.go:657 msg="Starting TSDB ..."
    level=info ts=2022-06-07T13:05:01.488Z caller=web.go:450 component=web msg="Start listening for connections" address=0.0.0.0:9090
    level=info ts=2022-06-07T13:05:01.494Z caller=head.go:514 component=tsdb msg="replaying WAL, this may take awhile"
    level=info ts=2022-06-07T13:05:01.495Z caller=head.go:562 component=tsdb msg="WAL segment loaded" segment=0 maxSegment=1
    level=info ts=2022-06-07T13:05:01.495Z caller=head.go:562 component=tsdb msg="WAL segment loaded" segment=1 maxSegment=1
    level=info ts=2022-06-07T13:05:01.497Z caller=main.go:672 fs_type=1a
    level=info ts=2022-06-07T13:05:01.497Z caller=main.go:673 msg="TSDB started"
    level=info ts=2022-06-07T13:05:01.497Z caller=main.go:743 msg="Loading configuration file" filename=promconfig_mars.yaml
    level=info ts=2022-06-07T13:05:01.501Z caller=main.go:771 msg="Completed loading of configuration file" filename=promconfig_mars.yaml
    level=info ts=2022-06-07T13:05:01.501Z caller=main.go:626 msg="Server is ready to receive web requests."

Fourthly, run a Mars task:

.. code-block:: python

    In [3]: import numpy as np

    In [4]: import mars.dataframe as md

    In [5]: df1 = md.DataFrame(np.random.randint(0, 3, size=(10, 4)),
       ...:                    columns=list('ABCD'), chunk_size=5)
       ...: df2 = md.DataFrame(np.random.randint(0, 3, size=(10, 4)),
       ...:                    columns=list('ABCD'), chunk_size=5)
       ...:
       ...: r = md.merge(df1, df2, on='A').execute()

Finally, we can check metrics in Prometheus web http://localhost:9090.

Ray
````````````````

We could config ``metrics.backend`` when creating a Ray cluster or new a session.

Metrics Naming Convention
------------------

We propose a naming convention for metrics as follows:

``namespace.[component].metric_name[_units]``

* ``namespace`` could be ``mars``.
* ``component`` could be `supervisor`, `worker` or `band` etc, and can be omitted.
* ``units`` is the metric unit which may be seconds when recording time, or
  ``_count`` when metric type is ``Counter``, ``_number`` when metric type is
  ``Gauge`` if there is no suitable unit.
