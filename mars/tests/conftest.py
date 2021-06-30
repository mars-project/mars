import pytest
from mars.serialization.ray import register_ray_serializers, unregister_ray_serializers
from mars.oscar.backends.router import Router
from mars.oscar.backends.ray.communication import RayServer
from mars.utils import lazy_import

ray = lazy_import('ray')


@pytest.fixture
def ray_start_regular(request):
    param = getattr(request, "param", {})
    if not param.get('enable', True):
        yield
    else:
        register_ray_serializers()
        yield ray.init(num_cpus=20)
        ray.shutdown()
        unregister_ray_serializers()
        Router.set_instance(None)
        RayServer.clear()


@pytest.fixture
def ray_large_cluster():
    try:
        from ray.cluster_utils import Cluster
    except ModuleNotFoundError:
        from ray._private.cluster_utils import Cluster
    cluster = Cluster()
    remote_nodes = []
    num_nodes = 3
    for i in range(num_nodes):
        remote_nodes.append(cluster.add_node(num_cpus=10))
        if len(remote_nodes) == 1:
            ray.init(address=cluster.address)
    register_ray_serializers()
    yield
    unregister_ray_serializers()
    Router.set_instance(None)
    RayServer.clear()
    ray.shutdown()
    cluster.shutdown()
