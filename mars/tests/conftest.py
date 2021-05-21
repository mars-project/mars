import pytest
from mars.serialization.ray import register_ray_serializers, unregister_ray_serializers
from mars.oscar.backends.router import Router
from mars.utils import lazy_import

ray = lazy_import('ray')


@pytest.fixture
def ray_start_regular(request):
    param = getattr(request, "param", {})
    if not param.get('enable', True):
        yield
    else:
        register_ray_serializers()
        yield ray.init(num_cpus=10)
        ray.shutdown()
        unregister_ray_serializers()
        Router.set_instance(None)
