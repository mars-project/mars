class JaxOptimizer(object):
    def __init__(self, graph):
        self._graph = graph

    @property
    def graph(self):
        return self._graph

    def optimize(self, keys=None):
        from .utils import Composer
        return Composer(self, keys).compose()
