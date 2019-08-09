from .utils import Composer


class JaxOptimizer(Composer):
    def __init__(self, graph):
        super(JaxOptimizer, self).__init__(graph, 'numexpr')

    def optimize(self, keys=None):
        self.compose(keys)
