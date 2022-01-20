"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
from algorithms import Algorithm
from model.individual import Individual

INF = 9999999
class NSGAII(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(name='NSGA-II', **kwargs)
        self.individual = Individual(rank=INF, crowding=-1)

    def _initialize(self):
        P = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            P[i].set('F', F)
            self.E_Archive_search.update(P[i], algorithm=self)
        self.pop = P

if __name__ == '__main__':
    pass
