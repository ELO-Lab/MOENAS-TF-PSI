"""
Reference: https://github.com/msu-coinlab/pymoo
"""
import numpy as np
import time

from utils import (
    get_hashKey,
    set_seed,
    ElitistArchive,
    calculate_IGD_value,
    do_each_gen,
    finalize
)
from model import Population

class Algorithm:
    def __init__(self, **kwargs):
        """
        List of general hyperparameters:\n
        - *name*: the algorithms name
        - *pop_size*: the population size
        - *sampling*: the processor of sampling process
        - *crossover*: the processor of crossover process
        - *mutation*: the processor of mutation process
        - *survival*: the processor of survival process
        - *pop*: the population
        - *problem*: the problem which are being solved
        - *seed*: random seed
        - *nGens*: the number of generations was passed
        - *nEvals*: the number of evaluate function calls (or the number of trained architectures (in NAS problems))
        - *path_results*: the folder where the results will be saved on
        - *IGD_history*: list of IGD value each generation
        - *nEvals_history*: list of the number of trained architectures each generation
        - *reference_point*: the reference point (for calculation Hypervolume indicator)
        - *E_Archive*: the Elitist Archive
        """
        # General hyperparameters
        self.name = kwargs['name']

        self.pop_size = None
        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.survival = None

        self.pop = None
        self.problem = None

        self.seed = 0
        self.nGens = 0
        self.nEvals = 0

        self.path_results = None
        self.debug = False

        self.reference_point_search = [-np.inf, -np.inf]
        self.reference_point_evaluate = [-np.inf, -np.inf]

        self.E_Archive_search = ElitistArchive(log_each_change=True)
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

        """ [Method] - Potential Solutions Improving """
        self.PSI_processor = None

        self.ZC_predictor = None

        ##############################################################################
        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0

        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]

        self.tmp = 0.0
        self.nEvals_history = []

        self.E_Archive_search_history = []
        self.E_Archive_evaluate_history = []

        self.IGD_evaluate_history = []

        self.nEvals_runningtime_IGD_each_gen = []
        self.E_Archive_evaluate_each_gen = []

    """ ---------------------------------- Setting Up ---------------------------------- """
    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        self.pop = None

        self.seed = 0
        self.nGens = 0
        self.nEvals = 0

        self.path_results = None
        if self.PSI_processor is not None:
            self.PSI_processor.reset()
        self.ZC_predictor = None

        self.reference_point_search = [-np.inf, -np.inf]
        self.reference_point_evaluate = [-np.inf, -np.inf]

        self.E_Archive_search = ElitistArchive(log_each_change=True)
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0

        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]

        self.tmp = 0.0
        self.nEvals_history = []

        self.E_Archive_search_history = []
        self.E_Archive_evaluate_history = []

        self.IGD_evaluate_history = []

        self.nEvals_runningtime_IGD_each_gen = []
        self.E_Archive_evaluate_each_gen = []
        self._reset()

    """ ---------------------------------- Evaluate ---------------------------------- """
    def evaluate(self, X, using_zc=False):
        """
        - Call function *problem.evaluate* to evaluate the fitness values of solutions.
        """
        self.finish_executed_time_algorithm = time.time()
        self.executed_time_algorithm_history.append(
            self.executed_time_algorithm_history[-1] + (self.finish_executed_time_algorithm - self.start_executed_time_algorithm))

        comp_metric, perf_metric, benchmark_time, indicator_time = self.problem.evaluate(X, using_zc=using_zc)
        self.nEvals += 1
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + benchmark_time)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + benchmark_time + indicator_time)
        self.running_time_history.append(self.evaluated_time_history[-1] + self.tmp + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return [comp_metric, perf_metric]

    """ ---------------------------------- Initialize ---------------------------------- """
    def initialize(self):
        self._initialize()

    """ ---------------------------------- Mating ---------------------------------- """
    def mating(self, P):
        self._mating(P)

    """ ----------------------------------- Next ----------------------------------- """
    def next(self, pop):
        self._next(pop)

    """ ---------------------------------- Setting Up ---------------------------------- """
    def set_up(self, problem, seed):
        self.problem = problem
        self.seed = seed
        set_seed(self.seed)

        self.sampling.nSamples = self.pop_size

    """ ---------------------------------- Solving ---------------------------------- """
    def solve(self, problem, seed):
        self.set_up(problem, seed)
        self._solve()

    """ -------------------------------- Do Each Gen -------------------------------- """
    def do_each_gen(self):
        """
        Operations which algorithms perform at the end of each generation.
        """
        self._do_each_gen()
        do_each_gen(algorithm=self)
        if self.debug:
            print(f'{self.nEvals}/{self.problem.maxEvals}')

    """ -------------------------------- Perform when having a new change in EA -------------------------------- """
    def log_elitist_archive(self, **kwargs):
        self.nEvals_history.append(self.nEvals)

        non_dominated_front = np.array(self.E_Archive_search.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        self.reference_point_search[0] = max(self.reference_point_search[0], max(non_dominated_front[:, 0]))
        self.reference_point_search[1] = max(self.reference_point_search[1], max(non_dominated_front[:, 1]))
        self.E_Archive_search_history.append([self.E_Archive_search.X.copy(),
                                              self.E_Archive_search.hashKey.copy(),
                                              self.E_Archive_search.F.copy()])

        dummy_pop = Population(1)
        for x in self.E_Archive_search.X:
            X = x
            hashKey = get_hashKey(x, self.problem.name)
            perf_metric, _, _ = self.problem.get_performance_metric(arch=X, final=True)
            comp_metric = self.problem.get_computational_metric(X)
            F = [comp_metric, perf_metric]
            dummy_pop[-1].set('X', X)
            dummy_pop[-1].set('hashKey', hashKey)
            dummy_pop[-1].set('F', F)
            self.E_Archive_evaluate.update(dummy_pop[-1])
        non_dominated_front = np.array(self.E_Archive_evaluate.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        self.reference_point_evaluate[0] = max(self.reference_point_evaluate[0], max(non_dominated_front[:, 0]))
        self.reference_point_evaluate[1] = max(self.reference_point_evaluate[1], max(non_dominated_front[:, 1]))

        IGD_value_evaluate = calculate_IGD_value(pareto_front=self.problem.pareto_front_testing,
                                                 non_dominated_front=non_dominated_front)

        self.IGD_evaluate_history.append(IGD_value_evaluate)
        self.E_Archive_evaluate_history.append([self.E_Archive_evaluate.X.copy(),
                                                self.E_Archive_evaluate.hashKey.copy(),
                                                self.E_Archive_evaluate.F.copy()])
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

    """ --------------------------------------------------- Finalize ----------------------------------------------- """
    def finalize(self):
        self._finalize()
        finalize(algorithm=self)

    """ -------------------------------------------- Abstract Methods -----------------------------------------------"""
    def _solve(self):
        self.start_executed_time_algorithm = time.time()
        self.initialize()
        self.do_each_gen()
        while self.nEvals < self.problem.maxEvals:
            self.nGens += 1
            self.next(self.pop)
            self.do_each_gen()
        self.finalize()

    def _reset(self):
        pass

    def _initialize(self):
        pass

    def _mating(self, P):
        O = self.crossover.do(self.problem, P, algorithm=self)
        O = self.mutation.do(self.problem, P, O, algorithm=self)
        return O

    def _next(self, pop):
        offsprings = self._mating(pop)
        pool = pop.merge(offsprings)
        pop = self.survival.do(pool, self.pop_size)
        if self.PSI_processor is not None:
            pop = self.PSI_processor.do(pop=pop, algorithm=self, problem=self.problem)
        self.pop = pop

    def _do_each_gen(self):
        pass

    def _finalize(self):
        pass

if __name__ == '__main__':
    pass
