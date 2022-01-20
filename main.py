import os
import time
import argparse
from sys import platform
from datetime import datetime

from PSI import PSI_processor

from factory import get_problems, get_algorithm
from operators.crossover import PointCrossover
from operators.mutation import BitStringMutation
from operators.sampling.random_sampling import RandomSampling
from operators.selection import RankAndCrowdingSurvival

from zero_cost_methods import get_config_for_zero_cost_predictor, get_zero_cost_predictor

population_size_dict = {
    'NAS101': 20, 'NAS201-C10': 20, 'NAS201-C100': 20, 'NAS201-IN16': 20,
    'NAS201-T': 20,
}

def main(kwargs):
    if platform == "linux" or platform == "linux2":
        root_project = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    elif platform == "win32" or platform == "win64":
        root_project = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
    else:
        raise ValueError()

    if kwargs.path_results is None:
        try:
            os.makedirs(f'{root_project}/results/{kwargs.problem}')
        except FileExistsError:
            pass
        PATH_RESULTS = f'{root_project}/results/{kwargs.problem}'
    else:
        try:
            os.makedirs(f'{kwargs.path_results}/{kwargs.problem}')
        except FileExistsError:
            pass
        PATH_RESULTS = f'{kwargs.path_results}/{kwargs.problem}'

    ''' ============================================== Set up problem ============================================== '''
    path_data = root_project + '/data'
    problem = get_problems(problem_name=kwargs.problem, path_data=path_data)
    problem.set_up()

    ''' ==================================================================================================== '''
    pop_size = kwargs.pop_size
    if pop_size == 0:
        pop_size = population_size_dict[kwargs.problem]

    n_runs = kwargs.n_runs
    init_seed = kwargs.seed

    zero_cost_method = 'synflow'

    sampling = RandomSampling()
    crossover = PointCrossover('2X')
    mutation = BitStringMutation()

    algorithm = get_algorithm(algorithm_name=kwargs.algorithm)
    if algorithm.name == 'NSGA-II':
        survival = RankAndCrowdingSurvival()
    else:
        raise ValueError()

    """ Potential Solutions Improving """
    potential_solutions_improving = bool(kwargs.PSI)
    psi_nPoints = kwargs.PSI_nPoints
    psi_using_zc = bool(kwargs.PSI_using_zc)
    psi_processor = None
    if potential_solutions_improving:
        psi_processor = PSI_processor(nPoints=psi_nPoints, using_zc=psi_using_zc)

    algorithm.set_hyperparameters(pop_size=pop_size,
                                  sampling=sampling,
                                  crossover=crossover,
                                  mutation=mutation,
                                  survival=survival,
                                  PSI_processor=psi_processor,
                                  debug=bool(kwargs.debug))

    ''' ==================================== Set up experimental environment ======================================= '''
    time_now = datetime.now()

    dir_name = time_now.strftime(
        f'{kwargs.problem}_'
        f'{potential_solutions_improving}_{psi_nPoints}_{psi_using_zc}_'
        f'd%d_m%m_H%H_M%M_S%S')

    root_path = PATH_RESULTS + '/' + dir_name
    os.mkdir(root_path)
    print(f'--> Create folder {root_path} - Done\n')

    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]
    executed_time_list = []

    ''' =============================================== Log Information ============================================ '''
    print(f'******* PROBLEM *******')
    print(f'- Benchmark: {problem.name}')
    print(f'- Dataset: {problem.dataset}')
    print(f'- Maximum number of evaluations: {problem.maxEvals}')
    print(f'- The first objective (minimize): {problem.objective_0}')
    print(f'- The second objective (minimize): {problem.objective_1}')
    print(f'- Number of training epochs: {problem.epoch}\n')

    print(f'******* ALGORITHM *******')
    print(f'- Algorithm name: {algorithm.name}')
    print(f'- Population size: {pop_size}')
    print(f'- Crossover method: {algorithm.crossover.method}')
    print(f'- Mutation method: Bit-string')
    print(f'- Selection method: {algorithm.survival.name}\n')

    if potential_solutions_improving:
        print(f'******* POTENTIAL SOLUTIONS IMPROVING *******')
        print(f'- Method: {psi_processor.method_name}')
        print(f'- Local search on nPoints: {psi_processor.nPoints}\n')

    print(f'******* ENVIRONMENT *******')
    print(f'- Number of running experiments: {n_runs}')
    print(f'- Random seed each run: {random_seeds_list}')
    print(f'- Path for saving results: {root_path}')
    print(f'- Debug: {algorithm.debug}\n')

    with open(f'{root_path}/logging.txt', 'w') as f:
        f.write(f'******* PROBLEM *******\n')
        f.write(f'- Benchmark: {problem.name}\n')
        f.write(f'- Dataset: {problem.dataset}\n')
        f.write(f'- Maximum number of evaluations: {problem.maxEvals}\n')
        f.write(f'- The first objective (minimize): {problem.objective_0}\n')
        f.write(f'- The second objective (minimize): {problem.objective_1}\n')
        f.write(f'- Number of training epochs: {problem.epoch}\n\n')

        f.write(f'******* ALGORITHM *******\n')
        f.write(f'- Algorithm name: {algorithm.name}\n')
        f.write(f'- Population size: {pop_size}\n')
        f.write(f'- Crossover method: {algorithm.crossover.method}\n')
        f.write(f'- Mutation method: Bit-string\n')
        f.write(f'- Selection method: {algorithm.survival.name}\n\n')

        if potential_solutions_improving:
            f.write(f'******* POTENTIAL SOLUTIONS IMPROVING *******\n')
            f.write(f'- Method: {psi_processor.method_name}\n')
            f.write(f'- Local search on nPoints: {psi_processor.nPoints}\n\n')

        f.write(f'******* ENVIRONMENT *******\n')
        f.write(f'- Number of running experiments: {n_runs}\n')
        f.write(f'- Random seed each run: {random_seeds_list}\n')
        f.write(f'- Path for saving results: {root_path}\n')
        f.write(f'- Debug: {algorithm.debug}\n\n')

    ''' ==================================================================================================== '''
    for run_i in range(n_runs):
        algorithm.reset()
        print(f'---- Run {run_i + 1}/{n_runs} ----')
        random_seed = random_seeds_list[run_i]

        if psi_using_zc:
            config = get_config_for_zero_cost_predictor(problem=problem, seed=random_seed, path_data=path_data)
            ZC_predictor = get_zero_cost_predictor(config=config, method_type=zero_cost_method)
            if len(ZC_predictor.arch2score_dict) != 0:
                raise ValueError()
            problem.set_zero_cost_predictor(ZC_predictor)
            algorithm.set_hyperparameters(ZC_predictor=ZC_predictor)

        path_results = root_path + '/' + f'{run_i}'

        os.mkdir(path_results)
        s = time.time()

        algorithm.set_hyperparameters(path_results=path_results)
        algorithm.solve(problem, random_seed)

        executed_time = time.time() - s
        executed_time_list.append(executed_time)
        print('This run take', executed_time_list[-1], 'seconds')
    print('==' * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--problem', type=str, default='NAS201-C10', help='the problem name',
                        choices=['NAS101', 'NAS201-C10', 'NAS201-C100', 'NAS201-IN16'])

    ''' EVOLUTIONARY ALGORITHM '''
    parser.add_argument('--pop_size', type=int, default=0, help='the population size')
    parser.add_argument('--algorithm', type=str, default='NSGA-II', help='the algorithm name', choices=['NSGA-II'])

    ''' POTENTIAL SOLUTIONS IMPROVING '''
    parser.add_argument('--PSI', type=int, default=0)
    parser.add_argument('--PSI_nPoints', type=int, default=1)
    parser.add_argument('--PSI_using_zc', type=int, default=0)

    ''' ENVIRONMENT '''
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--n_runs', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    args = parser.parse_args()

    main(args)
