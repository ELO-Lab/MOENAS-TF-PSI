import numpy as np
import matplotlib.pyplot as plt
import pickle as p


def save_reference_point(reference_point, path_results, error='None'):
    p.dump(reference_point, open(f'{path_results}/reference_point({error}).p', 'wb'))


def save_Non_dominated_Front_and_Elitist_Archive(non_dominated_front, n_evals, elitist_archive, n_gens, path_results):
    """
    - This function is used to save the non-dominated front and Elitist Archive at the end of each generation.
    """
    p.dump([non_dominated_front, n_evals], open(f'{path_results}/non_dominated_front/gen_{n_gens}.p', 'wb'))
    p.dump(elitist_archive, open(f'{path_results}/elitist_archive/gen_{n_gens}.p', 'wb'))


def visualize_IGD_value_and_nEvals(nEvals_history, IGD_history, path_results, error='evaluate'):
    """
    - This function is used to visualize 'IGD_values' and 'nEvals' at the end of the search.
    """
    plt.xscale('log')
    plt.xlabel('#Evals')
    plt.ylabel('IGD value')
    plt.grid()
    plt.step(nEvals_history, IGD_history, where='post')
    plt.savefig(f'{path_results}/#Evals-IGD ({error})')
    plt.clf()

def visualize_runningtime_and_nEvals(running_time_history, path_results):
    """
    - This function is used to visualize 'running time' and 'nEvals' at the end of the search.
    """
    plt.xscale('log')
    plt.xlabel('#Evals')
    plt.ylabel('Total running time')
    plt.grid('--')
    plt.plot(range(1, len(running_time_history) + 1), running_time_history)
    plt.savefig(f'{path_results}/#Evals-RunningTime')
    plt.clf()

def visualize_Elitist_Archive_and_Pareto_Front(elitist_archive, pareto_front, objective_0, path_results, error='testing'):
    non_dominated_front = np.array(elitist_archive)
    non_dominated_front = np.unique(non_dominated_front, axis=0)

    plt.scatter(pareto_front[:, 0], pareto_front[:, 1],
                facecolors='none', edgecolors='b', s=40, label=f'Pareto-optimal Front')
    plt.scatter(non_dominated_front[:, 0], non_dominated_front[:, 1],
                c='red', s=15, label=f'Non-dominated Front')

    plt.xlabel(objective_0 + '(normalize)')
    plt.ylabel('Error')

    plt.legend()
    plt.grid()
    plt.savefig(f'{path_results}/non_dominated_front({error})')
    plt.clf()

def visualize_Elitist_Archive(elitist_archive, objective_0, path_results):
    non_dominated_front = np.array(elitist_archive)
    non_dominated_front = np.unique(non_dominated_front, axis=0)

    plt.scatter(non_dominated_front[:, 0], non_dominated_front[:, 1],
                facecolors='none', edgecolors='b', s=40, label=f'Non-dominated Front')

    plt.xlabel(objective_0 + '(normalize)')
    plt.ylabel('Error')

    plt.legend()
    plt.grid()
    plt.savefig(f'{path_results}/non_dominated_front')
    plt.clf()

def do_each_gen(**kwargs):
    algorithm = kwargs['algorithm']

    algorithm.nEvals_runningtime_IGD_each_gen.append([algorithm.nEvals, algorithm.running_time_history[-1],
                                                      algorithm.IGD_evaluate_history[-1]])
    algorithm.E_Archive_evaluate_each_gen.append([algorithm.E_Archive_evaluate.X.copy(),
                                                  algorithm.E_Archive_evaluate.hashKey.copy(),
                                                  algorithm.E_Archive_evaluate.F.copy()])


def finalize(**kwargs):
    algorithm = kwargs['algorithm']

    p.dump(algorithm.nEvals_runningtime_IGD_each_gen, open(f'{algorithm.path_results}/#Evals_runningtime_IGD_each_gen.p', 'wb'))

    p.dump(algorithm.E_Archive_evaluate_each_gen, open(f'{algorithm.path_results}/E_Archive_evaluate_each_gen.p', 'wb'))

    p.dump([algorithm.nEvals_history, algorithm.IGD_evaluate_history], open(f'{algorithm.path_results}/#Evals_and_IGD_evaluate.p', 'wb'))

    algorithm.benchmark_time_algorithm_history = np.array(algorithm.benchmark_time_algorithm_history)[1:]
    algorithm.indicator_time_history = np.array(algorithm.indicator_time_history)[1:]
    algorithm.evaluated_time_history = np.array(algorithm.evaluated_time_history)[1:]
    algorithm.executed_time_algorithm_history = np.array(algorithm.executed_time_algorithm_history)[1:]
    algorithm.running_time_history = np.array(algorithm.running_time_history)[1:]
    p.dump(algorithm.benchmark_time_algorithm_history, open(f'{algorithm.path_results}/benchmark_time.p', 'wb'))
    p.dump(algorithm.indicator_time_history, open(f'{algorithm.path_results}/indicator_time.p', 'wb'))
    p.dump(algorithm.evaluated_time_history, open(f'{algorithm.path_results}/evaluated_time.p', 'wb'))
    p.dump(algorithm.executed_time_algorithm_history, open(f'{algorithm.path_results}/executed_time.p', 'wb'))
    p.dump(algorithm.running_time_history, open(f'{algorithm.path_results}/running_time.p', 'wb'))

    p.dump([algorithm.nEvals_history, algorithm.E_Archive_search_history],
           open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_search.p', 'wb'))
    p.dump([algorithm.nEvals_history, algorithm.E_Archive_evaluate_history],
           open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_evaluate.p', 'wb'))

    save_reference_point(reference_point=algorithm.reference_point_search, path_results=algorithm.path_results,
                         error='search')
    save_reference_point(reference_point=algorithm.reference_point_evaluate, path_results=algorithm.path_results,
                         error='evaluate')

    visualize_Elitist_Archive_and_Pareto_Front(elitist_archive=algorithm.E_Archive_evaluate_history[-1][-1],
                                               pareto_front=algorithm.problem.pareto_front_testing,
                                               objective_0=algorithm.problem.objective_0,
                                               path_results=algorithm.path_results, error='evaluate')

    visualize_IGD_value_and_nEvals(IGD_history=algorithm.IGD_evaluate_history, nEvals_history=algorithm.nEvals_history,
                                   path_results=algorithm.path_results, error='evaluate')

    visualize_runningtime_and_nEvals(running_time_history=algorithm.running_time_history,
                                     path_results=algorithm.path_results)

def do_each_gen_(**kwargs):
    algorithm = kwargs['algorithm']

    algorithm.nEvals_evaluatedtime_IGD_each_gen.append(
        [algorithm.nEvals, algorithm.evaluated_time_history[-1], algorithm.IGD_evaluate_history[-1]])
    algorithm.nEvals_evaluatedtime_IGD_C100_each_gen.append(
        [algorithm.nEvals, algorithm.evaluated_time_history[-1], algorithm.IGD_evaluate_C100_history[-1]])
    algorithm.nEvals_evaluatedtime_IGD_IN16_each_gen.append(
        [algorithm.nEvals, algorithm.evaluated_time_history[-1], algorithm.IGD_evaluate_IN16_history[-1]])

    algorithm.E_Archive_evaluate_each_gen.append([algorithm.E_Archive_evaluate.X.copy(),
                                                  algorithm.E_Archive_evaluate.hashKey.copy(),
                                                  algorithm.E_Archive_evaluate.F.copy()])

    algorithm.E_Archive_evaluate_C100_each_gen.append([algorithm.E_Archive_evaluate_C100.X.copy(),
                                                       algorithm.E_Archive_evaluate_C100.hashKey.copy(),
                                                       algorithm.E_Archive_evaluate_C100.F.copy()])

    algorithm.E_Archive_evaluate_IN16_each_gen.append([algorithm.E_Archive_evaluate_IN16.X.copy(),
                                                       algorithm.E_Archive_evaluate_IN16.hashKey.copy(),
                                                       algorithm.E_Archive_evaluate_IN16.F.copy()])


def finalize_(**kwargs):
    algorithm = kwargs['algorithm']

    p.dump(algorithm.nEvals_evaluatedtime_IGD_each_gen,
           open(f'{algorithm.path_results}/#Evals_evaluatedtime_IGD_evaluate_each_gen.p', 'wb'))
    p.dump(algorithm.nEvals_evaluatedtime_IGD_C100_each_gen,
           open(f'{algorithm.path_results}/#Evals_evaluatedtime_IGD_evaluate_C100_each_gen.p', 'wb'))
    p.dump(algorithm.nEvals_evaluatedtime_IGD_IN16_each_gen,
           open(f'{algorithm.path_results}/#Evals_evaluatedtime_IGD_evaluate_IN16_each_gen.p', 'wb'))

    p.dump(algorithm.E_Archive_evaluate_each_gen, open(f'{algorithm.path_results}/E_Archive_evaluate_each_gen.p', 'wb'))
    p.dump(algorithm.E_Archive_evaluate_C100_each_gen, open(f'{algorithm.path_results}/E_Archive_evaluate_C100_each_gen.p', 'wb'))
    p.dump(algorithm.E_Archive_evaluate_IN16_each_gen, open(f'{algorithm.path_results}/E_Archive_evaluate_IN16_each_gen.p', 'wb'))

    p.dump([algorithm.nEvals_history, algorithm.IGD_evaluate_history],
           open(f'{algorithm.path_results}/#Evals_and_IGD_evaluate.p', 'wb'))
    p.dump([algorithm.nEvals_history, algorithm.IGD_evaluate_C100_history],
           open(f'{algorithm.path_results}/#Evals_and_IGD_evaluate_C100.p', 'wb'))
    p.dump([algorithm.nEvals_history, algorithm.IGD_evaluate_IN16_history],
           open(f'{algorithm.path_results}/#Evals_and_IGD_evaluate_IN16.p', 'wb'))

    algorithm.benchmark_time_algorithm_history = np.array(algorithm.benchmark_time_algorithm_history)[1:]
    algorithm.indicator_time_history = np.array(algorithm.indicator_time_history)[1:]
    algorithm.evaluated_time_history = np.array(algorithm.evaluated_time_history)[1:]
    algorithm.executed_time_algorithm_history = np.array(algorithm.executed_time_algorithm_history)[1:]
    algorithm.running_time_history = np.array(algorithm.running_time_history)[1:]
    p.dump(algorithm.benchmark_time_algorithm_history, open(f'{algorithm.path_results}/benchmark_time.p', 'wb'))
    p.dump(algorithm.indicator_time_history, open(f'{algorithm.path_results}/indicator_time.p', 'wb'))
    p.dump(algorithm.evaluated_time_history, open(f'{algorithm.path_results}/evaluated_time.p', 'wb'))
    p.dump(algorithm.executed_time_algorithm_history, open(f'{algorithm.path_results}/executed_time.p', 'wb'))
    p.dump(algorithm.running_time_history, open(f'{algorithm.path_results}/running_time.p', 'wb'))

    p.dump([algorithm.nEvals_history, algorithm.E_Archive_search_history],
           open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_search.p', 'wb'))
    p.dump([algorithm.nEvals_history, algorithm.E_Archive_evaluate_history],
           open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_evaluate.p', 'wb'))
    p.dump([algorithm.nEvals_history, algorithm.E_Archive_evaluate_C100_history],
           open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_evaluate_C100.p', 'wb'))
    p.dump([algorithm.nEvals_history, algorithm.E_Archive_evaluate_IN16_history],
           open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_evaluate_IN16.p', 'wb'))

    save_reference_point(reference_point=algorithm.reference_point_evaluate, path_results=algorithm.path_results,
                         error='evaluate')
    save_reference_point(reference_point=algorithm.reference_point_evaluate_C100, path_results=algorithm.path_results,
                         error='evaluate_C100')
    save_reference_point(reference_point=algorithm.reference_point_evaluate_IN16, path_results=algorithm.path_results,
                         error='evaluate_IN16')

    visualize_Elitist_Archive_and_Pareto_Front(elitist_archive=algorithm.E_Archive_evaluate_history[-1][-1],
                                               pareto_front=algorithm.problem.pareto_front_testing_C10,
                                               objective_0=algorithm.problem.objective_0,
                                               path_results=algorithm.path_results, error='evaluate')

    visualize_Elitist_Archive_and_Pareto_Front(elitist_archive=algorithm.E_Archive_evaluate_C100_history[-1][-1],
                                               pareto_front=algorithm.problem.pareto_front_testing_C100,
                                               objective_0=algorithm.problem.objective_0,
                                               path_results=algorithm.path_results, error='evaluate_C100')

    visualize_Elitist_Archive_and_Pareto_Front(elitist_archive=algorithm.E_Archive_evaluate_IN16_history[-1][-1],
                                               pareto_front=algorithm.problem.pareto_front_testing_IN16,
                                               objective_0=algorithm.problem.objective_0,
                                               path_results=algorithm.path_results, error='evaluate_IN16')

    visualize_IGD_value_and_nEvals(IGD_history=algorithm.IGD_evaluate_history, nEvals_history=algorithm.nEvals_history,
                                   path_results=algorithm.path_results, error='evaluate')

    visualize_IGD_value_and_nEvals(IGD_history=algorithm.IGD_evaluate_C100_history, nEvals_history=algorithm.nEvals_history,
                                   path_results=algorithm.path_results, error='evaluate_C100')

    visualize_IGD_value_and_nEvals(IGD_history=algorithm.IGD_evaluate_IN16_history, nEvals_history=algorithm.nEvals_history,
                                   path_results=algorithm.path_results, error='evaluate_IN16')

    visualize_runningtime_and_nEvals(running_time_history=algorithm.running_time_history,
                                     path_results=algorithm.path_results)