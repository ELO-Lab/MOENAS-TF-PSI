import pickle as p
import os
import numpy as np
import argparse

from scipy.stats import ks_2samp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    args = parser.parse_args()

    alpha = 0.01
    print('Alpha:', alpha)
    path_results = args.path_results
    checkpoints_list = list(range(100, 3001, 100))
    checkpoints_list_ = ['last' for _ in range(len(checkpoints_list))]

    for folder in os.listdir(path_results):
        if folder in ['nEvals_IGD', 'nEvals_Hypervolume']:
            avg_igd_lst = []
            original = []
            original_performance = []
            variants = []
            variants_performance = []
            for experiment in os.listdir(path_results + '/' + folder):
                configuration = experiment.split('_')
                if configuration[1] == 'False' and configuration[3] == 'False':
                    print(configuration)
                    print('original')
                    nEvals_and_Performance_metric = p.load(open(f'{path_results}/{folder}/{experiment}', 'rb'))
                    nEvals_history = nEvals_and_Performance_metric[0]
                    performance_history_testing = nEvals_and_Performance_metric[1]

                    nEvals_mean = np.round(np.mean(nEvals_history, axis=0, dtype=int), 4)
                    nRuns = len(performance_history_testing)
                    for nEvals in checkpoints_list_:
                        if nEvals == 'last':
                            idx = len(nEvals_mean) - 1
                        else:
                            idx_1 = np.where(nEvals == nEvals_mean)[0]
                            if len(idx_1) == 0:
                                idx_2 = np.where(nEvals < nEvals_mean)[0]
                                idx = idx_2[0] - 1
                            else:
                                idx = idx_1[-1]
                        performance = performance_history_testing[:, [idx]].reshape(nRuns)
                        original.append(performance)
                        original_performance.append(f'{np.round(np.mean(original[-1]), 4)} '
                                                        f'({np.round(np.std(original[-1]), 4)})')
                    # for performance in original_performance:
                    #     print(f'${performance}$')
            for experiment in os.listdir(path_results + '/' + folder):
                configuration = experiment.split('_')
                if configuration[1] == 'False' and configuration[3] == 'False':
                    pass
                else:
                    print()
                    print(configuration)
                    nEvals_and_Performance_metric = p.load(open(f'{path_results}/{folder}/{experiment}', 'rb'))
                    nEvals_history = nEvals_and_Performance_metric[0]
                    performance_history_testing = nEvals_and_Performance_metric[1]

                    nEvals_mean = np.round(np.mean(nEvals_history, axis=0, dtype=int), 4)
                    nRuns = len(performance_history_testing)

                    for i, nEvals in enumerate(checkpoints_list):
                        if nEvals == 'last':
                            idx = len(nEvals_mean) - 1
                        else:
                            idx_1 = np.where(nEvals == nEvals_mean)[0]
                            if len(idx_1) == 0:
                                idx_2 = np.where(nEvals < nEvals_mean)[0]
                                idx = idx_2[0] - 1
                            else:
                                idx = idx_1[-1]
                        performance = performance_history_testing[:, [idx]].reshape(nRuns)
                        variants.append(performance)

                        p_value = ks_2samp(original[i], variants[-1])[-1]
                        if np.isnan(p_value):
                            p_value = 1
                        if p_value > alpha:
                            variants_performance.append(f'{np.round(np.mean(variants[-1]), 4)} '
                                                        f'({np.round(np.std(variants[-1]), 4)})')
                            # print(nEvals, '| Accept |')
                        else:
                            mean_1 = np.mean(original[i])
                            mean_2 = np.mean(variants[-1])
                            std_1 = np.std(original[i])
                            std_2 = np.std(variants[-1])
                            cohen_d = (abs(mean_1 - mean_2)) / ((std_1**2 + std_2**2) / 2)**(1/2)
                            if cohen_d >= 0.8:
                                effect_size = 'large'
                            elif cohen_d >= 0.5:
                                effect_size = 'medium'
                            elif cohen_d >= 0.2:
                                effect_size = 'small'
                            else:
                                effect_size = 'trivial'
                            if configuration[-1].split('.')[0] == 'IGD':
                                if np.mean(original[i]) < np.mean(variants[-1]):
                                    compare = 'worse'
                                else:
                                    compare = 'better'
                            else:
                                if np.mean(original[i]) > np.mean(variants[-1]):
                                    compare = 'worse'
                                else:
                                    compare = 'better'
                            variants_performance.append(f'{np.round(np.mean(variants[-1]), 4)} '
                                                            f'({np.round(np.std(variants[-1]), 4)})')
                            print(nEvals, '| Reject |', compare, effect_size)
                    # for performance in variants_performance:
                    #     print(f'${performance}$')
                    variants, variants_performance = [], []
            print('-='*40)
