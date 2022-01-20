import time
import numpy as np
import pickle as p
from problems.direct.NAS_problem import Problem
from zero_cost_methods import modify_input_for_fitting

def get_key_in_data(arch):
    """
    Get the key which is used to represent the architecture in "self.data".
    """
    return ''.join(map(str, arch))


class NASBench201(Problem):
    def __init__(self, dataset, maxEvals, **kwargs):
        """
        # NAS-Benchmark-201 provides us the information (e.g., the training loss, the testing accuracy,
        the validation accuracy, the number of FLOPs, etc) of all architectures in the search space. Therefore, if we
        want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        # Additional Hyper-parameters:\n
        - path_data -> the path contains NAS-Bench-201 data.
        - data -> NAS-Bench-201 data.
        - min_max -> the maximum and minimum of architecture's metrics. They are used to normalize the results.
        - pareto_front_testing -> the Pareto-optimal front in the search space (nFLOPs - testing error)
        - available_ops -> the available operators can choose in the search space.
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(maxEvals, 'NASBench201', dataset, **kwargs)

        self.objective_0 = 'nFLOPs'
        self.objective_1 = 'val_error'

        self.epoch = kwargs['epoch']

        ''' ------- Additional Hyper-parameters ------- '''
        self.available_ops = [0, 1, 2, 3, 4]
        self.maxLength = 6

        self.path_data = kwargs['path_data'] + '/NASBench201'
        self.data = None
        self.min_max = None

        self.pareto_front_testing = None
        self.zc_predictor = None

    def _set_up(self):
        available_datasets = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
        if self.dataset not in available_datasets:
            raise ValueError(f'Just only supported these datasets: CIFAR-10; CIFAR-100; ImageNet16-120.'
                             f'{self.dataset} dataset is not supported at this time.')

        f_data = open(f'{self.path_data}/[{self.dataset}]_data.p', 'rb')
        self.data = p.load(f_data)
        f_data.close()

        f_min_max = open(f'{self.path_data}/[{self.dataset}]_min_max.p', 'rb')
        self.min_max = p.load(f_min_max)
        f_min_max.close()

        f_pareto_front_testing = open(f'{self.path_data}/[{self.dataset}]_pareto_front(testing).p', 'rb')
        self.pareto_front_testing = p.load(f_pareto_front_testing)
        f_pareto_front_testing.close()

        print('--> Set Up - Done')

    def set_zero_cost_predictor(self, zc_predictor):
        self.zc_predictor = zc_predictor

    def _get_a_compact_architecture(self):
        return np.random.choice(self.available_ops, self.maxLength)

    def get_cost_time(self, arch, final=False):
        key = get_key_in_data(arch)
        if final:
            cost_time = self.data['200'][key]['train_time'] + self.data['200'][key]['val_time']
            # cost_time = self.data[key]['train_time']['200'] + self.data[key]['val_time']['200']
        else:
            if self.epoch not in ['12', '200']:
                raise ValueError()
            cost_time = self.data[self.epoch][key]['train_time'] + self.data[self.epoch][key]['val_time']
            # cost_time = self.data[key]['train_time']['12'] + self.data[key]['val_time']['12']
        return cost_time

    def _get_performance_metric(self, arch, **kwargs):
        """
        - Get the performance of architecture. E.g., the testing error, the validation error.
        """
        try:
            final = kwargs['final']
        except KeyError:
            final = False
        try:
            using_zc = kwargs['using_zc']
        except KeyError:
            using_zc = False

        key = get_key_in_data(arch)
        benchmark_time = 0.0
        indicator_time = 0.0
        if final:
            perf_metric = 1 - self.data['200'][key]['test_acc']
            # perf_metric = round(1.0 - self.data[key]['test_acc']/100, 4)
        else:
            if using_zc:
                s = time.time()
                X_modified = modify_input_for_fitting(arch, self.name)
                score = self.zc_predictor.query_(arch_mod=X_modified, arch_ori=arch)
                perf_metric = -score
                indicator_time = time.time() - s
            else:
                perf_metric = 1 - self.data[self.epoch][key]['val_acc']
                # perf_metric = round(1.0 - self.data[key]['val_acc'][self.epoch - 1]/100, 4)
                benchmark_time = self.get_cost_time(arch)
        return perf_metric, benchmark_time, indicator_time

    def _get_computational_metric(self, arch):
        """
        - In NAS-Bench-201 problem, the computational metric is nFLOPs.
        - The returned nFLOPs is normalized.
        """
        key = get_key_in_data(arch)
        nFLOPs = round((self.data['200'][key]['FLOPs'] - self.min_max['FLOPs']['min']) /
                      (self.min_max['FLOPs']['max'] - self.min_max['FLOPs']['min']), 6)
        # nFLOPs = round((self.data[key]['FLOPs'] - self.min_max['FLOPs']['min']) /
        #               (self.min_max['FLOPs']['max'] - self.min_max['FLOPs']['min']), 6)
        return nFLOPs

    def _evaluate(self, arch, using_zc):
        performance_metric, benchmark_time, indicator_time = self.get_performance_metric(arch=arch, using_zc=using_zc)
        computational_metric = self.get_computational_metric(arch)
        return computational_metric, performance_metric, benchmark_time, indicator_time

    def _isValid(self, arch):
        return True


if __name__ == '__main__':
    pass
