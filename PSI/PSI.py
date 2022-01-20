import numpy as np

from .utils import (
    find_all_neighbors,
    find_the_better,
    get_idx_non_dominated_front,
    check_above_or_below,
    calculate_angle_measure,
    check_valid
)
from utils import get_hashKey


class PSI_processor:
    def __init__(self, gamma=210, nPoints=1, **kwargs):
        self.nPoints = nPoints
        self.gamma = gamma

        self.using_zc = kwargs['using_zc']
        if not self.using_zc:
            self.method_name = 'Potential Solutions Improving'
        else:
            self.method_name = 'Training-free Potential Solutions Improving'
        self.neighbors_history = []

    def reset(self):
        self.neighbors_history = []

    def do(self, pop, **kwargs):
        non_dominated_set, potential_sols, idx_non_dominated_front = self.seeking(pop, **kwargs)
        if self.using_zc:
            pop = self.improving_with_zc_predictor(pop, non_dominated_set, potential_sols, idx_non_dominated_front, **kwargs)
        else:
            pop = self.improving(pop, non_dominated_set, potential_sols, idx_non_dominated_front, **kwargs)
        return pop

    ''' --------------------------------------------------------------------------------------- '''
    def seeking(self, P, **kwargs):
        """
        The first phase in the method: \n Identifying potential solutions
        """
        P_F = P.get('F')

        idx_non_dominated_front = get_idx_non_dominated_front(P_F)

        non_dominated_set = P[idx_non_dominated_front].copy()

        non_dominated_front = P_F[idx_non_dominated_front].copy()

        new_idx = np.argsort(non_dominated_front[:, 0])

        non_dominated_set = non_dominated_set[new_idx]
        non_dominated_front = non_dominated_front[new_idx]

        non_dominated_front_norm = non_dominated_front.copy()

        min_f0 = np.min(non_dominated_front[:, 0])
        max_f0 = np.max(non_dominated_front[:, 0])

        min_f1 = np.min(non_dominated_front[:, 1])
        max_f1 = np.max(non_dominated_front[:, 1])

        non_dominated_front_norm[:, 0] = (non_dominated_front_norm[:, 0] - min_f0) / (max_f0 - min_f0)
        non_dominated_front_norm[:, 1] = (non_dominated_front_norm[:, 1] - min_f1) / (max_f1 - min_f1)

        idx_non_dominated_front = idx_non_dominated_front[new_idx]

        potential_sols = [
            [0, 'best_f0']  # (idx (in pareto front), property)
        ]

        for i in range(len(non_dominated_front) - 1):
            if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i + 1])) != 0:
                break
            else:
                potential_sols.append([i, 'best_f0'])

        for i in range(len(non_dominated_front) - 1, -1, -1):
            if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i - 1])) != 0:
                break
            else:
                potential_sols.append([i - 1, 'best_f1'])
        potential_sols.append([len(non_dominated_front) - 1, 'best_f1'])

        # find the knee solutions
        start_idx = potential_sols[0]
        end_idx = potential_sols[-1][0]

        for sols in potential_sols:
            if sols[-1] == 'best_f1':
                break
            else:
                start_idx = sols[0] + 1

        for i in range(len(potential_sols) - 1, -1, -1):
            if potential_sols[i][1] == 'best_f0':
                break
            else:
                end_idx = potential_sols[i][0] - 1

        for i in range(start_idx, end_idx + 1):
            l = None
            h = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, len(non_dominated_front), 1):
                if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                    h = m
                    break

            if (h is not None) and (l is not None):
                position = check_above_or_below(considering_pt=non_dominated_front[i],
                                                remaining_pt_1=non_dominated_front[l],
                                                remaining_pt_2=non_dominated_front[h])
                if position == -1:
                    angle_measure = calculate_angle_measure(considering_pt=non_dominated_front_norm[i],
                                                            neighbor_1=non_dominated_front_norm[l],
                                                            neighbor_2=non_dominated_front_norm[h])
                    if angle_measure > self.gamma:
                        potential_sols.append([i, 'knee'])
        return non_dominated_set, potential_sols, idx_non_dominated_front

    ''' --------------------------------------------------------------------------------------- '''
    def improving(self, P, non_dominated_set, potential_sols, idx_non_dominated_front, **kwargs):
        """
        The second phase in the method: \n Improving potential solutions
        """
        problem_name = kwargs['problem'].name

        _non_dominated_set = P.new()
        P_hashKey = P.get('hashKey')

        l_sol = len(P[0].X)

        for i, property_sol in potential_sols:
            nSearchs, maxSearchs = 0, l_sol
            _nSearchs, _maxSearchs = 0, 100

            found_neighbors_list = [non_dominated_set[i].hashKey]
            while (nSearchs < maxSearchs) and (_nSearchs < _maxSearchs):
                _nSearchs += 1

                """ Find a neighboring solution """
                neighborSol = non_dominated_set[i].copy()

                if problem_name == 'NASBench101':
                    neighborSol_X = neighborSol.X.copy()
                    idxs_lst = []
                    for _ in range(self.nPoints):
                        while True:
                            idx = np.random.randint(22)
                            if idx == 0 or idx == 21:
                                pass
                            else:
                                idxs_lst.append(idx)
                                break
                    for idx in idxs_lst:
                        if idx in kwargs['problem'].IDX_OPS:
                            available_ops = kwargs['problem'].OPS.copy()
                        else:
                            available_ops = kwargs['problem'].EDGES.copy()
                        available_ops.remove(neighborSol_X[idx])
                        neighborSol_X[idx] = np.random.choice(available_ops)
                else:
                    idxs = np.random.choice(range(kwargs['problem'].maxLength), size=self.nPoints, replace=False)
                    neighborSol_X = non_dominated_set[i].X.copy()
                    for idx in idxs:
                        allowed_ops = kwargs['problem'].available_ops.copy()
                        allowed_ops.remove(non_dominated_set[i].X[idx])
                        new_op = np.random.choice(allowed_ops)
                        neighborSol_X[idx] = new_op

                if kwargs['problem'].isValid(neighborSol_X):
                    neighborSol_hashKey = get_hashKey(neighborSol_X, problem_name)

                    DS = []
                    # DS = kwargs['algorithm'].E_Archive_search.DS
                    if check_valid(neighborSol_hashKey,
                                   neighbors=found_neighbors_list,
                                   P=P_hashKey, DS=DS):
                        """ Improve the neighboring solution """
                        found_neighbors_list.append(neighborSol_hashKey)
                        nSearchs += 1

                        neighborSol_F = kwargs['algorithm'].evaluate(neighborSol_X)
                        neighborSol.set('X', neighborSol_X)
                        neighborSol.set('hashKey', neighborSol_hashKey)
                        neighborSol.set('F', neighborSol_F)
                        kwargs['algorithm'].E_Archive_search.update(neighborSol, algorithm=kwargs['algorithm'])
                        if property_sol == 'best_f0':
                            betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=0)
                        elif property_sol == 'best_f1':
                            betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=1)
                        else:
                            betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F)

                        if betterSol == 0:  # --> the neighbor is better
                            # tmp = P.new(1)
                            # tmp[0].set('X', non_dominated_set[i].X)
                            # tmp[0].set('hashKey', non_dominated_set[i].hashKey)
                            # tmp[0].set('F', non_dominated_set[i].F)
                            # _non_dominated_set = _non_dominated_set.merge(tmp)

                            non_dominated_set[i].set('X', neighborSol_X)
                            non_dominated_set[i].set('hashKey', neighborSol_hashKey)
                            non_dominated_set[i].set('F', neighborSol_F)

                        # else:  # --> no one is better || the current is better
                        #     tmp_pop = P.new(1)
                        #     tmp_pop[0].set('X', neighborSol_X)
                        #     tmp_pop[0].set('hashKey', neighborSol_hashKey)
                        #     tmp_pop[0].set('F', neighborSol_F)
                        #     _non_dominated_set = _non_dominated_set.merge(tmp_pop)

        P[idx_non_dominated_front] = non_dominated_set
        # pool = P.merge(_non_dominated_set)
        # P = kwargs['algorithm'].survival.do(pool, len(P))
        return P
    '''---------------------------------------------------------------------------------------'''
    def improving_with_zc_predictor(self, P, non_dominated_set, potential_sols, idx_non_dominated_front, **kwargs):
        """
        The second phase in the method: \n Improving potential solutions
        """
        problem_name = kwargs['problem'].name

        _non_dominated_set = P.new()
        P_hashKey = P.get('hashKey')

        for i, property_sol in potential_sols:
            if non_dominated_set[i].hashKey not in self.neighbors_history:
                potentialSol_computational_metric = kwargs['problem'].get_computational_metric(non_dominated_set[i].X)
                potentialSol_tf_ind, _, _ = kwargs['problem'].get_performance_metric(non_dominated_set[i].X, using_zc=True)

                potentialSol_F_proxy = [potentialSol_computational_metric, potentialSol_tf_ind]

                self.neighbors_history.append(non_dominated_set[i].hashKey)

                betterSol_X_lst = []
                betterSol_F_lst = []
                nonDominatedSol_X_lst = []
                nonDominatedSol_F_lst = []
                all_neighbors = find_all_neighbors(non_dominated_set[i].X, distance=self.nPoints, **kwargs)
                for neighbor_X in all_neighbors:
                    if kwargs['problem'].isValid(neighbor_X):
                        neighborSol_hashKey = get_hashKey(neighbor_X, problem_name)
                        if check_valid(neighborSol_hashKey, P=P_hashKey):
                            neighborSol_computational_metric = kwargs['problem'].get_computational_metric(neighbor_X)
                            neighborSol_tf_ind, _, _ = kwargs['problem'].get_performance_metric(neighbor_X, using_zc=True)

                            neighborSol_F_proxy = [neighborSol_computational_metric, neighborSol_tf_ind]

                            if property_sol == 'best_f0':
                                betterSol = find_the_better(neighborSol_F_proxy, potentialSol_F_proxy, position=0)
                            elif property_sol == 'best_f1':
                                betterSol = find_the_better(neighborSol_F_proxy, potentialSol_F_proxy, position=1)
                            else:
                                betterSol = find_the_better(neighborSol_F_proxy, potentialSol_F_proxy)

                            if betterSol == 0:
                                betterSol_X_lst.append(neighbor_X)
                                betterSol_F_lst.append(neighborSol_F_proxy)
                            elif betterSol == -1:
                                nonDominatedSol_X_lst.append(neighbor_X)
                                nonDominatedSol_F_lst.append(neighborSol_F_proxy)

                idx_bestSol1 = get_idx_non_dominated_front(betterSol_F_lst)
                idx_bestSol2 = get_idx_non_dominated_front(nonDominatedSol_F_lst)

                for idx in idx_bestSol1:
                    neighborSol_hashKey = get_hashKey(betterSol_X_lst[idx], problem_name)
                    tmp_pop = P.new(1)
                    neighborSol_F = kwargs['algorithm'].evaluate(betterSol_X_lst[idx])

                    tmp_pop[0].set('X', betterSol_X_lst[idx])
                    tmp_pop[0].set('hashKey', neighborSol_hashKey)
                    tmp_pop[0].set('F', neighborSol_F)
                    kwargs['algorithm'].E_Archive_search.update(tmp_pop[0], algorithm=kwargs['algorithm'])

                    if property_sol == 'best_f0':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=0)
                    elif property_sol == 'best_f1':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=1)
                    else:
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F)

                    if betterSol == 0:  # --> the neighbor is better
                        tmp = P.new(1)
                        tmp[0].set('X', non_dominated_set[i].X)
                        tmp[0].set('hashKey', non_dominated_set[i].hashKey)
                        tmp[0].set('F', non_dominated_set[i].F)
                        _non_dominated_set = _non_dominated_set.merge(tmp)

                        non_dominated_set[i].set('X', betterSol_X_lst[idx])
                        non_dominated_set[i].set('hashKey', neighborSol_hashKey)
                        non_dominated_set[i].set('F', neighborSol_F)
                    else:
                        tmp_pop = P.new(1)
                        tmp_pop[0].set('X', betterSol_X_lst[idx])
                        tmp_pop[0].set('hashKey', neighborSol_hashKey)
                        tmp_pop[0].set('F', neighborSol_F)
                        _non_dominated_set = _non_dominated_set.merge(tmp_pop)

                for idx in idx_bestSol2:
                    neighborSol_hashKey = get_hashKey(nonDominatedSol_X_lst[idx], problem_name)
                    tmp_pop = P.new(1)
                    neighborSol_F = kwargs['algorithm'].evaluate(nonDominatedSol_X_lst[idx])
                    tmp_pop[0].set('X', nonDominatedSol_X_lst[idx])
                    tmp_pop[0].set('hashKey', neighborSol_hashKey)
                    tmp_pop[0].set('F', neighborSol_F)
                    kwargs['algorithm'].E_Archive_search.update(tmp_pop[0], algorithm=kwargs['algorithm'])

                    if property_sol == 'best_f0':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=0)
                    elif property_sol == 'best_f1':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=1)
                    else:
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F)

                    if betterSol == 0:  # --> the neighbor is better
                        tmp = P.new(1)
                        tmp[0].set('X', non_dominated_set[i].X)
                        tmp[0].set('hashKey', non_dominated_set[i].hashKey)
                        tmp[0].set('F', non_dominated_set[i].F)
                        _non_dominated_set = _non_dominated_set.merge(tmp)

                        non_dominated_set[i].set('X', nonDominatedSol_X_lst[idx])
                        non_dominated_set[i].set('hashKey', neighborSol_hashKey)
                        non_dominated_set[i].set('F', neighborSol_F)
                    else:
                        tmp_pop = P.new(1)
                        tmp_pop[0].set('X', nonDominatedSol_X_lst[idx])
                        tmp_pop[0].set('hashKey', neighborSol_hashKey)
                        tmp_pop[0].set('F', neighborSol_F)
                        _non_dominated_set = _non_dominated_set.merge(tmp_pop)

        P[idx_non_dominated_front] = non_dominated_set
        pool = P.merge(_non_dominated_set)
        P = kwargs['algorithm'].survival.do(pool, len(P))
        return P