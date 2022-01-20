import numpy as np
import itertools

def find_the_better(F_x, F_y, position=-1):
    """
    This function is used to find the better solution between two input solutions.\n
    If one of the solutions is an extreme solution, just using only one objective which
    that solution is the best at for comparing.
    """
    if isinstance(F_x, list):
        F_x = np.array(F_x)
    if isinstance(F_y, list):
        F_y = np.array(F_y)
    if position == -1:
        if isinstance(F_x[-1], dict):
            vote_lst = []
            for key in F_x[-1]:
                x_new = np.array([F_x[0], F_x[-1][key]])
                y_new = np.array([F_y[0], F_y[-1][key]])
                sub_ = x_new - y_new
                x_better = np.all(sub_ <= 0)
                y_better = np.all(sub_ >= 0)
                if x_better == y_better:  # True - True
                    vote_lst.append(-1)
                elif y_better:  # False - True
                    vote_lst.append(1)
                else:
                    vote_lst.append(0)  # True - False
            count_vote_lst = [vote_lst.count(-1), vote_lst.count(0), vote_lst.count(1)]
            better_lst = np.array([-1, 0, 1])
            # if count_vote_lst[0] == count_vote_lst[1] == count_vote_lst[2] == 1:
            if count_vote_lst[0] == 1 or count_vote_lst[1] == 1 or count_vote_lst[2] == 1:
                return None
            idx = np.argmax(count_vote_lst)
            return better_lst[idx]
        else:
            sub_ = F_x - F_y
            x_better = np.all(sub_ <= 0)
            y_better = np.all(sub_ >= 0)
            if x_better == y_better:  # True - True
                return -1
            if y_better:  # False - True
                return 1
            return 0  # True - False
    else:
        if position == 0:
            if F_x[position] < F_y[position]:
                return 0
            elif F_x[position] > F_y[position]:
                return 1
            else:
                if F_x[1] < F_y[1]:
                    return 0
                elif F_x[1] > F_y[1]:
                    return 1
                return -1
        else:
            if isinstance(F_x[-1], dict):
                vote_lst = []
                for key in F_x[-1]:
                    x_new = F_x[-1][key]
                    y_new = F_y[-1][key]
                    if x_new < y_new:
                        vote_lst.append(0)
                    elif x_new > y_new:
                        vote_lst.append(1)
                    else:
                        vote_lst.append(-1)
                count_vote_lst = [vote_lst.count(-1), vote_lst.count(0), vote_lst.count(1)]
                better_lst = np.array([-1, 0, 1])
                # if count_vote_lst[0] == count_vote_lst[1] == count_vote_lst[2] == 1:
                if count_vote_lst[0] == 1 or count_vote_lst[1] == 1 or count_vote_lst[2] == 1:
                    return None
                idx = np.argmax(count_vote_lst)
                return better_lst[idx]
            else:
                if F_x[1] < F_y[1]:
                    return 0
                elif F_x[1] > F_y[1]:
                    return 1
                else:
                    return -1


def get_idx_non_dominated_front(F):
    """
    This function is used to get the zero front in the population.
    """
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    idx = np.array(list(range(l)))
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:  # the current is better
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    idx_non_dominated_front = idx[r == 0]
    return idx_non_dominated_front


def check_above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calculate_angle_measure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)


def find_all_neighbors(X, distance=1, **kwargs):
    # TODO: Speech Recognition (distance = 2)
    """
    This function is used to find all neighboring solutions of the considering solution. The distance from the current
    solution to the neighbors is 1 (or 2).
    """
    all_neighbors = []
    idx = list(itertools.combinations(range(len(X)), distance))
    if kwargs['problem'].name == 'NASBench101':
        for i in idx:
            if i[0] == 0 or i[0] == 21:
                continue
            elif i[0] in kwargs['problem'].IDX_OPS:
                available_ops = kwargs['problem'].OPS.copy()
            else:
                available_ops = kwargs['problem'].EDGES.copy()

            for op in available_ops:
                if op != X[i]:
                    neighbor = X.copy()
                    neighbor[i] = op
                    all_neighbors.append(neighbor)
    else:
        if kwargs['problem'].name != 'SpeechRecognition':
            available_ops = kwargs['problem'].available_ops.copy()
            if distance == 1:
                for i in idx:
                    for op in available_ops:
                        if op != X[i[0]]:
                            neighbor = X.copy()
                            neighbor[i[0]] = op
                            all_neighbors.append(neighbor)
            elif distance == 2:
                for i, j in idx:
                    for op_0 in available_ops:
                        if op_0 != X[i]:
                            neighbor = X.copy()
                            neighbor[i] = op_0
                            for op_1 in available_ops:
                                if op_1 != X[j]:
                                    neighbor[j] = op_1
                                    all_neighbors.append(neighbor)
            else:
                for i0, i1, i2 in idx:
                    for op_0 in available_ops:
                        if op_0 != X[i0]:
                            neighbor = X.copy()
                            neighbor[i0] = op_0
                            for op_1 in available_ops:
                                if op_1 != X[i1]:
                                    neighbor[i1] = op_1
                                    for op_2 in available_ops:
                                        if op_2 != X[i2]:
                                            neighbor[i2] = op_2
                                            all_neighbors.append(neighbor)

        else:
            for i in range(len(X)):
                if i in kwargs['problem'].IDX_MAIN_OPS:
                    valid_ops = kwargs['problem'].MAIN_OPS.copy()
                else:
                    valid_ops = kwargs['problem'].SKIP_OPS.copy()
                for op in valid_ops:
                    if op != X[i]:
                        neighbor = X.copy()
                        neighbor[i] = op
                        all_neighbors.append(neighbor)
    return all_neighbors


def check_valid(hash_key, **kwargs):
    """
    This function is used to check if the current solution is valid or not.
    """
    return np.all([hash_key not in kwargs[L] for L in kwargs])
