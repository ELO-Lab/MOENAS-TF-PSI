import numpy as np

def find_the_better(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(x[-1], dict):
        vote_lst = []
        for key in x[-1]:
            x_new = np.array([x[0], x[-1][key]])
            y_new = np.array([y[0], y[-1][key]])
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
        sub_ = x - y
        x_better = np.all(sub_ <= 0)
        y_better = np.all(sub_ >= 0)
        if x_better == y_better:  # True - True
            return -1
        if y_better:  # False - True
            return 1
        return 0  # True - False