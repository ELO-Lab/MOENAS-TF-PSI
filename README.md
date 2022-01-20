# Enhancing Multi-Objective Evolutionary Neural Architecture Search with Training-Free Pareto Local Search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong

In Applied Intelligence 2021.
## Setup
- Clone this repo
- Install dependencies:
```
$ pip install -r requirements.txt
```
- Although we utilize NAS-Bench-101 and NAS-Bench-201 for our experiments, we do not utilize directly their APIs to for the purpose of reduce the time and memory capability (fact: our experiments are conducted on a personal laptop). We thus downloaded the database of NAS-Bench-101 and NAS-Bench-201, modify their APIs, and using on our way. Therefore, to reproduce the results in the article, we have to download the data at first.
- Download data in [here](https://drive.google.com/drive/folders/1oNk21qWKs_8hmBBkM1ye0zAV-08LABaE?usp=sharing) and put into [*data*] folder
## Usage
### Search
```shell
python main.py --problem <problem_name>
               --algorithm <algorithm_name> --pop_size <population_size>
               --PSI <using_PSI_method> --PSI_nPoints <k-opt> --PSI_using_zc <using_Training-free-PSI_method>
               --debug <debug_mode> --seed <random_seed> --n_runs <the_number_of_experiment_runs>
               --path_results <path_for_saving_experiment_results>

```
|Hyperparameter           |Help                                    |Default          |Choices                                          |                
|:------------------------|:---------------------------------------|:----------------|:------------------------------------------------|
|`--problem`              |the problem                             |`NAS201-C10`     |`NAS101` `NAS201-C10` `NAS201-C100` `NAS201-IN16`|
|`--algorithm`            |the algorithm                           |`NSGA-II`        |`NSGA-II`                                        |

To reproduce results in our experiments

- To search with **the vanilla NSGA-II (baseline)**:
```
python main.py --problem <problem_name> --PSI 0 --PSI_nPoints 0 --PSI_using_zc 0
```
- To search with **NSGA-II with Potential Solutions Improving (PSI) k = 1**:
```
python main.py --problem <problem_name> --PSI 1 --PSI_nPoints 1 --PSI_using_zc 0
```
- To search with **NSGA-II with Potential Solutions Improving (PSI) k = 2**:
```
python main.py --problem <problem_name> --PSI 1 --PSI_nPoints 2 --PSI_using_zc 0
```
- To search with **NSGA-II with training-free PSI (TF-PSI)**:
```
python main.py --problem <problem_name> --PSI 1 --PSI_nPoints 1 --PSI_using_zc 1
```

In our experiments, we set the `population_size` is equal to `20` for all problems. To experiment with the different `population_size`, set the value of `--pop_size <population_size>`.

To experiment with the different `maximum_number_of_evaluations`, set in [*factory.py*]

### Evaluate & Visualize
```shell
python evaluate.py  --path_results <path_results>
```
For example:
```shell
python evaluate.py  --path_results .\results\NAS201-C10
```
### Kolmogorov–Smirnov (KS) test
```shell
python ks_test.py  --path_results <path_results>
```
For example:
```shell
python ks_test.py  --path_results .\results\NAS201-C10
```
***Note:*** `<path_results>` ***must only contains results of experiments are conducted on the same problem.***
## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [How Powerful are Performance Predictors in Neural Architecture Search?](https://github.com/automl/NASLib)
