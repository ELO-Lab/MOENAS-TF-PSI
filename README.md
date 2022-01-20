# ...
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong
<!-- In Applied_Intelligence 2021. -->
## Setup
- Clone this repo:
```
$ git clone https://github.com/f4nku4n/Applied_Intelligence
$ cd Applied_Intelligence
```
- Install dependencies:
```
$ pip install -r requirements.txt
```
- Download data in [here](https://drive.google.com/drive/folders/1oNk21qWKs_8hmBBkM1ye0zAV-08LABaE?usp=sharing) and put into [*data*](https://github.com/f4nku4n/Applied_Intelligence/tree/master/data) folder
## Usage
### Search
```shell
python main.py --problem <problem_name>
               --algorithm <algorithm_name> --pop_size <population_size>
               --IPS <using_IPS_method> --IPS_nPoints <k-opt> --IPS_using_zc <using_IPS_method_with_tf_indicator>
               --warm_up <using_Warmup_method> --nSamples_for_warm_up <the_number_of_samples_for_warmup>
               --debug <debug_mode> --seed <random_seed> --n_runs <the_number_of_experiment_runs>
               --path_results <path_for_saving_experiment_results>

```
|Hyperparameter           |Help                                    |Default          |Choices                                          |                
|:------------------------|:---------------------------------------|:----------------|:------------------------------------------------|
|`--problem`              |the problem                             |`NAS201-C10`     |`NAS101` `NAS201-C10` `NAS201-C100` `NAS201-IN16`|
|                         |                                        |                 |`NAS201-C100-T` `NAS201-IN16-T`                  | 
|`--algorithm`            |the algorithm                           |`NSGA-II`        |`NSGA-II`                                        |
<!-- |`--pop_size`             |the population size                     |`0`              |                                                 |
|`--IPS`                  |using IPS method?                       |`0`              |`0`: no; `1`: yes                                |
|`--IPS_nPoints`          |`k`-opt value                           |`1`              |                                                 |
|`--IPS_using_zc`         |using training-free IPS method?         |`0`              |`0`: no; `1`: yes                                |
|`--warm_up`              |using Warmup method?                    |`0`              |`0`: no; `1`: yes                                |
|`--nSamples_for_warm_up` |the number of samples for warmup        |`500`            |                                                 |
|`--seed`                 |random seed                             |`0`              |                                                 |
|`--n_runs`               |the number of experiment runs           |`31`             |                                                 |
|`--path_results`         |the path for saving experiment results  |`None`           |                                                 | -->

To reproduce results in our experiments

- To search with **the original MOEA (baseline)**:
```
python main.py --problem <problem_name> --IPS 0 --IPS_using_zc 0 --warm_up 0 --nSamples_for_warm_up 0
```
- To search with **MOEA with Warmup** (in our experiments, we sample `500` architectures for warming up):
```
python main.py --problem <problem_name> --IPS 0 --IPS_using_zc 0 --warm_up 1 --nSamples_for_warm_up 500
```
- To search with **MOEA with Improving Potential Solutions (IPS)**:
```
python main.py --problem <problem_name> --IPS 1 --IPS_using_zc 0 --warm_up 0 --nSamples_for_warm_up 0
```
- To search with **MOEA with both Warmup and IPS**:
```
python main.py --problem <problem_name> --IPS 1 --IPS_using_zc 0 --warm_up 1 --nSamples_for_warm_up 500
```
- To search with **MOEA with training-free IPS**:
```
python main.py --problem <problem_name> --IPS 1 --IPS_using_zc 1 --warm_up 0 --nSamples_for_warm_up 0
```
- To search with **MOEA with both Warmup and training-free IPS**:
```
python main.py --problem <problem_name> --IPS 1 --IPS_using_zc 1 --warm_up 1 --nSamples_for_warm_up 500
```

In our experiments, we set the `population_size` is equal to `20` for all problems. To experiment with the different `population_size`, set the value of `--pop_size <population_size>`.

To experiment with the different `maximum_number_of_evaluations`, set in [factory.py](https://github.com/f4nku4n/Applied_Intelligence/blob/master/factory.py).

### Evaluate & Visualize
- For problems which are utilized for assesing the **generalizability** of algorithms (i.e., `NAS101`, `NAS201-C10`, `NAS201-C100`, `NAS201-IN16`)
```shell
python visualize_direct.py  --path_results <path_results>
```
- For problems which are utilized for assesing the **transferability** of algorithms  (i.e., `NAS201-C100-T`, `NAS201-IN16-T`)
```shell
python visualize_transfer.py  --path_results <path_results>
```

For example: ```python visualize_direct.py  --path_results .\results\NAS201-C100```

***Note:*** `<path_results>` ***must only contains results of experiments are conducted on the same problem.***
<!-- ## Results (in paper)
- Single-objective NAS problems:
![](https://github.com/FanKuan44/NICS/blob/master/figs/SONAS(1).png)

- Multi-objective NAS problems:
![](https://github.com/FanKuan44/NICS/blob/master/figs/MONAS(1).png) -->

## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [How Powerful are Performance Predictors in Neural Architecture Search?](https://github.com/automl/NASLib)
- [Efficiency Enhancement of Evolutionary Neural Architecture Search via Training-Free Initialization](https://github.com/f4nku4n/NICS)
