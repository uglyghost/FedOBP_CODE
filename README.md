# FedOBP

Official repository for FedOBP

This repository is based on the [FL-bench](https://github.com/KarhouTam/FL-bench.git) implementation.

## Installation
```sh
pip install -r .env/requirements.txt
```

### Step 1. Generate FL Dataset
Partition the MNIST according to Dir(0.1) for 100 clients
```shell
python generate_data.py -d mnist -a 0.1 -cn 100
```
About methods of generating federated dastaset, go check [`data/README.md`](data/#readme) for full details.

### Step 2. Run FedOBP Main Experiment

```sh
python main_fedobp.py [--config-path, --config-name] [dataset.name=<DATASET_NAME> args...]
```

### Step 4. FedOBP Ablation Experiment

```sh
python run_script_ablation.py
```

### Step 3. Run Baselines Experiment

```sh
python main.py [--config-path, --config-name] [method=<METHOD_NAME> args...]
```

## Monitor runs
This implementation supports `tensorboard`.
1. Run `tensorboard --logdir=<your_log_dir>` on terminal.
2. Go check `localhost:6006` on your browser.
