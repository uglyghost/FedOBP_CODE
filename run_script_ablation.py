# Ablation Experiment for FedOBP

import subprocess
import sys
from pathlib import Path


def run_command(command, log_file):
    """
    运行命令并将输出写入日志文件。
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            try:
                # 尝试用 UTF-8 解码
                decoded_line = line.decode('utf-8')
            except UnicodeDecodeError:
                # 如果解码失败，可以选择忽略或使用其他编码
                decoded_line = line.decode('utf-8', errors='ignore')
            print(decoded_line, end='')  # 实时输出到控制台
            f.write(decoded_line)
        process.wait()
        if process.returncode != 0:
            print(f"命令失败，查看日志文件: {log_file}")
        else:
            print(f"命令成功完成，日志文件: {log_file}")


def main():
    # 定义参数
    # datasets_ig_dict = {'cifar10': 0.983, 'cifar100': 0.91078}  # 'cifar10': 0.9999, 'cifar100': 0.99505, 'emnist': 0.995, 'svhn': 0.9999,
    # 'medmnistA': 0.83, 'medmnistC': 0.99995, 'mnist': 0.9995, 'fmnist': 0.999705  'emnist': 0.9057,(cls)
    datasets_ig_dict = {'medmnistA': 0.9484}
    methods = ['fedobp']
    norms = ['global']  # , 'layer', 'other'
    EMA = ['true', 'false']
    CLS = ['true']
    experiment_type = 'Ablation'

    alpha = 0.5

    # 创建一个目录来保存所有日志
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    # 遍历所有组合
    for method in methods:
        for dataset, ig_ratio in datasets_ig_dict.items():
            for norm in norms:
                if norm == 'global':
                    for ema in EMA:
                        for cls in CLS:
                            command = [
                                sys.executable,
                                'main_ablation.py',
                                f'method={method}',
                                f'dataset.name={dataset}',
                                f'{method}.type={experiment_type}',
                                f'{method}.ig_ratio={ig_ratio}',
                                f'{method}.alpha={alpha}',
                                f'{method}.norm={norm}',
                                f'{method}.EMA={ema}',
                                f'{method}.CLS={cls}',
                            ]
                            # 构建日志文件名（可根据需要自定义）
                            log_filename = f"{dataset}_{method}_{norm}_{ema}.log"
                            log_path = log_dir / log_filename

                            print(f"运行命令: {' '.join(command)}")
                            run_command(command, log_path)
                else:
                    command = [
                        sys.executable,
                        'main_ablation.py',
                        f'method={method}',
                        f'dataset.name={dataset}',
                        f'{method}.type={experiment_type}',
                        f'{method}.ig_ratio={ig_ratio}',
                        f'{method}.alpha={alpha}',
                        f'{method}.norm={norm}',
                        f'{method}.EMA=false',
                        # f'{method}.CLS={cls}',
                    ]
                    # 构建日志文件名（可根据需要自定义）
                    log_filename = f"{dataset}_{method}_{norm}_false.log"
                    log_path = log_dir / log_filename

                    print(f"运行命令: {' '.join(command)}")
                    run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()
