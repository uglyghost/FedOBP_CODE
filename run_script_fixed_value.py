# Comparison of Importance Scores for FedFew (改为手动配置 dataset-param 映射)

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
                decoded_line = line.decode('utf-8')
            except UnicodeDecodeError:
                decoded_line = line.decode('utf-8', errors='ignore')
            print(decoded_line, end='')  # 实时输出到控制台
            f.write(decoded_line)
        process.wait()
        if process.returncode != 0:
            print(f"命令失败，查看日志文件: {log_file}")
        else:
            print(f"命令成功完成，日志文件: {log_file}")


def main():
    # 手动配置：每个 dataset 对应一个 ig_ratio（或其他参数）
    #### alpha=0.1
    # dataset_ig_map = {'medmnistC': 0.9997, 'medmnistA': 0.9999, } # 'fmnist': 0.99993, 'svhn': 0.99997, 'mnist': 0.9998, 'cifar100': 0.9998, 'cifar10': 0.9999, 'emnist': 0.9995,

    # #### alpha=0.5
    # dataset_ig_map = {'fmnist': 0.9999, 'svhn': 0.9998, 'mnist': 0.99993, 'cifar100': 0.99992,
    #                   'cifar10': 0.99979, 'emnist': 0.9991, 'medmnistC': 0.99995, 'medmnistA': 0.99995, }

    dataset_ig_map = {'medmnistC': 0.99995, 'medmnistA': 0.99995, }

    method_list = ['fedobp',]


    # 创建日志目录
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    # 遍历 dataset-param 对
    for method in method_list:
        # 构建命令
        for dataset, ig_ratio in dataset_ig_map.items():
            command = [
                sys.executable,
                'main_fedobp.py',
                f'method={method}',
                f'dataset.name={dataset}',
                f'{method}.ig_ratio={ig_ratio}',
            ]
            # 构建日志路径
            log_filename = f"fedobp_{dataset}_{ig_ratio}.log"
            log_path = log_dir / log_filename
            print(f"运行命令: {' '.join(command)}")
            run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()
