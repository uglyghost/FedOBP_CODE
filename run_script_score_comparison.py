# Comparison of Importance Scores for FedOBP

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
    datasets_name = ['cifar10', 'cifar100', 'svhn']  # emnist, 'medmnistA', 'medmnistC'
    datasets_name = ['svhn']  # emnist, 'medmnistA', 'medmnistC'
    # datasets_name = ['medmnistA', 'medmnistC']  # emnist, 'medmnistA', 'medmnistC'
    # datasets_name = ['emnist', 'cifar10', 'cifar100', 'svhn', 'fmnist', 'mnist']  # emnist, 'medmnistA', 'medmnistC'
    # datasets_name = ['cifar10', 'cifar100']
    # ig_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    #              0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998,
    #              0.999, 0.9991, 0.9993, 0.9995, 0.9997,
    #              0.9999, 0.99991, 0.99993, 0.99995, 0.99997,
    #              0.99999, 0.999999, 1.0]
    # ig_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
    #              0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
    #              0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999]
    # ig_values = [0.9992, 0.99931, 0.99932, 0.99933, 0.99934, 0.99935, 0.99936, 0.99937, 0.99938, 0.99939,
    #              0.9994, 0.99941, 0.99942, 0.99943, 0.99944, 0.99945, 0.99946, 0.99947, 0.99948, 0.99949,
    #              0.99951, 0.99952, 0.99953, 0.99954, 0.99955, 0.99956, 0.99957, 0.99958, 0.99959, 0.9996, 0.9998]
    ig_values = [0.9999]
    #              0.999991, 0.999992, 0.999993, 0.999994, 0.999995, 0.999996, 0.999997, 0.999998, 0.999999]
    # ig_values = [0.9995]
    # methods = ['feddpag', 'fedobp', 'feddpa']
    methods = ['fedobp']
    alpha = [0.5]

    # 创建一个目录来保存所有日志
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    # 遍历所有组合
    for method in methods:
        for ig_ratio in ig_values:
            for alpha_tmp in alpha:
                for dataset in datasets_name:
                    # 构建命令
                    if method == 'fedobp':
                        command = [
                            sys.executable,
                            'main_fedobp.py',
                            f'method={method}',
                            f'dataset.name={dataset}',
                            f'{method}.ig_ratio={ig_ratio}',
                            f'{method}.alpha={alpha_tmp}',
                        ]
                    else:
                        command = [
                            sys.executable,
                            'main_fedobp.py',
                            f'method={method}',
                            f'dataset.name={dataset}',
                            f'{method}.fisher_threshold={ig_ratio}',
                        ]

                    # 构建日志文件名
                    log_filename = f"{datasets_name}.log"
                    log_path = log_dir / log_filename

                    print(f"运行命令: {' '.join(command)}")
                    run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()
