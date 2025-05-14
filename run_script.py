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
    datasets_name = ['cifar10', 'cifar100', 'emnist', 'svhn',
                     'mnist', 'fmnist', 'medmnistC', 'medmnistA']
    # datasets_name = ['cifar100']
    # methods = ['fedavg', 'local']  # baselines
    # methods = ['fedper', 'apfl', 'lgfedavg', 'fedrep', 'pfedla', 'feddpa', 'flute', 'floco']  SOTA solutions
    methods = ['fedobp']

    # for logging
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    # 遍历每个方法和数据集的组合
    for method in methods:
        for dataset in datasets_name:
            # 构建命令
            command = [
                sys.executable,
                'main.py',
                f'method={method}',
                f'dataset.name={dataset}'
            ]

            # logging
            log_filename = f"{method}_{dataset}.log"
            log_path = log_dir / log_filename

            print(f"运行命令: {' '.join(command)}")
            run_command(command, log_path)

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()

