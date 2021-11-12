import time
import argparse
from dataclasses import dataclass
import subprocess
import os
import shlex
from typing import Any, TextIO
from datetime import datetime
import psutil
import GPUtil

PYTHON_PATH = '/home/mhp0009/anaconda3/envs/molnlp/bin/python '
CURR_CWD = os.getcwd()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('queue_file', type=str, help='path to queue file')
    parser.add_argument('--gpu', type=int, help='number of GPUs', default=3)
    return parser.parse_args()


@dataclass
class ExpProc():
    proc: Any
    log_object: TextIO
    experiment_name: str
    start_time: datetime


def check_proc(proc_data) -> bool:
    return proc_data.proc.poll() is None


def notify_completion(proc_data: ExpProc, system_status) -> None:
    notification_message = (
        f'Experiment Completed\n'
        + f'Experiment Name: {proc_data.experiment_name}\n'
        + f'Start Time: {proc_data.start_time}\n' # TODO
        + f'Time Δ: {datetime.now() - proc_data.start_time}\n'
        + f'Exit Time: {datetime.now()}\n' # TODO
        + f'Exit Code: {proc_data.proc.poll()}\n'
        + f'\nSystem Status:\n'
        + system_status)

    with open('runscripts/dsc_msg.txt', 'w') as dsc_f:
        dsc_f.write(notification_message)
        dsc_f.truncate()

    with open('runscripts/schd_log.log', 'a+') as log_f:
        log_f.write('\n\n' + notification_message)

    subprocess.run(shlex.split(f'python discord_notify.py runscripts/dsc_msg.txt results/{proc_data.experiment_name}.log'))
    #print(notification_message)


def main():
    with open('schd_pid.txt', 'w') as pid_f:
        pid_f.write(f'{os.getpid()}')
    args = parse_args()
    gpu_status = [False for _ in range(args.gpu)]
    proc_data = [ExpProc(None, '', '', -1) for _ in range(args.gpu)]

    last_sys_status = get_system_status()

    while True:
        for i, status in enumerate(gpu_status):
            curr_status = status
            if curr_status:
               if not check_proc(proc_data[i]):
                   notify_completion(proc_data[i], last_sys_status)
                   proc_data[i].log_object.close()
                   gpu_status[i] = False

            if not curr_status and get_queue_length(args.queue_file) > 0:
                proc_data[i] = spawn_proc(pop_queue_item(args.queue_file), i)
                gpu_status[i] = True
        last_sys_status = get_system_status()
        time.sleep(30)


def pop_queue_item(queue_file: str) -> str:
    with open(queue_file, 'r+') as q_f:
        lines = q_f.readlines()

        i = 0
        for i, line in enumerate(lines):
            if len(line.split(' ')) > 1:
                break

        q_f.seek(0)
        if i < (len(lines) - 1):
            q_f.write(''.join(lines[i+1:]))
        q_f.truncate()

    return lines[i].strip()


def get_queue_length(queue_file: str) -> int:
    with open(queue_file, 'r') as q_f:
        lines = q_f.readlines()
    return len(lines)


def spawn_proc(command: str, gpu_id: int) -> ExpProc:
    experiment_name = command.split(' ')[1]
    log_f = open(f'logs/{experiment_name.strip()}.log', 'a+')

    cmd = PYTHON_PATH + command
    cmd += f' --gpu {gpu_id}'
    proc = subprocess.Popen(shlex.split(cmd), cwd=CURR_CWD, stdout=log_f, stderr=subprocess.STDOUT)

    return ExpProc(proc, log_f, experiment_name, datetime.now())

def get_system_status() -> str:
    system_status = ""
    system_status += f'CPU Util: {psutil.cpu_percent()}\n'
    mem = psutil.virtual_memory()
    system_status += f'Mem: {mem.percent}% ({mem.used / 1e9 :.1f} / {mem.total/1e9 :.1f} GB)\n'
    disk = psutil.disk_usage('/')
    system_status += f'Disk: {disk.percent}% ({disk.used / 1e9 :.1f} / {disk.total/1e9 :.1f} GB)\n'
    for i, gpu in enumerate(GPUtil.getGPUs()):
        system_status += f'GPU: {i} Utilization: {gpu.load} | Mem: {float(gpu.memoryUsed) / 1e3 :.1f} / {float(gpu.memoryTotal/1e3) :.1f} GB)\n'

    return system_status


if __name__ == '__main__':
    main()