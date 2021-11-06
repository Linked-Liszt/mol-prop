import time
import argparse
from dataclasses import dataclass
import subprocess
import os
import shlex
from typing import Any, TextIO
from datetime import datetime

PYTHON_PATH = '/home/mhp0009/anaconda3/envs/molnlp/bin/python '
CURR_CWD = os.getcwd()

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    parser.add_argument('queue_file', type=str, help='path to queue file')
    parser.add_argument('--gpu', type=int, help='number of GPUs', default=3)


@dataclass
class ExpProc():
    proc: Any
    log_object: TextIO
    experiment_name: str
    start_time: datetime


def check_proc(proc_data) -> bool:
    return proc_data.poll() is None


def notify_completion(proc_data: ExpProc) -> None:
    notification_message = (
        f'Experiment Completed\n'
        + f'Experiment Name: {proc_data.experiment_name}\n'
        + f'Start Time: \n' # TODO
        + f'Time Δ: \n') # TODO


def main():
    args = parse_args()
    gpu_status = [False for _ in range(args.gpu)]
    proc_data = [ExpProc(None, '', '', -1) for _ in range(args.gpu)]


    while True:
        time.sleep(30)
        for i, status in enumerate(gpu_status):
            curr_status = status
            if curr_status:
               if not check_proc(proc_data[i]):
                   notify_completion(proc_data[i])
                   proc_data[i].log_object.close()
                   gpu_status[i] = False

            if not curr_status and get_queue_length(args.queue_file) > 0:
                proc_data[i] = spawn_proc(pop_queue_item(args.queue_file))


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

    return lines[i]


def get_queue_length(queue_file: str) -> int:
    with open(queue_file, 'r') as q_f:
        lines = q_f.readlines()
    return len(lines)


def spawn_proc(command: str) -> ExpProc:
    experiment_name = command.split(' ')[1]
    log_f = open(f'logs/{experiment_name}.log', 'a+')

    cmd = PYTHON_PATH + command
    proc = subprocess.Popen(shlex.split(cmd), cwd=CURR_CWD, stdout=log_f, stderr=subprocess.STDOUT)

    return ExpProc(proc, log_f, experiment_name, datetime.now())



if __name__ == '__main__':
    main()