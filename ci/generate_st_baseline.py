import argparse
import os
from pathlib import Path
from tests.st.test_tools.acquire_json import transfer_logs_as_json


class GenerateBaseLine:
    def __init__(self, args):
        self.input_path = args.input_shell_path
        self.out_path = args.baseline_outpath
        self.tmp_log_path = os.path.join(self.out_path, 'tmp_log')
        Path(self.tmp_log_path).mkdir(parents=True, exist_ok=True)
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.out_path, 'efficient')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.out_path, 'loss')).mkdir(parents=True, exist_ok=True)

    def acquire_exitcode(self, command):
        exitcode = os.system(command)
        real_code = os.WEXITSTATUS(exitcode)
        return real_code

    def generate_base_line(self):
        efficient_shell_path = os.path.join(self.input_path, 'shell_scripts')
        loss_shell_path = os.path.join(self.input_path, 'loss_shell_scripts')

        config_copy_cmd = f"cp -r /home/ci_resource/local_st/run_configs /home/MindSpeed-MM/tests/st/"
        _ = self.acquire_exitcode(config_copy_cmd)

        rm_cmd = f"rm -rf /home/MindSpeed-MM/tests/st/shell_scripts"
        cp_cmd = f"cp -r {efficient_shell_path} /home/MindSpeed-MM/tests/st/shell_scripts"
        _ = self.acquire_exitcode(rm_cmd)
        _ = self.acquire_exitcode(cp_cmd)
        for shell in os.listdir(efficient_shell_path):
            file_name_prefix = shell.split('.')[0]
            command = (f'cd /home/MindSpeed-MM && '
                       f'bash /home/MindSpeed-MM/tests/st/shell_scripts/{shell} '
                       f'| tee {self.tmp_log_path}/{file_name_prefix}.log')
            exitcode = self.acquire_exitcode(command)

            if exitcode == 0:
                transfer_logs_as_json(f'{self.tmp_log_path}/{file_name_prefix}.log',
                                      f'{self.out_path}/efficient/{file_name_prefix}.json')

        rm_cmd = f"rm -rf /home/MindSpeed-MM/tests/st/loss_shell_scripts"
        cp_cmd = f"cp -r {loss_shell_path} /home/MindSpeed-MM/tests/st/loss_shell_scripts"
        _ = self.acquire_exitcode(rm_cmd)
        _ = self.acquire_exitcode(cp_cmd)
        for shell in os.listdir(loss_shell_path):
            file_name_prefix = shell.split('.')[0]
            command = (f'cd /home/MindSpeed-MM && '
                       f'bash /home/MindSpeed-MM/tests/st/loss_shell_scripts/{shell} '
                       f'| tee {self.tmp_log_path}/{file_name_prefix}.log')
            exitcode = self.acquire_exitcode(command)

            if exitcode == 0:
                transfer_logs_as_json(f'{self.tmp_log_path}/{file_name_prefix}.log',
                                      f'{self.out_path}/loss/{file_name_prefix}_loss.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate st baseline")
    parser.add_argument("baseline_outpath", type=str,
                        default='/home/ci_resource/local_st/local_st_baseline',
                        help='Path to save st baseline')
    parser.add_argument("--input_shell_path", type=str,
                        default='/home/ci_resource/local_st/update',
                        help='Config path')
    args = parser.parse_args()

    g = GenerateBaseLine(args)
    g.generate_base_line()
