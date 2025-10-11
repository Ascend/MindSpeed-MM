import argparse
import os
import subprocess
import shlex
from pathlib import Path


TEST_RESULT_SUCCESS = 0
TEST_RESULT_FAILURE = 1
TEST_RESULT_INVALID_INPUT = 2


def read_files_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def is_examples(file):
    return file.startswith("examples/") and not file.endswith(".py")


def is_markdown(file):
    return file.endswith(".md")


def is_txt(file):
    return file.endswith(".txt")


def is_image(file):
    return file.endswith(".jpg") or file.endswith(".png")


def is_vedio(file):
    return file.endswith(".gif")


def is_owners(file):
    return file.startswith("OWNERS")


def is_license(file):
    return file.startswith("LICENSE")


def is_no_suffix(file):
    return os.path.splitext(file)[1] == ''


def skip_ci_file(files, skip_cond):
    for file in files:
        if not any(condition(file) for condition in skip_cond):
            return False
    return True


def alter_skip_ci():
    parent_dir = Path(__file__).absolute().parents[2]
    raw_txt_file = os.path.join(parent_dir, "modify.txt")

    if not os.path.exists(raw_txt_file):
        return False

    file_list = read_files_from_txt(raw_txt_file)
    skip_conds = [
        is_examples,
        is_markdown,
        is_txt,
        is_image,
        is_vedio,
        is_owners,
        is_license,
        is_no_suffix
    ]

    return skip_ci_file(file_list, skip_conds)


def acquire_exitcode(command):
    """不使用 shell 的更安全版本"""
    args = shlex.split(command)
    process = subprocess.Popen(
        args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 将stderr合并到stdout
        universal_newlines=True,   # 文本模式
        bufsize=1                 # 行缓冲
    )
    
    # 实时读取并输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end='', flush=True)
    
    # 等待进程结束
    return process.wait()


# =============================
# UT test, run with pytest
# =============================

class UT_Test:

    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')
        self.ut_file = os.path.join(test_dir, "ut")

    def run_ut(self):
        command = f"pytest -x {self.ut_file}"
        code = acquire_exitcode(command)
        if code == 0:
            print("UT test success")
        else:
            print("UT failed")
        return code


# ===============================================
# ST test, run with sh.
# ===============================================

class ST_Test:

    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')

        st_dir = "st"
        self.st_shell = os.path.join(
            test_dir, st_dir, "st_run.sh"
        )
        self.local_st_shell = os.path.join(
            test_dir, st_dir, 'local_st_run.sh'
        )

    def run_st(self, local=False):
        if local:
            command = f'bash {self.local_st_shell}'
        else:
            command = f"bash {self.st_shell}"
        code = acquire_exitcode(command)

        if code == 0:
            print("ST test success")
        else:
            print("ST failed")
        return code


def run_ut_tests():
    ut = UT_Test()
    return ut.run_ut()


def run_st_tests():
    st = ST_Test()
    return st.run_st()


def run_st_local_tests():
    st = ST_Test()
    return st.run_st(local=True)


def run_tests(options):
    if options.type == "st":
        st_code = run_st_tests()
        return TEST_RESULT_FAILURE if st_code != 0 else TEST_RESULT_SUCCESS
    elif options.type == "ut":
        ut_code = run_ut_tests()
        return TEST_RESULT_FAILURE if ut_code != 0 else TEST_RESULT_SUCCESS
    elif options.type == "all":
        code = run_ut_tests()
        if code != 0:
            return TEST_RESULT_FAILURE
        st_code = run_st_tests()
        return TEST_RESULT_FAILURE if st_code != 0 else TEST_RESULT_SUCCESS
    elif options.type == 'all_loss':
        code = run_ut_tests()
        if code != 0:
            return TEST_RESULT_FAILURE
        st_code = run_st_local_tests()
        return TEST_RESULT_FAILURE if st_code != 0 else TEST_RESULT_SUCCESS
    else:
        print(f"TEST CASE TYPE ERROR: no type '{options.type}'")
        return TEST_RESULT_INVALID_INPUT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control needed test cases")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "ut", "st", "codecheck", "all_loss"],
                        help='Test cases type. `all`: run all test cases; `ut`: run ut case, `st`: run st cases; `codecheck`: used for codecheck; `all_loss`: used for local ci')
    args = parser.parse_args()
    print(f"options: {args}")
    if alter_skip_ci():
        print("Skipping CI: Success")
    elif args.type == "codecheck":
        print("Skipping CI: Failed")
        exit(1) # codecheck阶段不执行ut/st，直接返回异常值
    else:
        print("Skipping CI: Failed")
        exit_code = run_tests(args)
        if exit_code != 0:
            exit(exit_code)