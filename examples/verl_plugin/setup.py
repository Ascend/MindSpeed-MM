import os
import sys
import sysconfig
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


# 插件注入逻辑
def inject_verl_plugin(custom_path=None):
    """将NPU加速支持注入到verl包中"""
    print("Starting verl plugin injection...")
    
    # 优先级：环境变量 > 自定义路径 > 自动查找
    if 'VERL_PATH' in os.environ:
        verl_path = os.path.join(os.environ['VERL_PATH'], "verl")
        print(f"Using verl path from environment variable: {verl_path}")
    elif custom_path:
        verl_path = custom_path
        print(f"Using custom verl path: {verl_path}")
    else:
        print("Searching for verl package automatically...")
        # 尝试多种方式查找verl安装路径
        paths_to_try = [
            sysconfig.get_paths()["purelib"],
            sysconfig.get_paths()["platlib"],
        ] + sys.path  # 搜索所有Python路径
        
        verl_path = None
        for path in paths_to_try:
            if not path:
                continue
                
            candidate = os.path.join(path, "verl")
            if os.path.exists(candidate) and os.path.isdir(candidate):
                verl_path = candidate
                break
        
        # 使用pip show作为备用方案
        if not verl_path:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", "verl"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.splitlines():
                    if line.startswith("Location:"):
                        verl_path = os.path.join(line.split(": ")[1], "verl")
                        break
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"pip show failed: {e}")
    
    if not verl_path:
        print("Error: verl package not found. Please specify with VERL_PATH environment variable.")
        return False
    
    print(f"Found verl at: {verl_path}")
    
    # 1. 修改 __init__.py 文件
    init_file = os.path.join(verl_path, "__init__.py")
    if not os.path.exists(init_file):
        print(f"Error: verl initialization file not found: {init_file}")
        return False
    
    # 检查是否已经注入过
    import_content = """
# NPU acceleration support added by mindspeed-mm plugin
from verl.utils.device import is_npu_available

if is_npu_available:
    import verl_npu
    print("NPU acceleration enabled for verl")
"""
    
    # 读取当前内容
    try:
        with open(init_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {init_file}: {e}")
        return False
    
    if import_content in content:
        print(f"Info: {init_file} already contains NPU acceleration import")
    else:
        # 添加注入内容
        try:
            with open(init_file, "a") as f:
                f.write(import_content)
            print(f"Successfully modified {init_file} to add NPU acceleration support")
        except Exception as e:
            print(f"Error writing to {init_file}: {e}")
            return False

    return True


# vllm ascend patch
def inject_vllm_plugin():
    print("Searching for vllm ascend package automatically...")
    # 尝试多种方式查找vllm安装路径
    paths_to_try = [
        sysconfig.get_paths()["purelib"],
        sysconfig.get_paths()["platlib"],
    ] + sys.path  # 搜索所有Python路径

    vllm_path = None
    for path in paths_to_try:
        if not path:
            continue

        candidate = os.path.join(path, "vllm_ascend")
        if os.path.exists(candidate) and os.path.isdir(candidate):
            vllm_path = candidate
            break

    # 使用pip show作为备用方案
    if not vllm_path:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "vllm_ascend"],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.splitlines():
                if line.startswith("Editable project location:"):
                    vllm_path = os.path.join(line.split(": ")[1], "vllm_ascend")
                    break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"pip show failed: {e}")

    if not vllm_path:
        print("Error: vllm_ascend package not found. Please specify with VLLM_PATH environment variable.")
        return False

    print(f"Found vllm_ascend at: {vllm_path}")

     # 2. 修改 rotary_embedding.py 文件
    rotary_embedding_file = os.path.join(vllm_path, "ops", "rotary_embedding.py")
    if not os.path.exists(rotary_embedding_file):
        print(f"Warning: rotary_embedding file not found: {rotary_embedding_file}")
        return True

    # 需要修改的行
    line_to_change = "query, key = torch_npu.npu_mrope(positions,"
    line_change_to = "    query, key = torch_npu.npu_mrope(positions.contiguous(),\n"

    try:
        with open(rotary_embedding_file, "r") as f:
            lines = f.readlines()

        modified = False
        new_lines = []
        for line in lines:
            # 检查是否是需要注释的行（并且尚未被注释）
            if line.strip() == line_to_change:
                new_lines.append(line_change_to)  # 修改
                print(f"Changed out line in {rotary_embedding_file}: {line.strip()}")
                modified = True
            else:
                new_lines.append(line)

        if modified:
            # 写回修改后的内容
            with open(rotary_embedding_file, "w") as f:
                f.writelines(new_lines)
            print(f"Successfully modified {rotary_embedding_file}")
        else:
            # 检查是否已经被注释
            already_changed = any(line_change_to in line for line in lines)
            if already_changed:
                print(f"Info: line already changed in {rotary_embedding_file}")
            else:
                print(f"Warning: line to change not found in {rotary_embedding_file}: {line_to_change}")

    except Exception as e:
        print(f"Error modifying {rotary_embedding_file}: {e}")
        return False
    return True


# 自定义安装命令
class CustomInstallCommand(install):
    """自定义安装命令"""
    def run(self):
        super().run()
        print("Running verl injection after standard install...")
        # 尝试从环境变量获取路径
        custom_path = os.environ.get('VERL_PATH', None)
        if not inject_verl_plugin(custom_path):
            print("Error: verl injection failed. Please check installation.")
        if not inject_vllm_plugin():
            print("Error: vllm injection failed. Please check installation.")


# 自定义开发模式安装命令
class CustomDevelopCommand(develop):
    """自定义开发模式安装命令"""
    def run(self):
        super().run()
        print("Running verl injection after develop install...")
        # 尝试从环境变量获取路径
        custom_path = os.environ.get('VERL_PATH', None)
        if not inject_verl_plugin(custom_path):
            print("Error: verl injection failed. Please check installation.")
        if not inject_vllm_plugin():
            print("Error: vllm injection failed. Please check installation.")


# 主安装函数
def main():
    print("Setting up verl_npu plugin...")

    setup(
        name="verl_npu",
        version="0.0.1",
        license="Apache 2.0",
        description="verl npu backend plugin",
        packages=find_packages(include=["verl_npu"]),
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
        ],
        python_requires=">=3.9",
        cmdclass={
            'install': CustomInstallCommand,
            'develop': CustomDevelopCommand,
        },
    )


if __name__ == '__main__':
    main()