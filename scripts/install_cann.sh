#!/bin/bash

# install_cann.sh - 安装CANN的脚本
# 用法: ./install_cann.sh ARCH
# 参数: ARCH - CPU架构 (x86|arm)

show_help() {
    cat << EOF
Usage: $0 ARCH

Arguments:
    ARCH    CPU architecture (x86|arm)

Examples:
    $0 x86
    $0 arm
EOF
}

# 函数：检测操作系统类型
detect_os() {
    local os_name=""

    if [ -f /etc/os-release ]; then
        # 读取操作系统信息
        source /etc/os-release
        os_name=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
    fi

    echo "$os_name"
}

# 操作系统安装函数定义
# 函数命名格式: install_cann_{NPU_TYPE}_{ARCH}_{OS_TYPE}

# 910B NPU安装函数
install_cann_910b_x86_ubuntu() {
    echo "Installing CANN for NPU 910B on x86 Ubuntu"
    echo "执行910B + x86 + Ubuntu的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)

    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 Ubuntu installation completed!"
}

install_cann_910b_x86_openeuler() {
    echo "Installing CANN for NPU 910B on x86 openEuler"
    echo "执行910B + x86 + openEuler的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install

    echo "NPU 910B x86 openEuler installation completed!"
}

install_cann_910b_x86_centos() {
    echo "Installing CANN for NPU 910B on x86 CentOS"
    echo "执行910B + x86 + CentOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)

    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 CentOS installation completed!"
}

install_cann_910b_x86_debian() {
    echo "Installing CANN for NPU 910B on x86 Debian"
    echo "执行910B + x86 + Debian的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)

    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 Debian installation completed!"
}

install_cann_910b_x86_kylin() {
    echo "Installing CANN for NPU 910B on x86 Kylin"
    echo "执行910B + x86 + Kylin的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 Kylin installation completed!"
}

install_cann_910b_x86_bclinux() {
    echo "Installing CANN for NPU 910B on x86 BCLinux"
    echo "执行910B + x86 + BCLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 BCLinux installation completed!"
}

install_cann_910b_x86_uosv20() {
    echo "Installing CANN for NPU 910B on x86 UOSV20"
    echo "执行910B + x86 + UOSV20的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 UOSV20 installation completed!"
}

install_cann_910b_x86_antos() {
    echo "Installing CANN for NPU 910B on x86 AntOS"
    echo "执行910B + x86 + AntOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 AntOS installation completed!"
}

install_cann_910b_x86_alios() {
    echo "Installing CANN for NPU 910B on x86 AliOS"
    echo "执行910B + x86 + AliOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 AliOS installation completed!"
}

install_cann_910b_x86_ctyunos() {
    echo "Installing CANN for NPU 910B on x86 CTyunOS"
    echo "执行910B + x86 + CTyunOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 CTyunOS installation completed!"
}

install_cann_910b_x86_culinux() {
    echo "Installing CANN for NPU 910B on x86 CULinux"
    echo "执行910B + x86 + CULinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 CULinux installation completed!"
}

install_cann_910b_x86_tlinux() {
    echo "Installing CANN for NPU 910B on x86 Tlinux"
    echo "执行910B + x86 + Tlinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 Tlinux installation completed!"
}

install_cann_910b_x86_mtos() {
    echo "Installing CANN for NPU 910B on x86 MTOS"
    echo "执行910B + x86 + MTOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 MTOS installation completed!"
}

install_cann_910b_x86_velinux() {
    echo "Installing CANN for NPU 910B on x86 veLinux"
    echo "执行910B + x86 + veLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910B x86 veLinux installation completed!"
}

install_cann_910b_arm_ubuntu() {
    echo "Installing CANN for NPU 910B on ARM Ubuntu"
    echo "执行910B + ARM + Ubuntu的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM Ubuntu installation completed!"
}

install_cann_910b_arm_openeuler() {
    echo "Installing CANN for NPU 910B on ARM openEuler"
    echo "执行910B + ARM + openEuler的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM openEuler installation completed!"
}

install_cann_910b_arm_centos() {
    echo "Installing CANN for NPU 910B on ARM CentOS"
    echo "执行910B + ARM + CentOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM CentOS installation completed!"
}

install_cann_910b_arm_debian() {
    echo "Installing CANN for NPU 910B on ARM Debian"
    echo "执行910B + ARM + Debian的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM Debian installation completed!"
}

install_cann_910b_arm_kylin() {
    echo "Installing CANN for NPU 910B on ARM Kylin"
    echo "执行910B + ARM + Kylin的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM Kylin installation completed!"
}

install_cann_910b_arm_bclinux() {
    echo "Installing CANN for NPU 910B on ARM BCLinux"
    echo "执行910B + ARM + BCLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM BCLinux installation completed!"
}

install_cann_910b_arm_uosv20() {
    echo "Installing CANN for NPU 910B on ARM UOSV20"
    echo "执行910B + ARM + UOSV20的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM UOSV20 installation completed!"
}

install_cann_910b_arm_antos() {
    echo "Installing CANN for NPU 910B on ARM AntOS"
    echo "执行910B + ARM + AntOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM AntOS installation completed!"
}

install_cann_910b_arm_alios() {
    echo "Installing CANN for NPU 910B on ARM AliOS"
    echo "执行910B + ARM + AliOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM AliOS installation completed!"
}

install_cann_910b_arm_ctyunos() {
    echo "Installing CANN for NPU 910B on ARM CTyunOS"
    echo "执行910B + ARM + CTyunOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM CTyunOS installation completed!"
}

install_cann_910b_arm_culinux() {
    echo "Installing CANN for NPU 910B on ARM CULinux"
    echo "执行910B + ARM + CULinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM CULinux installation completed!"
}

install_cann_910b_arm_tlinux() {
    echo "Installing CANN for NPU 910B on ARM Tlinux"
    echo "执行910B + ARM + Tlinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM Tlinux installation completed!"
}

install_cann_910b_arm_mtos() {
    echo "Installing CANN for NPU 910B on ARM MTOS"
    echo "执行910B + ARM + MTOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM MTOS installation completed!"
}

install_cann_910b_arm_velinux() {
    echo "Installing CANN for NPU 910B on ARM veLinux"
    echo "执行910B + ARM + veLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910B ARM veLinux installation completed!"
}

# 910C NPU安装函数
install_cann_910c_x86_ubuntu() {
    echo "Installing CANN for NPU 910C on x86 Ubuntu"
    echo "执行910C + x86 + Ubuntu的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 Ubuntu installation completed!"
}

install_cann_910c_x86_openeuler() {
    echo "Installing CANN for NPU 910C on x86 openEuler"
    echo "执行910C + x86 + openEuler的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 openEuler installation completed!"
}

install_cann_910c_x86_centos() {
    echo "Installing CANN for NPU 910C on x86 CentOS"
    echo "执行910C + x86 + CentOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 CentOS installation completed!"
}

install_cann_910c_x86_debian() {
    echo "Installing CANN for NPU 910C on x86 Debian"
    echo "执行910C + x86 + Debian的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 Debian installation completed!"
}

install_cann_910c_x86_kylin() {
    echo "Installing CANN for NPU 910C on x86 Kylin"
    echo "执行910C + x86 + Kylin的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 Kylin installation completed!"
}

install_cann_910c_x86_bclinux() {
    echo "Installing CANN for NPU 910C on x86 BCLinux"
    echo "执行910C + x86 + BCLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 BCLinux installation completed!"
}

install_cann_910c_x86_uosv20() {
    echo "Installing CANN for NPU 910C on x86 UOSV20"
    echo "执行910C + x86 + UOSV20的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 UOSV20 installation completed!"
}

install_cann_910c_x86_antos() {
    echo "Installing CANN for NPU 910C on x86 AntOS"
    echo "执行910C + x86 + AntOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 AntOS installation completed!"
}

install_cann_910c_x86_alios() {
    echo "Installing CANN for NPU 910C on x86 AliOS"
    echo "执行910C + x86 + AliOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 AliOS installation completed!"
}

install_cann_910c_x86_ctyunos() {
    echo "Installing CANN for NPU 910C on x86 CTyunOS"
    echo "执行910C + x86 + CTyunOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 CTyunOS installation completed!"
}

install_cann_910c_x86_culinux() {
    echo "Installing CANN for NPU 910C on x86 CULinux"
    echo "执行910C + x86 + CULinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 CULinux installation completed!"
}

install_cann_910c_x86_tlinux() {
    echo "Installing CANN for NPU 910C on x86 Tlinux"
    echo "执行910C + x86 + Tlinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 Tlinux installation completed!"
}

install_cann_910c_x86_mtos() {
    echo "Installing CANN for NPU 910C on x86 MTOS"
    echo "执行910C + x86 + MTOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 MTOS installation completed!"
}

install_cann_910c_x86_velinux() {
    echo "Installing CANN for NPU 910C on x86 veLinux"
    echo "执行910C + x86 + veLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
    echo "NPU 910C x86 veLinux installation completed!"
}

install_cann_910c_arm_ubuntu() {
    echo "Installing CANN for NPU 910C on ARM Ubuntu"
    echo "执行910C + ARM + Ubuntu的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM Ubuntu installation completed!"
}

install_cann_910c_arm_openeuler() {
    echo "Installing CANN for NPU 910C on ARM openEuler"
    echo "执行910C + ARM + openEuler的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM openEuler installation completed!"
}

install_cann_910c_arm_centos() {
    echo "Installing CANN for NPU 910C on ARM CentOS"
    echo "执行910C + ARM + CentOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM CentOS installation completed!"
}

install_cann_910c_arm_debian() {
    echo "Installing CANN for NPU 910C on ARM Debian"
    echo "执行910C + ARM + Debian的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM Debian installation completed!"
}

install_cann_910c_arm_kylin() {
    echo "Installing CANN for NPU 910C on ARM Kylin"
    echo "执行910C + ARM + Kylin的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM Kylin installation completed!"
}

install_cann_910c_arm_bclinux() {
    echo "Installing CANN for NPU 910C on ARM BCLinux"
    echo "执行910C + ARM + BCLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM BCLinux installation completed!"
}

install_cann_910c_arm_uosv20() {
    echo "Installing CANN for NPU 910C on ARM UOSV20"
    echo "执行910C + ARM + UOSV20的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM UOSV20 installation completed!"
}

install_cann_910c_arm_antos() {
    echo "Installing CANN for NPU 910C on ARM AntOS"
    echo "执行910C + ARM + AntOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM AntOS installation completed!"
}

install_cann_910c_arm_alios() {
    echo "Installing CANN for NPU 910C on ARM AliOS"
    echo "执行910C + ARM + AliOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM AliOS installation completed!"
}

install_cann_910c_arm_ctyunos() {
    echo "Installing CANN for NPU 910C on ARM CTyunOS"
    echo "执行910C + ARM + CTyunOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM CTyunOS installation completed!"
}

install_cann_910c_arm_culinux() {
    echo "Installing CANN for NPU 910C on ARM CULinux"
    echo "执行910C + ARM + CULinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM CULinux installation completed!"
}

install_cann_910c_arm_tlinux() {
    echo "Installing CANN for NPU 910C on ARM Tlinux"
    echo "执行910C + ARM + Tlinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM Tlinux installation completed!"
}

install_cann_910c_arm_mtos() {
    echo "Installing CANN for NPU 910C on ARM MTOS"
    echo "执行910C + ARM + MTOS的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM MTOS installation completed!"
}

install_cann_910c_arm_velinux() {
    echo "Installing CANN for NPU 910C on ARM veLinux"
    echo "执行910C + ARM + veLinux的安装命令..."
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    echo "NPU 910C ARM veLinux installation completed!"
}

# 主函数
main() {
    # 检查是否提供了参数
    if [ $# -eq 0 ]; then
        echo "Error: No architecture specified"
        show_help
        return 1
    fi

    # 获取参数
    local ARCH="$1"

    # 验证参数
    if [ "$ARCH" != "x86" ] && [ "$ARCH" != "arm" ]; then
        echo "Error: Invalid architecture '$ARCH'. Must be 'x86' or 'arm'"
        return 1
    fi

    echo "========================================"
    echo "CANN Installation"
    echo "========================================"
    echo ""
    echo "Detected architecture: $ARCH"

    # 检测操作系统类型
    local OS_TYPE=$(detect_os)
    if [ -z "$OS_TYPE" ]; then
        echo "Error: Unable to detect operating system"
        return 1
    fi
    echo "Detected OS: $OS_TYPE"

    # 验证操作系统是否支持
    case "$OS_TYPE" in
        "ubuntu"|"openeuler"|"centos"|"debian"|"kylin"|"bclinux"|"uosv20"|"antos"|"alios"|"ctyunos"|"culinux"|"tlinux"|"mtos"|"velinux")
            echo "Supported OS detected: $OS_TYPE"
            ;;
        *)
            echo "Error: Unsupported operating system: $OS_TYPE"
            echo "Supported operating systems: Ubuntu, openEuler, CentOS, Debian, Kylin, BCLinux, UOSV20, AntOS, AliOS, CTyunOS, CULinux, Tlinux, MTOS, veLinux"
            return 1
            ;;
    esac

    # 检测NPU硬件类型
    echo ""
    echo "Detecting NPU hardware type..."

    local npu_device_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2)

    if [ -z "$npu_device_id" ]; then
        echo "Error: No NPU device found via lspci command."
        echo "Please ensure NPU device is properly installed and detected."
        echo "Command 'lspci -n -D | grep -o \"19e5:d[0-9a-f]\{3\}\"' returned empty result."
        return 1
    fi

    echo "Detected NPU device ID: $npu_device_id"

    # 转换为小写方便比较
    local npu_device_id_lower=$(echo "$npu_device_id" | tr '[:upper:]' '[:lower:]')
    local NPU_TYPE=""

    # 根据设备ID确定NPU类型（忽略大小写，完全匹配）
    case "$npu_device_id_lower" in
        d802)
            NPU_TYPE="910b"
            echo "NPU Type: Ascend 910B (d802)"
            ;;
        d803)
            NPU_TYPE="910c"
            echo "NPU Type: Ascend 910C (d803)"
            ;;
        *)
            echo "Error: Unsupported NPU device ID: $npu_device_id"
            echo "Supported device IDs: d802 (Ascend 910B), d803 (Ascend 910C)"
            echo "Your device ID: $npu_device_id"
            return 1
            ;;
    esac

    echo ""
    echo "System Configuration:"
    echo "  Architecture: $ARCH"
    echo "  OS: $OS_TYPE"
    echo "  NPU Type: $NPU_TYPE"

    # 构建安装函数名
    local INSTALL_FUNC="install_cann_${NPU_TYPE}_${ARCH}_${OS_TYPE}"
    echo "Looking for installation function: $INSTALL_FUNC"

    # 检查安装函数是否存在
    if ! declare -f "$INSTALL_FUNC" > /dev/null; then
        echo "Error: Installation function '$INSTALL_FUNC' not found"
        echo ""
        echo "Available installation functions:"
        declare -f | grep "^install_cann_" | awk '{print $1}' | sed 's/()$//' | sort
        return 1
    fi

    echo ""
    echo "Starting CANN installation for NPU $NPU_TYPE on $ARCH $OS_TYPE..."

    # 执行对应的安装函数
    $INSTALL_FUNC

    # 检查安装结果
    if [ $? -eq 0 ]; then
        echo ""
        echo "CANN installation completed successfully!"
        echo "  OS: $OS_TYPE"
        echo "  Architecture: $ARCH"
        echo "  NPU Type: $NPU_TYPE"
        return 0
    else
        echo ""
        echo "CANN installation failed!"
        echo "  OS: $OS_TYPE"
        echo "  Architecture: $ARCH"
        echo "  NPU Type: $NPU_TYPE"
        return 1
    fi
}

# 调用主函数
main "$@"

# 根据主函数返回值退出
exit $?