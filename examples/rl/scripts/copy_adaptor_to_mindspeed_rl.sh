#!/bin/bash

echo "Copying adaptor files from MindSpeed MM to MindSpeed RL/mindspeed_rl"
echo "[Warning] Some files will be overwritten in MindSpeed RL/mindspeed_rl"

while true
do
    read -r -p "Are You Sure? [Y/n] " input

    case $input in
        [yY][eE][sS]|[yY])
            echo "Copying ..."
            break
            ;;

        [nN][oO]|[nN])
            echo "Exit"
            exit 1
            ;;

        *)
            echo "Invalid input..."
            ;;
    esac
done

cp examples/rl/code/base_worker.py mindspeed_rl/workers/base_worker.py

echo "All adaptor files copied successfully!"
echo "Total files copied: 1"
