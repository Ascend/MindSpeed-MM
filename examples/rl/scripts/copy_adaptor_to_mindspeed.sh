#!/bin/bash

echo "Copying adaptor files from MindSpeed MM to MindSpeed/Megatron"
echo "[Warning] Some files will be overwritten in MindSpeed/Megatron"

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

cp examples/rl/code/build_tokenizer.py mindspeed/features_manager/tokenizer/build_tokenizer.py
cp examples/rl/code/dot_product_attention.py megatron/core/transformer/dot_product_attention.py

echo "All adaptor files copied successfully!"
echo "Total files copied: 2"
