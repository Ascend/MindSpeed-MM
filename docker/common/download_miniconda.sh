#!/bin/bash
set -e

OUTPUT_DIR="${1:-.}"
REQUESTED_ARCH="${2:-}"
PYTHON_VERSION="${3:-3.11}"

if [ -n "$REQUESTED_ARCH" ]; then
    ARCH="$REQUESTED_ARCH"
else
    ARCH=$(uname -m)
fi

# Map "3.11" -> "311" to select the matching Miniconda variant, which in turn
# determines the Python version of the conda base environment.
PY_STR="${PYTHON_VERSION//./}"

echo "=========================================="
echo "Downloading Miniconda for Python ${PYTHON_VERSION}"
echo "Architecture: ${ARCH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

MINICONDA_VERSION="26.1.1-1"

if [ "$ARCH" = "x86_64" ]; then
    MINICONDA_FILE="Miniconda3-py${PY_STR}_${MINICONDA_VERSION}-Linux-x86_64.sh"
elif [ "$ARCH" = "aarch64" ]; then
    MINICONDA_FILE="Miniconda3-py${PY_STR}_${MINICONDA_VERSION}-Linux-aarch64.sh"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    echo "Supported architectures: x86_64, aarch64"
    exit 1
fi

MIRROR_URLS=(
    "https://repo.anaconda.com/miniconda/${MINICONDA_FILE}"
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/${MINICONDA_FILE}"
)

mkdir -p "${OUTPUT_DIR}"

cd "${OUTPUT_DIR}"

if [ -f "${MINICONDA_FILE}" ]; then
    FILE_SIZE=$(stat -c%s "${MINICONDA_FILE}" 2>/dev/null || stat -f%z "${MINICONDA_FILE}" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 50000000 ]; then
        echo ">>> File already exists and looks valid: ${MINICONDA_FILE} (${FILE_SIZE} bytes)"
        echo ">>> Skipping download. Delete the file if you want to re-download."
    else
        echo ">>> Existing file seems too small (${FILE_SIZE} bytes), re-downloading..."
        rm -f "${MINICONDA_FILE}"
    fi
fi

if [ ! -f "${MINICONDA_FILE}" ]; then
    DOWNLOAD_SUCCESS=false
    for MINICONDA_URL in "${MIRROR_URLS[@]}"; do
        echo ">>> Downloading ${MINICONDA_FILE}..."
        echo ">>> URL: ${MINICONDA_URL}"

        if wget -c "${MINICONDA_URL}"; then
            FILE_SIZE=$(stat -c%s "${MINICONDA_FILE}" 2>/dev/null || stat -f%z "${MINICONDA_FILE}" 2>/dev/null)
            if [ "$FILE_SIZE" -gt 50000000 ]; then
                echo ">>> Download completed successfully! (${FILE_SIZE} bytes)"
                DOWNLOAD_SUCCESS=true
                break
            else
                echo ">>> Downloaded file too small (${FILE_SIZE} bytes), trying next mirror..."
                rm -f "${MINICONDA_FILE}"
            fi
        else
            echo ">>> Download failed from ${MINICONDA_URL}, trying next mirror..."
            rm -f "${MINICONDA_FILE}"
        fi
    done

    if [ "$DOWNLOAD_SUCCESS" = false ]; then
        echo ">>> ERROR: All download mirrors failed!"
        exit 1
    fi
fi

if [ -f "${MINICONDA_FILE}" ]; then
    FILE_SIZE=$(stat -c%s "${MINICONDA_FILE}" 2>/dev/null || stat -f%z "${MINICONDA_FILE}" 2>/dev/null)
    echo ">>> File size: ${FILE_SIZE} bytes"
fi

echo "=========================================="
echo "Miniconda downloaded successfully!"
echo "File: ${OUTPUT_DIR}/${MINICONDA_FILE}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify the file: bash ${MINICONDA_FILE} --help"
echo "  2. Use it to build Docker image:"
echo "     bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11"
# Last line: the resulting installer file name (consumed by build.sh).
echo "${MINICONDA_FILE}"
