# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$BASE_DIR:$PYTHONPATH
echo $BASE_DIR
SHELL_SCRIPTS_DIR="$BASE_DIR/shell_scripts"
BASELINE_DIR="$BASE_DIR/baseline_results"
EXEC_PY_DIR=$(dirname "$BASE_DIR")

echo "SHELL_SCRIPTS_DIR: $SHELL_SCRIPTS_DIR"
echo "BASELINE_DIR: $BASELINE_DIR"
echo "EXEC_PY_DIR: $EXEC_PY_DIR"

GENERATE_LOG_DIR="$BASE_DIR/run_logs"
GENERATE_JSON_DIR="$BASE_DIR/run_jsons"

mkdir -p $GENERATE_LOG_DIR
mkdir -p $GENERATE_JSON_DIR

rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*

cd ../
# step 2: running scripts and execute `test_ci_st.py`
declare -A TEST_CASE_TIMES
for test_case in "$SHELL_SCRIPTS_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")
    echo "$file_name_prefix"

    START_TIME=$(date +%s)
    # create empty json file to receive the result parsered from log
    touch "$GENERATE_JSON_DIR/$file_name_prefix.json"

    # if executing the shell has failed, then just exit, no need to compare.
    bash $test_case | tee "$GENERATE_LOG_DIR/$file_name_prefix.log"
    SCRIPT_EXITCODE=${PIPESTATUS[0]}
    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script $file_name has failed. Exit!"
        exit 1
    fi

    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME-START_TIME))
    MINUTES=$((ELAPSED_TIME / 60))
    SECONDS=$((ELAPSED_TIME % 60))
    TEST_CASE_TIMES["$file_name"]="$MINUTES m $SECONDS s"

    if [ "$MINUTES" -gt 0 ]; then
        echo "$(printf '*%.0s' {1..20}) Execution Time for $file_name: *${MINUTES}m ${SECONDS}s* $(printf '*%.0s' {1..20})"
    else
        echo "$(printf '*%.0s' {1..20}) Execution Time for $file_name: *${SECONDS}s* $(printf '*%.0s' {1..20})"
    fi
    if [[ $file_name == inference* ]]; then
            echo "st is an inference task, skip compare result"
        else
            # begin to execute the logic of compare
            pytest -k "not loss" -x $BASE_DIR/test_tools/test_ci_st.py \
                --baseline-json $BASELINE_DIR/$file_name_prefix.json \
                --generate-log $GENERATE_LOG_DIR/$file_name_prefix.log \
                --generate-json $GENERATE_JSON_DIR/$file_name_prefix.json
    fi

    PYTEST_EXITCODE=$?
    echo $PYTEST_EXITCODE
    if [ $PYTEST_EXITCODE -ne 0 ]; then
        echo "$file_name_prefix compare to baseline has failed, check it!"
        exit 1
    else
        echo "Pretrain $file_name_prefix execution success."
    fi

done

echo "$(printf '*%.0s' {1..40})"
echo "* Summary of Execution Times for All Test Cases *"
echo "$(printf '*%.0s' {1..40})"
for file_name in "${!TEST_CASE_TIMES[@]}"; do
    echo "* Execution Time for $file_name: ${TEST_CASE_TIMES[$file_name]} *"
done
echo "$(printf '*%.0s' {1..40})"

