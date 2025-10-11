# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$BASE_DIR:$PYTHONPATH
echo $BASE_DIR

GENERATE_BASELINE_PATH=$1
echo "GENERATE_BASELINE_PATH: $GENERATE_BASELINE_PATH"

EXEC_SUCCESS=0 # default 0 success
SHELL_SCRIPTS_DIR="$BASE_DIR/shell_scripts"
LOSS_SHELL_SCRIPTS_DIR="$BASE_DIR/loss_shell_scripts"
CONFIGS_DIR='$BASE_DIR/run_configs'
BASELINE_DIR="$BASE_DIR/baseline_results"
EXEC_PY_DIR=$(dirname "$BASE_DIR")
RESULTS_DIR="/home/ci_resource/ci_results"

echo "SHELL_SCRIPTS_DIR: $SHELL_SCRIPTS_DIR"
echo "LOSS_SHELL_SCRIPTS_DIR: $LOSS_SHELL_SCRIPTS_DIR"
echo "BASELINE_DIR: $BASELINE_DIR"
echo "EXEC_PY_DIR: $EXEC_PY_DIR"

rm -rf $SHELL_SCRIPTS_DIR
rm -rf $LOSS_SHELL_SCRIPTS_DIR
cp -r "/home/ci_resource/local_st/st_shell/shell_scripts" $SHELL_SCRIPTS_DIR
cp -r '/home/ci_resource/local_st/st_shell/loss_shell_scripts' $LOSS_SHELL_SCRIPTS_DIR

GENERATE_LOG_DIR="$BASE_DIR/run_logs"
FAIL_LOG_DIR="$BASE_DIR/fail_logs"
GENERATE_JSON_DIR="$BASE_DIR/run_jsons"
TMP_LOG_DIR="$BASE_DIR/tmp_logs"

# 清空历史累计
rm -rf $TMP_LOG_DIR

mkdir -p $GENERATE_LOG_DIR
mkdir -p $GENERATE_JSON_DIR

rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*


# clear fail last fail-log path
rm -rf $FAIL_LOG_DIR
mkdir -p $FAIL_LOG_DIR
rm -rf $TMP_LOG_DIR
mkdir $TMP_LOG_DIR

rm -rf $CONFIGS_DIR
cp -r /home/ci_resource/local_st/run_configs $BASE_DIR


## step 2: running scripts and execute `test_ci_st.py`
rm -rf $BASELINE_DIR
cp -r "/home/ci_resource/local_st/local_st_baseline/efficient/" $BASELINE_DIR
declare -A TEST_CASE_TIMES
for test_case in "$SHELL_SCRIPTS_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")
    echo "$file_name_prefix"

    START_TIME=$(date +%s)
    # create empty json file to receive the result parsered from log
    touch "$GENERATE_JSON_DIR/$file_name_prefix.json"

    # test_result_file on efficient
    test_result_file_efficient="$TMP_LOG_DIR/${file_name_prefix}_efficient.log"

    # if executing the shell has failed, then just exit, no need to compare.
    bash $test_case 2>&1 | tee "$GENERATE_LOG_DIR/${file_name_prefix}_run.log"
    SCRIPT_EXITCODE=${PIPESTATUS[0]}

    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script $file_name has failed."
        cp "$GENERATE_LOG_DIR/${file_name_prefix}_run.log" $FAIL_LOG_DIR
        continue
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
            pytest -k "not loss" $BASE_DIR/test_tools/test_ci_st.py \
                --baseline-json $BASELINE_DIR/$file_name_prefix.json \
                --generate-log $GENERATE_LOG_DIR/${file_name_prefix}_run.log \
                --generate-json $GENERATE_JSON_DIR/$file_name_prefix.json | tee $test_result_file_efficient
    fi


    PYTEST_EXITCODE=$?
    echo $PYTEST_EXITCODE
    if [ $PYTEST_EXITCODE -ne 0 ]; then
        echo "$file_name_prefix compare to baseline has failed, check it!"
        cp $test_result_file_efficient $FAIL_LOG_DIR
    else
        echo "Pretrain $file_name_prefix execution success."
    fi

done

# step 3: running loss scripts and execute 'test_ci_st.py'
rm -rf $BASELINE_DIR
cp -r "/home/ci_resource/local_st/local_st_baseline/loss/" $BASELINE_DIR
for test_case in "$LOSS_SHELL_SCRIPTS_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}_loss")
    echo "$file_name_prefix"

    START_TIME=$(date +%s)
    # create empty json file to receive the result parsered from log
    touch "$GENERATE_JSON_DIR/$file_name_prefix.json"

    # if executing the shell has failed, then just exit, no need to compare.
    log_file="$GENERATE_LOG_DIR/${file_name_prefix}_run.log"
    test_result_file_loss="$TMP_LOG_DIR/${file_name_prefix}.log"
    bash $test_case 2>&1 | tee $log_file
    SCRIPT_EXITCODE=${PIPESTATUS[0]}

    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script $file_name has failed!"
        cp $log_file $FAIL_LOG_DIR
        ((EXEC_SUCCESS++))
        continue
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
            pytest -k "loss" $BASE_DIR/test_tools/test_ci_st.py \
                --baseline-json $BASELINE_DIR/$file_name_prefix.json \
                --generate-log $GENERATE_LOG_DIR/${file_name_prefix}_run.log \
                --generate-json $GENERATE_JSON_DIR/$file_name_prefix.json >> $test_result_file_loss
    fi


    PYTEST_EXITCODE=$?
    echo $PYTEST_EXITCODE
    if [ $PYTEST_EXITCODE -ne 0 ]; then
        echo "$file_name_prefix compare to baseline has failed, check it!"
        cp $test_result_file_loss $FAIL_LOG_DIR
        ((EXEC_SUCCESS++))
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

if [ $EXEC_SUCCESS -ne 0 ]; then
  # conclude all results into results_dir
  current_data=$(date +'%Y-%m-%d')_$(date +'%H%M%S')
  results_path="${RESULTS_DIR}/${current_data}"
  mkdir -p $results_path
  cp -r $FAIL_LOG_DIR $results_path
  cp -r $TMP_LOG_DIR $results_path
  cp -r $GENERATE_LOG_DIR $results_path
  cp -r $GENERATE_JSON_DIR $results_path
  exit 1
fi