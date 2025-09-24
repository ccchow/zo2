#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "ZO2 Installation Test Suite"
echo "============================================================"
echo ""

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="test_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="${RESULTS_DIR}/test_log.txt"

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_script="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${YELLOW}Running: ${test_name}${NC}"
    echo "----------------------------------------"
    
    # Run test and capture output
    if python "$test_script" > "${RESULTS_DIR}/${test_name}.log" 2>&1; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "[PASS] ${test_name}" >> "$LOG_FILE"
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "[FAIL] ${test_name}" >> "$LOG_FILE"
        echo "  See ${RESULTS_DIR}/${test_name}.log for details"
    fi
    echo ""
}

# Test 1: Python version check
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python --version 2>&1)
echo "Python version: $python_version"
echo "$python_version" >> "$LOG_FILE"
echo ""

# Test 2: Import tests
run_test "import_test" "test/installation/test_imports.py"

# Test 3: CUDA compatibility
run_test "cuda_test" "test/installation/test_cuda_compatibility.py"

# Test 4: Minimal training
run_test "training_test" "test/installation/test_minimal_training.py"

# Test 5: Check key dependencies
echo -e "${YELLOW}Checking key dependencies...${NC}"
echo "----------------------------------------"
python -c "
import sys
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except:
    print('✗ PyTorch not found')
    sys.exit(1)

try:
    import transformers
    print(f'✓ Transformers: {transformers.__version__}')
except:
    print('✗ Transformers not found')
    sys.exit(1)

try:
    import accelerate
    print(f'✓ Accelerate: {accelerate.__version__}')
except:
    print('✗ Accelerate not found')
    sys.exit(1)

try:
    import zo2
    print('✓ ZO2 package found')
except:
    print('✗ ZO2 package not found')
    sys.exit(1)
" | tee "${RESULTS_DIR}/dependencies.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies found${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ Some dependencies missing${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""

# Summary
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo -e "Total tests: ${TOTAL_TESTS}"
echo -e "Passed: ${GREEN}${PASSED_TESTS}${NC}"
echo -e "Failed: ${RED}${FAILED_TESTS}${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo "Log file: ${LOG_FILE}"

# Exit code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review the logs.${NC}"
    exit 1
fi