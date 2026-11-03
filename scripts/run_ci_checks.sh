#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

checks=(
  _smoke_test_v2.py
  _test_hpmv2.py
  _test_config_override.py
  _test_et_only.py
  _test_hpm_changes.py
)

for check in "${checks[@]}"; do
  echo "Running ${check}"
  "${PYTHON_BIN}" "${check}"
done

echo "Running end-to-end v2 smoke"
smoke_log="$(mktemp)"
trap 'rm -f "${smoke_log}"' EXIT
"${PYTHON_BIN}" tmew1_run_v2.py --smoke 2>&1 | tee "${smoke_log}"

if grep -Eiq '(^|[^[:alnum:]_])(nan|inf|infinity)([^[:alnum:]_]|$)|skipping backward|non-finite' "${smoke_log}"; then
  echo "End-to-end smoke emitted a non-finite or skipped-step signal" >&2
  exit 1
fi
