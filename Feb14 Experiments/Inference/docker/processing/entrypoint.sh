#!/usr/bin/env bash
set -e

if [ "$JOB_TYPE" = "COMPARE" ]; then
    echo "Running compare_and_report.py"
    python /opt/program/compare_and_report.py
else
    echo "Running evaluation_processor.py"
    python /opt/program/evaluation_processor.py
fi