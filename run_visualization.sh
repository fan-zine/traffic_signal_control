#!/bin/bash

# Define input parameters
FILE_PATTERN="results/my_result_dcrnn_conn0_ep"
SEPARATOR=","
SYSTEM_METRICS=("system_total_stopped" "system_total_waiting_time" "system_mean_waiting_time" "system_mean_speed" "agents_total_stopped" "agents_total_accumulated_waiting_time")
AGENTS=("A0" "A1" "A2" "A3" "B0" "B1" "B2" "B3" "C0" "C1" "C2" "C3" "D0" "D1" "D2" "D3")
AGENT_METRICS=("stopped" "accumulated_waiting_time" "average_speed")
MOVING_AVERAGE=5
EPISODES_INTERVAL=20
OUTPUT_DIR="outputs"
HEATMAP=true

# Run the Python visualization script
python custom_plot.py \
    -f "$FILE_PATTERN" \
    -sep "$SEPARATOR" \
    -system "${SYSTEM_METRICS[@]}" \
    -agents "${AGENTS[@]}" \
    -metrics "${AGENT_METRICS[@]}" \
    -ma "$MOVING_AVERAGE" \
    -episodes_interval "$EPISODES_INTERVAL" \
    -heatmap \
    -output "$OUTPUT_DIR"
