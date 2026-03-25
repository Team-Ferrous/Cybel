#!/bin/bash
source venv/bin/activate

echo "Q1: Agent loop limits" > output.txt
saguaro query "agent loop limits task completion" --k 3 >> output.txt 2>&1

echo -e "\nQ2: Context management" >> output.txt
saguaro query "context management packing token limit" --k 3 >> output.txt 2>&1

echo -e "\nQ3: Patch generation" >> output.txt
saguaro query "patch generation diff bias full file rewrite bias" --k 3 >> output.txt 2>&1

echo -e "\nQ4: Planner executor verification" >> output.txt
saguaro query "planner executor verification completion" --k 3 >> output.txt 2>&1

echo -e "\nQ5: Safety slot machine prompt constraints" >> output.txt
saguaro query "safety layers slot machine prompt constraints partial completion" --k 3 >> output.txt 2>&1
