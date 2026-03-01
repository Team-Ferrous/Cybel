import os
import json

input_path  = "C:/Users/EnocEscalona/Documents/Supercell_AI_Dev/Data/Input/"
output_path = "C:/Users/EnocEscalona/Documents/Supercell_AI_Dev/Data/Output/"

# Ensure the Input folder exists
if not os.path.exists(input_path):
    os.makedirs(input_path)
    print(f"Created missing folder: {input_path}")
    print("Please add files to the Input folder and re-run the script.")
    exit()

# Ensure the Output folder exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

files = os.listdir(input_path)

for f in files:
    input_file = os.path.join(input_path, f)
    output_file = os.path.join(output_path, f"{os.path.splitext(f)[0]}.json")  # Match input file name with .json extension
    print(f"Converted lines to {output_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # Skip empty lines
                json.dump({"text": line.strip()}, outfile, ensure_ascii=False)
                outfile.write('\n')  # New line for each entry
