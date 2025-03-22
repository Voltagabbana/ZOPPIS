import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_input", required=True, help="Path to the input data file")
parser.add_argument("--data_output", required=False, help="Path to the output data file")

args = parser.parse_args()
path_input = args.data_input
path_output = args.data_output

print("Path to input file:", path_input)
print("Path to output file:", path_output)

