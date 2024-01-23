import argparse
import torch
import hashlib

def compare_files(file1, file2):
    # Load the state dictionaries
    state_dict1 = torch.load(file1)
    state_dict2 = torch.load(file2)["state"]

    # Compute the checksums
    checksum1 = hashlib.md5(str(state_dict1).encode()).hexdigest()
    checksum2 = hashlib.md5(str(state_dict2).encode()).hexdigest()

    # Compare the checksums
    if checksum1 == checksum2:
        print("The files have the same checksum.")
    else:
        print("The files have different checksums.")

        # breakpoint()
        # Compare the state dictionaries
        for key in state_dict1.keys():
            if key not in state_dict2:
                print(f"Key {key} not in second state dict.")
            # elif not torch.allclose(state_dict1[key], state_dict2[key]):
                # print(f"Difference in key {key}: {state_dict1[key]} vs {state_dict2[key]}")
        for key in state_dict2.keys():
            if key not in state_dict1:
                print(f"Key {key} not in first state dict.")
            # elif not torch.allclose(state_dict1[key], state_dict2[key]):
                # print(f"Difference in key {key}: {state_dict1[key]} vs {state_dict2[key]}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default="./output_trl/model.pt", help="Path to the first file")
    parser.add_argument("--file2", type=str, default="default_file2.pth", help="Path to the second file")
    args = parser.parse_args()

    # Compare the files
    compare_files(args.file1, args.file2)
