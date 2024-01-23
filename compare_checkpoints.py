import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        abs_tolerance = 1e-6
        for key in state_dict1.keys():
            if key not in state_dict2:
                print(f"Key {key} not in second state dict.")
            elif not torch.allclose(state_dict1[key], state_dict2[key], atol=abs_tolerance):
                # print(f"Difference in key {key}: {state_dict1[key]} vs {state_dict2[key]}")
                diff = state_dict1[key] - state_dict2[key]
                print(f"Difference in key {key}: {diff}")
        for key in state_dict2.keys():
            if key not in state_dict1:
                print(f"Key {key} not in first state dict.")
            # elif not torch.allclose(state_dict1[key], state_dict2[key]):
                # print(f"Difference in key {key}: {state_dict1[key]} vs {state_dict2[key]}")

    text1 = generate_text(file1)
    text2 = generate_text(file2)
    print(text1)
    print("="*80)
    print(text2)

def generate_text(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    try:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = torch.load(checkpoint_path)["state"]
        model.load_state_dict(state_dict)


    text = "Once upon a time, there was a"
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([tokens])

    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor)

    # breakpoint()
    token_ids = torch.argmax(outputs['logits'], dim=-1)
    decoded_output = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
    return decoded_output



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default="./output_trl/model.pt", help="Path to the first file")
    parser.add_argument("--file2", type=str, default="default_file2.pth", help="Path to the second file")
    args = parser.parse_args()

    # Compare the files
    compare_files(args.file1, args.file2)
