import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from vera import QwenVeRA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--user2id_path", type=str, required=True)
    parser.add_argument("--vera_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="result.json")
    parser.add_argument("--num_beams", type=int, default=10)
    args = parser.parse_args()

    # 1. Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")

    # Load Map & Init VeRA
    with open(args.user2id_path, 'r') as f:
        user2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f}

    model = QwenVeRA(base_model, len(user2id), rank=256)
    model.user_embeddings.load_state_dict(torch.load(args.vera_path))
    model.eval()

    # 2. Inference
    df = pd.read_csv(args.test_file)
    results = []

    print("Evaluating...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = str(row['user_id'])  # e.g. "A123"
        user_idx = user2id.get(uid, 0)

        # Parse History SID List -> String
        hist = row['history_item_sid']
        if isinstance(hist, str):
            try:
                hist = eval(hist)
            except:
                hist = []
        input_text = "".join(hist)

        inputs = tokenizer(input_text, return_tensors="pt").to(base_model.device)
        u_tensor = torch.tensor([user_idx], device=base_model.device)

        with torch.no_grad():
            # generate 会自动调用 forward 注入参数
            outputs = model.generate(
                input_ids=inputs.input_ids,
                user_ids=u_tensor,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id
            )

        gen_seqs = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)

        results.append({
            "user_id": uid,
            "output": row['item_sid'],
            "predict": preds
        })

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()