import json
import asyncio
from src.async_api import batched_streaming_query
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run batched LLM queries.")

    parser.add_argument("--model", type=str, default="Deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                        help="Model name to use")
    parser.add_argument("--batch_input_size", type=int, default=10,
                        help="Number of dialogues per input batch (for retry logic)")
    parser.add_argument("--n_dialogues", type=int, default=3,
                        help="Total number of dialogues")
    parser.add_argument("--load_path", type=str, required=True,
                        help="Path to input file with dialogues")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save results")

    return parser.parse_args()

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def process_batch(dialogues_batch, model_name):
    while True:
        try:
            return await batched_streaming_query(dialogues_batch, model_name)
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            raise
        except Exception as e:
            print(f"Batch failed with error: {e}. Retrying in 2 seconds...")
            await asyncio.sleep(2)

async def main(load_path: str, save_path: str, n_dialogues: int, model_name: str, batch_input_size: int):
    # --- Load customer info ---
    with open(load_path, "r") as f:
        customers_info = json.load(f)
        customers = customers_info[512:512+n_dialogues]

    dialogues = [
        [
            {"role": "system", "content": customer["system_prompt"]},
            {"role": "user", "content": customer["user_prompt"]}
        ] 
        for customer in customers
    ]

    # --- Prepare output file ---
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                results = json.load(f)
            except Exception:
                results = []
    else:
        results = []

    processed_count = len(results)
    print(f"Already processed {processed_count} dialogues.")

    # --- Process in batches ---
    for batch_idx, (dialogues_batch, customers_batch) in enumerate(
        zip(chunked(dialogues[processed_count:], batch_input_size), chunked(customers[processed_count:], batch_input_size))
    ):
        print(f"Processing batch {batch_idx+1} ({len(dialogues_batch)} dialogues)...")
        batch_results = await process_batch(dialogues_batch, model_name)
        for result, customer in zip(batch_results, customers_batch):
            resp = result["response"]
            exec_time = result["execution_time"]
            prompt_gender = customer["prompt_gender"]
            true_gender = customer["true_gender"]
            prompts = {"user_prompt": customer["user_prompt"], "system_prompt": customer["system_prompt"]}

            explanation = resp.choices[0].message.content

            comp_tokens, comp_logprobs = [], []
            comp_top_tokens, comp_top_logprobs = [], []

            if resp.choices[0].logprobs and hasattr(resp.choices[0].logprobs, "content"):
                for token_info in resp.choices[0].logprobs.content:
                    comp_tokens.append(token_info.token)
                    comp_logprobs.append(token_info.logprob)

                    top_tokens = [alt.token for alt in token_info.top_logprobs[:10]]
                    top_logps = [alt.logprob for alt in token_info.top_logprobs[:10]]

                    comp_top_tokens.append(top_tokens)
                    comp_top_logprobs.append(top_logps)

            results.append({
                "execution_time": exec_time,
                "prompts":  prompts,
                "prompt_gender": prompt_gender,
                "true_gender": true_gender,
                "completion": {
                    "text": explanation,
                    "tokens": comp_tokens,
                    "logprobs": comp_logprobs,
                    "top_tokens": comp_top_tokens,
                    "top_logprobs": comp_top_logprobs
                }
            })

            # --- Save after each dialogue ---
            with open(save_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()

    model_name = args.model
    batch_input_size = args.batch_input_size
    n_dialogues = args.n_dialogues
    load_path = args.load_path

    model_short_name = model_name.split("/")[-1]
    path_no_ext = args.save_path.split(".json")[0]
    save_path = f"{path_no_ext}_{model_short_name}_{n_dialogues}.json"

    try:
        asyncio.run(main(
            load_path=load_path,
            save_path=save_path,
            batch_input_size=batch_input_size,
            model_name=model_name,
            n_dialogues=n_dialogues
        ))
    except KeyboardInterrupt:
        print("Stopped by user.")

# Example usage:
# python run_explanations_generation.py --load_path ../data/customers_info_true_gender.json --save_path ../data/explanations_true_gender.json --n_dialogues 1 --batch_gather_size 5 --batch_input_size 10 --model "Deepseek-ai/DeepSeek-R1-Distill-Llama