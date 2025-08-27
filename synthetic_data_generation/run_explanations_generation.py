import json
import asyncio
from src.async_api import batched_streaming_query
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run batched LLM queries.")

    parser.add_argument("--model", type=str, default="Deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                        help="Model name to use")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of dialogues per batch")
    parser.add_argument("--n_dialogues", type=int, default=3,
                        help="Total number of dialogues")
    parser.add_argument("--load_path", type=str, required=True,
                        help="Path to input file with dialogues")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save results")

    return parser.parse_args()


async def main(load_path: str, save_path: str, n_dialogues: int, model_name: str, batch_size: int):
    # --- Load customer info ---
    with open(load_path, "r") as f:
        customers_info = json.load(f)
        customers = customers_info[:n_dialogues]

    dialogues = [
        [
            {"role": "system", "content": customer["system_prompt"]},
            {"role": "user", "content": customer["user_prompt"]}
        ] 
        for customer in customers
    ]

    # --- Get explanations from the model ---
    results = await batched_streaming_query(dialogues, model_name, batch_size)


    # --- Reformat results ---
    results_formatted = []
    for result, customer in zip(results, customers):
        resp = result["response"]
        exec_time = result["execution_time"]
        gender = customer["gender"]
        prompts = {"user_prompt": customer["user_prompt"], "system_prompt": customer["system_prompt"]}

        # --- Extract explanation text ---
        explanation = resp.choices[0].message.content

        # --- Completion tokens ---
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

        results_formatted.append({
            "execution_time": exec_time,
            "prompts":  prompts,
            "gender": gender,
            "completion": {
                "text": explanation,
                "tokens": comp_tokens,
                "logprobs": comp_logprobs,
                "top_tokens": comp_top_tokens,
                "top_logprobs": comp_top_logprobs
            }
        })

    # --- Save everything ---
    with open(save_path, "w") as f:
        json.dump(results_formatted, f, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model
    batch_size = args.batch_size
    n_dialogues = args.n_dialogues
    load_path = args.load_path
    save_path = args.save_path

    asyncio.run(main(
        load_path=load_path,
        save_path=save_path,
        batch_size=batch_size,
        model_name=model_name,
        n_dialogues=n_dialogues
    ))

# python run_explanations_generation.py --load_path ../data/customers_info_true_gender.json --save_path ../data/explanations_true_gender.json   --n_dialogues 1 --batch_size 5 --model "Deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 