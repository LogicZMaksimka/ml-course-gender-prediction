import asyncio
import time
from functools import wraps
import httpx
import openai
from tqdm.asyncio import tqdm_asyncio, tqdm

API_BASE_URL = "https://inference.airi.net:46783/v1"
API_KEY = ""
HTTPX_CLIENT = httpx.AsyncClient(verify=False)
MAX_CONCURRENT = 16
SEM = asyncio.Semaphore(MAX_CONCURRENT)
BATCH_GATHER_SIZE = MAX_CONCURRENT * 2  # Number of dialogues per async gather batch (limits memory usage)

def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.monotonic() - start
            return {"response": result, "execution_time": execution_time}
        except Exception:
            raise
    return wrapper


@async_timer
async def streaming_query(messages: list[dict], model: str) -> dict:
    """Выполняет один диалог с потоковой передачей"""
    client = openai.AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        http_client=HTTPX_CLIENT,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        # stream_options={"include_usage": True},
        temperature=0.8,
        top_p=0.9,
        logprobs=True,
        top_logprobs=10,
        stream=False
    )

    return response

async def limited_streaming_query(messages: list[dict], model: str) -> dict:
	"""Обертка для ограничения количества одновременных запросов"""
	async with SEM:
		return await streaming_query(messages, model)

async def batched_streaming_query(dialogues: list[list[dict]], model: str, batch_size: int = BATCH_GATHER_SIZE) -> list[dict]:
    tasks = [limited_streaming_query(dialogue, model) for dialogue in dialogues]
    results = []
    for i in tqdm(range(0, len(tasks), batch_size)):
        batch_tasks = tasks[i:i + batch_size]
        batch_results = await tqdm_asyncio.gather(*batch_tasks)
        results.extend(batch_results)
    return results