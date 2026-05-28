"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1
"""

import os
import time
from typing import Any, Callable

import openai
from google import genai
from google.genai import types
import anthropic

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    latency = time.time() - start

    text = response.choices[0].message.content

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Gemini
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

    start = time.time()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    latency = time.time() - start

    text = response.text

    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count
    }

    return text, latency, usage
# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    start = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    latency = time.time() - start

    text = response.content[0].text

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:

    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt)

    mini_text, mini_latency, mini_usage = call_openai(
        prompt,
        model=OPENAI_MINI_MODEL
    )

    gemini_text, gemini_latency, gemini_usage = call_gemini(prompt)

    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"]
        + gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]
        + mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"]
        + gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_latency,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"]
        },

        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_latency,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"]
        },

        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_latency,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"]
        }
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    history = []

    print("=== Gemini Streaming Chatbot ===")

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        history.append({
            "role": "user",
            "content": user_input
        })

        history = history[-6:]

        formatted_history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in history
            ]
        )

        print("Gemini: ", end="", flush=True)

        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=formatted_history
        )

        assistant_response = ""

        for chunk in response_stream:

            if hasattr(chunk, "text") and chunk.text:

                print(chunk.text, end="", flush=True)

                assistant_response += chunk.text

        print("\n")

        history.append({
            "role": "assistant",
            "content": assistant_response
        })


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:

    for attempt in range(max_retries + 1):

        try:
            return fn()

        except Exception:

            if attempt == max_retries:
                raise

            delay = base_delay * (2 ** attempt)

            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:

    results = []

    for prompt in prompts:

        try:
            result = compare_models(prompt)
        except TypeError as exc:
            if "positional" not in str(exc):
                raise
            result = compare_models()

        result["prompt"] = prompt

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:

    table = (
        "| Prompt | Model | Response | Latency | Tokens (In/Out) | Cost (USD) |\n"
        "|---|---|---|---|---|---|\n"
    )

    for result in results:

        prompt = result["prompt"]

        display_names = {
            "gpt4o": "GPT-4o",
            "gpt4o_mini": "GPT-4o-Mini",
"gemini_flash": "Gemini-Flash",
        }

        for model_name in ["gpt4o", "gpt4o_mini", "gemini_flash"]:

            model_data = result[model_name]

            response = model_data["response"][:50]

            latency = model_data["latency"]

            input_tokens = model_data["input_tokens"]

            output_tokens = model_data["output_tokens"]

            cost = model_data["cost"]

            table += (
                f"| {prompt} | {display_names[model_name]} | {response} | "
                f"{latency:.2f}s | "
                f"{input_tokens}/{output_tokens} | "
                f"${cost:.6f} |\n"
            )

    return table
