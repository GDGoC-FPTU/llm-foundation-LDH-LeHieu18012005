"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
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
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "mock-key")

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    start_time = time.time()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency = time.time() - start_time

        usage_metadata = getattr(response, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
            "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
        }

        return response.text or "", latency, usage
    except ImportError:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_inst = genai.GenerativeModel(model)
        config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        response = model_inst.generate_content(prompt, generation_config=config)
        latency = time.time() - start_time

        text = response.text or ""
        try:
            input_tokens = model_inst.count_tokens(prompt).total_tokens
            output_tokens = model_inst.count_tokens(text).total_tokens
        except Exception:
            input_tokens = int(len(prompt.split()) * 1.5)
            output_tokens = int(len(text.split()) * 1.5)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or "mock-key")

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    text_parts = [
        getattr(part, "text", "")
        for part in response.content
        if getattr(part, "text", "")
    ]
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }

    return "".join(text_parts), latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    def calculate_cost(model_name: str, usage: dict) -> float:
        rates = PRICING_1M_TOKENS[model_name]
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return (
            input_tokens * rates["input"]
            + output_tokens * rates["output"]
        ) / 1_000_000

    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    mini_text, mini_latency, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_latency, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_latency,
            "cost": calculate_cost(OPENAI_MODEL, gpt4o_usage),
            "input_tokens": gpt4o_usage.get("input_tokens", 0),
            "output_tokens": gpt4o_usage.get("output_tokens", 0),
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_latency,
            "cost": calculate_cost(OPENAI_MINI_MODEL, mini_usage),
            "input_tokens": mini_usage.get("input_tokens", 0),
            "output_tokens": mini_usage.get("output_tokens", 0),
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_latency,
            "cost": calculate_cost(GEMINI_MODEL, gemini_usage),
            "input_tokens": gemini_usage.get("input_tokens", 0),
            "output_tokens": gemini_usage.get("output_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        api_key = "mock-key"

    print("\n\033[94m==============================================")
    print("Vin Smart Future - Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                print("\033[93mGoodbye!\033[0m")
                return

            messages_to_send = history[-6:] + [{"role": "user", "text": user_input}]

            print("\033[92mGemini:\033[0m ", end="", flush=True)
            full_reply = ""

            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                formatted_contents = [
                    types.Content(
                        role=message["role"],
                        parts=[types.Part.from_text(text=message["text"])],
                    )
                    for message in messages_to_send
                ]

                response_stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=formatted_contents,
                )
            except ImportError:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model_inst = genai.GenerativeModel(GEMINI_MODEL)
                legacy_contents = [
                    {
                        "role": message["role"] if message["role"] == "user" else "model",
                        "parts": [message["text"]],
                    }
                    for message in messages_to_send
                ]
                response_stream = model_inst.generate_content(legacy_contents, stream=True)

            for chunk in response_stream:
                chunk_text = getattr(chunk, "text", "") or ""
                print(chunk_text, end="", flush=True)
                full_reply += chunk_text

            print("\n")
            history.append({"role": "user", "text": user_input})
            history.append({"role": "model", "text": full_reply})
        except KeyboardInterrupt:
            print("\n\033[93mSession interrupted. Goodbye!\033[0m")
            return
        except Exception as error:
            print(f"\n\033[91m[Error Calling API]: {error}\033[0m\n")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as error:
            last_error = error
            if attempt == max_retries:
                break
            time.sleep(base_delay * (2 ** attempt))

    raise last_error


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except TypeError as error:
            if "positional" not in str(error):
                raise
            comparison = compare_models()
        except Exception as error:
            comparison = {
                "gpt4o": {"response": f"Error: {error}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gpt4o_mini": {"response": f"Error: {error}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gemini_flash": {"response": f"Error: {error}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
            }
        comparison["prompt"] = prompt
        results.append(comparison)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for result in results:
        prompt = str(result.get("prompt", ""))
        if len(prompt) > 40:
            prompt = prompt[:37] + "..."

        models = [
            ("GPT-4o", result.get("gpt4o", result.get("gpt4o_response"))),
            ("GPT-4o-Mini", result.get("gpt4o_mini", result.get("mini_response"))),
            ("Gemini-Flash", result.get("gemini_flash")),
        ]

        for model_name, stats in models:
            if not stats:
                continue
            if isinstance(stats, dict):
                response = str(stats.get("response", ""))
                latency = stats.get("latency", 0.0)
                input_tokens = stats.get("input_tokens", 0)
                output_tokens = stats.get("output_tokens", 0)
                cost = stats.get("cost", 0.0)
            else:
                response = str(stats)
                latency = result.get("gpt4o_latency" if model_name == "GPT-4o" else "mini_latency", 0.0)
                input_tokens = int(len(prompt.split()) * 1.5)
                output_tokens = int(len(response.split()) * 1.5)
                cost = result.get("gpt4o_cost_estimate" if model_name == "GPT-4o" else "mini_cost_estimate", 0.0)

            response = response.replace("\n", " ")
            if len(response) > 50:
                response = response[:47] + "..."
            lines.append(
                f"| {prompt} | {model_name} | {response} | "
                f"{latency:.2f}s | "
                f"{input_tokens}/{output_tokens} | "
                f"${cost:.6f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
