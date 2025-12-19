import os
import time
from openai import OpenAI
import numpy as np
import anthropic
from httpx import HTTPStatusError
from together import Together

try:
    from api_keys import *
except ModuleNotFoundError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
    LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")

def run_model(
                    input_prompt = None,
                    persona = None,
                    model_card = 'gpt-3.5-turbo',
                    temperature = 0.9, 
                    top_p = 0.9,
                    max_tokens = 3000,
                    message = None,
                    system = None
                ):
    if "gpt" in model_card:
        return openai_chat_gen(input_prompt, persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens = max_tokens, message = message, system=system)
    elif "claude" in model_card:
        return claude_chat_gen(input_prompt, persona=persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens = max_tokens)
    elif "llama" in model_card:
        return llama_chat_gen(input_prompt, persona=persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens = max_tokens)

def openai_chat_gen(input_prompt = None,
                    persona = None,
                    apikey = OPENAI_API_KEY,
                    model_card = 'gpt-3.5-turbo',
                    temperature = 0.0, 
                    top_p = 0.9,
                    max_tokens = 4000,
                    max_attempt = 3,
                    time_interval = 2,
                    system=None,
                    message = None,
                   ):
    
  
    client = OpenAI(api_key=apikey)


    
    if not message:
        if persona:
            persona_prompt = f"Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."
            message=[{"role": "system", "content": persona_prompt},
                    {"role": "user", "content": input_prompt}]
        
        else:
            if system:
                message=[{"role": "system", "content": system},
                         {"role": "user", "content": input_prompt}]
            else:
                message=[{"role": "user", "content": input_prompt}]
        
    while max_attempt > 0:
        try:
            response = client.chat.completions.create(
                model= model_card,
                messages = message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return response.choices[0].message.content

        except Exception as e:

            print('Exception Raised: ', e)

            max_attempt -= 1
            time.sleep(time_interval)

            print('Retrying left: ', max_attempt)

    return 'Error'

def claude_chat_gen(
    input_prompt,
    persona=None,
    apikey=CLAUDE_API_KEY,
    model_card="claude-3-5-haiku-20241022",
    temperature=0,
    top_p=None,
    max_tokens=4000,
    max_attempt=3,
    time_interval=5,
    system=None,
):
    assert isinstance(input_prompt, str), "claude api does not support batch inference."

    client = anthropic.Anthropic(api_key=apikey)

    system_parts = []
    if system:
        system_parts.append(system)
    if persona:
        persona_prompt = (
            f"Adopt the identity of {persona}. "
            "Answer the questions while staying in strict accordance with the nature of this identity."
        )
        system_parts.append(persona_prompt)

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    messages = [{"role": "user", "content": input_prompt}]

    attempt = 0
    while attempt < max_attempt:
        attempt += 1
        try:
            kwargs = {
                "model": model_card,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text

        except HTTPStatusError as e:
            status = e.response.status_code
            print(f"[Attempt {attempt}] HTTP error: {status} - {e}")

            # Transient server issues: retry
            if status in (500, 502, 503, 504, 529):
                if attempt < max_attempt:
                    sleep_s = time_interval * attempt  # simple backoff
                    print(f"Transient error, retrying in {sleep_s} seconds...")
                    time.sleep(sleep_s)
                    continue
                else:
                    break
            else:
                # Non-retryable HTTP error (4xx etc.)
                break

        except Exception as e:
            print(f"[Attempt {attempt}] Exception Raised:", e)
            if attempt < max_attempt:
                sleep_s = time_interval * attempt
                print(f"Retrying in {sleep_s} seconds...")
                time.sleep(sleep_s)
                continue
            else:
                break

    return "As an AI Model I cannot answer"


def llama_chat_gen(input_prompt,
                   persona = None,
                    apikey = LLAMA_API_KEY,
                    model_card = 'meta-llama/Meta-Llama-3-70B',
                    temperature = 0.9, 
                    top_p = 0.9,
                    max_attempt = 3,
                    time_interval = 5
                   ):

    assert (type(input_prompt) == str
            ), "openai api does not support batch inference."


    client = Together(api_key=apikey)
    
    if persona:
        persona_prompt = f"Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."
        message=[{"role": "system", "content": persona_prompt},
                 {"role": "user", "content": input_prompt}]
    else:
        message=[{"role": "user", "content": input_prompt}]
    
    while max_attempt > 0:

        try:
            response = client.chat.completions.create(
                model= model_card,
                messages = message,
                temperature=temperature,
                top_p = top_p,
            )
            return response.choices[0].message.content

        except Exception as e:

            print('Exception Raised: ', e)

            max_attempt -= 1
            time.sleep(time_interval)

            print('Retrying left: ', max_attempt)

    return 'Error'
