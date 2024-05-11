import os
import json

import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt

import config
from prompts import (
    get_format_sql_response_messages,
    get_chat_completion_request_system_message,
    get_chat_completion_request_system_instructions,
)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages, tools=None, tool_choice=None, model=config.GPT_MODEL
):
    new_messages = []
    system_message = get_chat_completion_request_system_message()
    system_instruction = get_chat_completion_request_system_instructions()
    
    new_messages.append(system_message)
    new_messages.append(system_instruction)
    for message in messages:
        new_messages.append(message)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    }
    json_data = {"model": model, "messages": new_messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        print("\n\nInput json: chat_completion_request\n\n", json_data)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        print("\n\nOutput Response: chat_completion_request\n\n", response.json())
        return json.loads(response.text)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")

        return e


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def format_sql_response(sql_response: str, model: str = config.GPT_MODEL) -> str:
    messages = get_format_sql_response_messages(sql_response)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    }
    json_data = {"model": model, "messages": messages}
    try:
        print("\n\nInput json: format_sql_response\n\n", json_data)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        print("\n\nOutput Response: format_sql_response\n\n", response.json())
        return json.loads(response.text)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")

        return e
