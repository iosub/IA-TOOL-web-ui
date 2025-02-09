import os
import pdb
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

import sys

sys.path.append(".")

@dataclass
class LLMConfig:
    provider: str
    model_name: str
    temperature: float = 0.8
    base_url: str = None
    api_key: str = None

def create_message_content(text, image_path=None):
    content = [{"type": "text", "text": text}]

    if image_path:
        from src.utils import utils
        image_data = utils.encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })

    return content

def get_env_value(key, provider):
    env_mappings = {
        "openai": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_ENDPOINT"},
        "azure_openai": {"api_key": "AZURE_OPENAI_API_KEY", "base_url": "AZURE_OPENAI_ENDPOINT"},
        "google": {"api_key": "GOOGLE_API_KEY"},
        "deepseek": {"api_key": "DEEPSEEK_API_KEY", "base_url": "DEEPSEEK_ENDPOINT"},
        "mistral": {"api_key": "MISTRAL_API_KEY", "base_url": "MISTRAL_ENDPOINT"},
    }

    if provider in env_mappings and key in env_mappings[provider]:
        return os.getenv(env_mappings[provider][key], "")
    return ""

def test_llm(config, query, image_path=None, system_message=None):
    from src.utils import utils

    # Special handling for Ollama-based models
    if config.provider == "ollama":
        if "deepseek-r1" in config.model_name:
            from src.utils.llm import DeepSeekR1ChatOllama
            llm = DeepSeekR1ChatOllama(model=config.model_name)
        else:
            llm = ChatOllama(model=config.model_name)

        ai_msg = llm.invoke(query)
        print(ai_msg.content)
        if "deepseek-r1" in config.model_name:
            pdb.set_trace()
        return

    # For other providers, use the standard configuration
    llm = utils.get_llm_model(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("OPENAI_ENDPOINT", ""),
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    # Prepare messages for non-Ollama models
    messages = []
    if system_message:
        messages.append(SystemMessage(content=create_message_content(system_message)))
    messages.append(HumanMessage(content=create_message_content(query, image_path)))
    ai_msg = llm.invoke(messages)

    # Handle different response types
    if hasattr(ai_msg, "reasoning_content"):
        print(ai_msg.reasoning_content)
    print(ai_msg.content)

    if config.provider == "deepseek" and "deepseek-reasoner" in config.model_name:
        print(llm.model_name)
        pdb.set_trace()

def test_openai_model():
    config = LLMConfig(provider="openai", model_name="gpt-4o")
    test_llm(config, "Describe this image", "assets/examples/test.png")

def test_google_model():
    # Enable your API key first if you haven't: https://ai.google.dev/palm_docs/oauth_quickstart
    config = LLMConfig(provider="google", model_name="gemini-2.0-flash-exp")
    test_llm(config, "Describe this image", "assets/examples/test.png")

def test_azure_openai_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="azure_openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )
    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_deepseek_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.8,
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": "who are you?"}
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_deepseek_r1_model():
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",
        model_name="deepseek-reasoner",
        temperature=0.8,
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    messages = []
    sys_message = SystemMessage(
        content=[{"type": "text", "text": "you are a helpful AI assistant"}]
    )
    messages.append(sys_message)
    user_message = HumanMessage(
        content=[
            {"type": "text", "text": "9.11 and 9.8, which is greater?"}
        ]
    )
    messages.append(user_message)
    ai_msg = llm.invoke(messages)
    print(ai_msg.reasoning_content)
    print(ai_msg.content)
    print(llm.model_name)
    pdb.set_trace()


def test_ollama_model():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:7b")
    ai_msg = llm.invoke("Sing a ballad of LangChain.")
    print(ai_msg.content)
    
def test_deepseek_r1_ollama_model():
    from src.utils.llm import DeepSeekR1ChatOllama

    llm = DeepSeekR1ChatOllama(model="deepseek-r1:14b")
    ai_msg = llm.invoke("how many r in strawberry?")
    print(ai_msg.content)
    pdb.set_trace()


if __name__ == '__main__':
    # test_openai_model()
    # test_google_model()
    # test_azure_openai_model()
    #test_deepseek_model()
    # test_ollama_model()
    test_deepseek_r1_model()
    # test_deepseek_r1_ollama_model()
    # test_mistral_model()
