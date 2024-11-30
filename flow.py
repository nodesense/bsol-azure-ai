# https://microsoft.github.io/promptflow/how-to-guides/quick-start.html
# https://github.com/microsoft/promptflow/blob/main/examples/flex-flows/chat-minimal/flow.py
# https://github.com/microsoft/promptflow/tree/main/examples/flex-flows/chat-minimal

import os

from dotenv import load_dotenv
from pathlib import Path
from promptflow.tracing import trace
from promptflow.core import Prompty
import os
BASE_DIR = Path(__file__).absolute().parent

# Install the latest stable version
# pip install promptflow --upgrade

# Install the latest stable version
# pip install promptflow[azure] --upgrade


# pf --version

# winget install -e --id Microsoft.Promptflow

# pip install promptflow-tracing

# RUN 
# pf flow test --flow flow:chat --inputs question="What's the capital of France?"


@trace
def chat(question: str = "What's the capital of France?") -> str:
    """Flow entry function."""


    # AZURE_OPENAI_ENDPOINT="https://gks-ai-hub-service.openai.azure.com/"
    # AZURE_OPENAI_API_KEY="3MdceM28EeKjdtOfjPC1OAy7zCzRQjsCrIgVqeXxGhGe5rPPGIbHJQQJ99AKACHYHv6XJ3w3AAAAACOGneje"


    #if "OPENAI_API_KEY" not in os.environ and "AZURE_OPENAI_API_KEY" not in os.environ:
        # load environment variables from .env file
    load_dotenv()
    print (os.environ)

    prompty = Prompty.load(source=BASE_DIR / "data" / "prompts" / "mini-chat.prompty")
    output = prompty(question=question)
    return output


if __name__ == "__main__":
    from promptflow.tracing import start_trace

    start_trace()

    result = chat("What's the capital of France?")
    print(result)