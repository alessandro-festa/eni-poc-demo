import os
import mlflow
from mlflow.entities import Document
import openai

# Set your MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:30344/"
os.environ["OPENAI_API_KEY"] = "ollama" 
# Or directly in code
mlflow.set_tracking_uri("http://localhost:30344/")
# This will create a new experiment called "GenAI Evaluation Quickstart" and set it as active
mlflow.set_experiment("GenAI Evaluation Quickstart")

client = openai.OpenAI(
    base_url = 'http://localhost:31434/v1',
    api_key='ollama', # required, but unused
)
# mlflow.openai.autolog()  # Enable automatic tracing for OpenAI calls


def qa_predict_fn(question: str) -> str:
    """Simple Q&A prediction function using OpenAI"""
    response = client.chat.completions.create(
        model="qwen3:0.6b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
            },
            {"role": "user", "content": question},
        ]
    )
    return response.choices[0].message.content

# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "The capital of France is Paris."},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]

from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines


@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 50


scorers = [
    Correctness(model="ollama:/phi4-mini-reasoning:latest"),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=qa_predict_fn,
    scorers=scorers,
)