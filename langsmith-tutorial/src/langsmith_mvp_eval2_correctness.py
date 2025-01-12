# --------------------------------------------------------------
# Import Modules
# --------------------------------------------------------------

import os
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------
# Load API Keys From the .env File
# --------------------------------------------------------------

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------
# LangSmith Quick Start
# Load the LangSmith Client and Test Run
# --------------------------------------------------------------

client = Client()
openai_client = wrap_openai(OpenAI())

# --------------------------------------------------------------
# Evaluation Quick Start
# 1. Create a Dataset (Only Inputs, No Output)
# --------------------------------------------------------------

example_inputs = [
    "a rap battle between Atticus Finch and Cicero",
    "a rap battle between Barbie and Oppenheimer",
    "a Pythonic rap battle between two swallows: one European and one African",
    "a rap battle between Aubrey Plaza and Stephen Colbert",
]

dataset = client.create_dataset(
    "mvp_test_dataset",
    description="mvp_test_dataset."
)

for input_prompt in example_inputs:
    # Each example must be unique and have inputs defined.
    # Outputs are optional
    client.create_example(
        inputs={"question": input_prompt},
        dataset_id=dataset.id
    )

# --------------------------------------------------------------
# 2. Evaluate Datasets with LLM
# --------------------------------------------------------------

def chatbot(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": inputs["question"]}]
    )
    return {"answer": response.choices[0].message.content}

# Define custom evaluator
def is_creative(outputs: dict, reference_outputs: dict = None) -> bool:
    return len(outputs["answer"]) > 100

# Create LLM for evaluation
eval_llm = ChatOpenAI(model="gpt-4", temperature=0)

# Use criteria evaluator instead of cot_qa since we don't have reference outputs
criteria_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "creativity": "Is this submission creative and imaginative?",
            "coherence": "Is this submission coherent and well-structured?"
        }
    }
)

# Run evaluation
experiment_results = client.evaluate(
    chatbot,
    data=dataset,
    evaluators=[
        is_creative,
        criteria_evaluator
    ],
    experiment_prefix="mvp_test_eval",
    max_concurrency=4
)

