# --------------------------------------------------------------
# Import Modules
# --------------------------------------------------------------

import os
import json
import pandas as pd
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
# Load and Transform Dataset
# --------------------------------------------------------------
# Load JSON file with absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'agaile_synthetic_dataset.json')

with open(json_path, 'r') as file:
    data = json.load(file)

# Convert to DataFrame with correct mapping for LangSmith
df = pd.DataFrame(data)
df = df.rename(columns={
    'query': 'question',
    'expected_answer': 'answer'
})

# Create LangSmith dataset
client = Client()
dataset = client.create_dataset(
    "agaile_qa_dataset",
    description="Agaile Q&A evaluation dataset with ground truth answers"
)

# Add examples to dataset with correct structure
for _, row in df.iterrows():
    client.create_example(
        inputs={"question": row['question']},
        outputs={"answer": row['answer']},  # This contains the ground truth/reference answer
        dataset_id=dataset.id
    )

# --------------------------------------------------------------
# Define Chatbot and Evaluator
# --------------------------------------------------------------
def chatbot(inputs: dict) -> dict:
    # For testing, we'll just return the ground truth
    question = inputs["question"]
    # Find the matching ground truth answer from our dataframe
    answer = df[df['question'] == question]['answer'].iloc[0]
    return {"answer": answer}

# Create LLM for evaluation
eval_llm = ChatOpenAI(model="gpt-4", temperature=0)

# Configure QA evaluator with correct data preparation
qa_evaluator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": eval_llm},
    prepare_data=lambda run, example: {
        "input": example.inputs["question"],
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"]  # Access reference from example.outputs
    }
)

# --------------------------------------------------------------
# Run Evaluation
# --------------------------------------------------------------
experiment_results = client.evaluate(
    chatbot,
    data=dataset,
    evaluators=[qa_evaluator],
    experiment_prefix="agaile_qa_eval",
    max_concurrency=4
)

