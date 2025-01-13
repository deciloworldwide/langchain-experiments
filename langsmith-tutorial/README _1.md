# LangSmith Correctness Evaluation

This project demonstrates how to use LangSmith to evaluate the correctness of AI responses against ground truth answers. It includes synthetic dataset handling and LLM evaluation.
Annotation queue creation for low-scoring responses is currently handled manually in the dashboard.

## Requirements

- Python 3.11 or higher
- LangChain library
- OpenAI API key
- LangSmith API key

## Installation

### 1. Clone the repository


### 2. Create a Python environment

Using `venv`:
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

Or using `conda`:
```bash
conda create -n langsmith-env python=3.8
conda activate langsmith-env
```

### 3. Install required packages
```bash
pip install langchain langsmith openai pandas python-dotenv langchain-openai
```

### 4. Set up environment variables

Create a `.env` file in the root directory with the following content:
```env
OPENAI_API_KEY="your_openai_api_key_here"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_TRACING_V2="true"
```

## Dataset Preparation

1. Use the provided `agaile_synthetic_dataset.json` template
2. Structure includes:
   - `query`: The question
   - `expected_answer`: Ground truth answer
   - `chatbot_answer`: Answer to evaluate (initially same as expected_answer)
   - `metadata`: Additional information

Example entry:
```json
{
    "query": "What is Agaile's mission?",
    "expected_answer": "Agaile's mission is to...",
    "chatbot_answer": "Agaile's mission is to...",
    "metadata": {"type": "company_overview", "domain": "agaile"}
}
```

## Running the Evaluation

1. Make sure your environment is activated
2. Run the script:
```bash
python langsmith_mvp_eval2_correctness.py
```

The script will:
- Load and process the dataset
- Run evaluations using LangSmith


## Customization


- Modify the chatbot answers in the JSON file to test different response qualities
- Customize the evaluation criteria in the LangChainStringEvaluator configuration

## Project Structure
```
langsmith-tutorial/
├── src/
│   ├── langsmith_mvp_eval2_correctness.py
│   └── agaile_synthetic_dataset.json
├── .env
└── README_1.md
```

## LangSmith Dashboard

After running the evaluation:
1. Visit the [LangSmith Dashboard](https://smith.langchain.com/)
2. Check the "Datasets" section for:
   - Original evaluation dataset
3. Add traces with inccorrect answers to annotation queue
4. Review annotation queue and feedback
5. Review evaluation results and feedback

## Troubleshooting

- Ensure all API keys are correctly set in `.env`
- Verify the JSON file path matches your project structure
- Check that the dataset format matches the expected structure
- Make sure all required packages are installed

## Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
