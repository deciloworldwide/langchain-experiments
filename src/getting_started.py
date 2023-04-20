from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
from langchain.utilities import SerpAPIWrapper

# Load environment variables
load_dotenv(find_dotenv())

# --------------------------------------------------------------
# Models
# --------------------------------------------------------------

llm = OpenAI(model_name="text-davinci-003")
prompt = "One sentence poem about python"
print(llm(prompt))


# --------------------------------------------------------------
# Prompts
# --------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

prompt.format(product="toothbrushes")

# --------------------------------------------------------------
# Chains
# --------------------------------------------------------------

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

chain.run("AI Aquisition Agents")

# --------------------------------------------------------------
# Agents
# --------------------------------------------------------------

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["wikipedia", "google-search", "llm-math"], llm=llm)
get_all_tool_names()

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Now let's test it out!
agent.run("In what year was python released and who is the original creator?")


# --------------------------------------------------------------
# Memory
# --------------------------------------------------------------

llm = OpenAI(temperature=0.7)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(
    input="I'm doing well! Just having a conversation with an AI."
)
print(output)

params = {
    "engine": "youtube",
    "gl": "nl",
}
search = SerpAPIWrapper(params=params)

result = search.run("auto-gpt tutorial")

print(result)
