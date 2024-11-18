from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers.json import SimpleJsonOutputParser


ollama_models_list = [
    "llama3.2:latest",
    "mistral:latest",
    "mistrallite:latest",
    "falcon:latest",
]

template = """
Summarize the following conversation between multiple users:

{conversation}

Summary:
"""

prompt = PromptTemplate.from_template(template)

conversation = """
User1: Hi, how are you?
User2: I'm good, thanks! How about you?
User1: I'm doing well. Did you finish the project?
User2: Yes, I submitted it yesterday.
User1: Great! Let's discuss the next steps.
"""
for model in ollama_models_list:
    llm = OllamaLLM(model=model)
    chain = prompt | llm 
    print(f"Model: {model}")
    output = chain.invoke(input={'conversation':conversation})
    print(output)