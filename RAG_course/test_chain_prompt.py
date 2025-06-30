from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()

template = """Tell me a {adjective} joke about {content}"""
my_prompt = ChatPromptTemplate.from_template(template)

def format_prompt(variables):
    return my_prompt.format(**variables)

joke_chain = (
    RunnableLambda(format_prompt) | llm | StrOutputParser()
)

result = joke_chain.invoke({"adjective":"childish","content":"animals"})
print(result)