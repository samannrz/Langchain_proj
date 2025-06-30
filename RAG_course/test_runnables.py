from langchain.schema.runnable import RunnableParallel
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI()

parallel_chain = RunnableParallel({
    "summary": ChatPromptTemplate.from_template("Summarize this text: {text}") | llm,
    "translation": ChatPromptTemplate.from_template("Translate this text to French: {text}") | llm,
    "sentiment": ChatPromptTemplate.from_template(
        "What is the sentiment of this sentence? Positive, Negative or Neutral?: {text}"
    ) | llm,
})

result = parallel_chain.invoke({
    "text": "I really enjoyed the conference today. The speakers were informative."
})

print(result['sentiment'].content)
