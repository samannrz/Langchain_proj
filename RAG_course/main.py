from langchain_core.prompts import PromptTemplate

my_template = PromptTemplate.from_template("tell me a {adj} joke about {context}")
my_prompt = my_template.format(adj= 'childish', context = 'animals')
print(my_prompt)