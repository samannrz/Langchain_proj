from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """Your job is to come up with a classic dish from the area that the users suggest.
{location}
YOUR RESPONSE:
"""
prompt_template = ChatPromptTemplate.from_template(template=template)

# chain 1
location_chain = LLMChain(llm=ChatOpenAI(),prompt=prompt_template,output_key='meal')
##############
template = """ given a {meal} give a short recipe how to make that dish at home.
YOUR RESPONSE:
"""
prompt_template=ChatPromptTemplate.from_template(template=template)
# chain 2
meal_chain = LLMChain(llm=ChatOpenAI(),prompt=prompt_template,output_key='recipe')
################
template = """ given a {recipe} estimate how much time I need to cook it.
YOUR RESPONSE:
"""
prompt_template=ChatPromptTemplate.from_template(template=template)
# chain 3
time_chain = LLMChain(llm=ChatOpenAI(),prompt=prompt_template,output_key='time')
################
overal_chain = SequentialChain(chains= [location_chain,meal_chain,time_chain],
                               input_variables=['location'],
                               output_variables=['meal','recipe','time'],
                               verbose=True)
result = overal_chain.invoke(input={'location':'Iran'})
# print(result['meal'])
# print(result['recipe'])
# print(result['time'])