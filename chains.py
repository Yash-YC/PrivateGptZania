from langchain.prompts import (
    ChatPromptTemplate,
)
from config import config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# initialize llm chat model
llm = ChatOpenAI(openai_api_key=config.base_config.OPENAI_API_KEY,
                 model_name=config.base_config.LLM)


def qa_(context: str = None, question: str = None):
    input_data = {
        "context": context,
        "question": question
    }
    prompt = ChatPromptTemplate.from_template(
        config.base_config.PROMPT_TEMPLATE)
    formatted_prompt = prompt.invoke(input_data)
    response = llm.invoke(formatted_prompt)
    return StrOutputParser().parse(response)
