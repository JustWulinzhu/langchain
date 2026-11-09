from dotenv import load_dotenv
from langchain_openai import  ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEP_SEEKER_API_KEY")
)

response = llm.invoke("简单介绍下你自己")
print(response)