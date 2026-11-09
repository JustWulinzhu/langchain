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

prompts = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的翻译官，请将用户的输入从{source_lang} 翻译成 {target_lang}。"),
    ("human", "{text}")
])

chain = prompts | llm | StrOutputParser()

response = chain.invoke({
    "source_lang": "中文",
    "target_lang": "英文",
    "text": "你好"
})

# 也可以去掉StrOutputParser() 取response.content字段 是相同的
print(response)