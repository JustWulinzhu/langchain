import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# 定义查询天气的天气的工具
def get_weather(city: str) -> str :
    """测试"""
    return f"{city} 今天是暴雨天 加 大风。"

# 定义计算器的工具
def calculate(expression: str) -> int :
    """计算器"""
    return eval(expression)

# 初始化大模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL")
)

# 创建智能体agent
# 使用 create_agent 将大模型和工具组合起来
agent = create_agent(
    model = llm,
    tools = [get_weather, calculate],
    system_prompt = "你是一个智能体助手"
)

# 调用智能体并且传入问题
response = agent.invoke({
    "messages" : [{"role": "user", "content": "北京天气怎么样？如果天气不好的话，帮我算一下 25 x 3"}]
})

print(response["messages"][-1].content)