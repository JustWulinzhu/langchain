import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# 定义一个工具，这是一个普通的 Python 函数，用来模拟获取某个城市的天气。
# 参数 city: str 和类型注解非常重要，大模型会通过这些类型和函数名、Docstring（文档字符串 """..."""）来理解这个工具是干什么的、需要传入什么参数
# 如果大模型收到了用户问“北京天气怎么样？”，它会扫描所有工具的文档字符串，发现 Get weather for a given city. 和查询意图匹配，从而决定调用 get_weather
# 当前函数是写死的模拟返回（返回“某市今天晴，25°C”），在实际开发中，这里可以替换为真实的第三方天气 API 请求
def get_weather(city: str) -> str :
    """ 获取某个城市的天气 """
    return f"{city} 今天龙卷风，暴雨。"

# 初始化大模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)

# 创建智能体
# 使用 create_agent 将大模型和工具组合起来，生成一个智能体实例
# model=llm：告诉 Agent 使用上面配置好的 DeepSeek 模型作为“大脑”
# tools=[get_weather]：赋予 Agent 能够调用的工具列表（这里把刚才写的 get_weather 函数传进去了）
# 另外 tools是一个列表，可以传很多方法，大模型在选择工具的时候会选取与问题匹配的工具进行使用
# system_prompt="你是一个智能体助手。"：给 Agent 设置系统提示词，规范它的基本行为和人设
agent = create_agent(
    model = llm,
    tools = [get_weather],
    system_prompt = "你是一个智能体助手！"
)

# 运行智能体并传入问题
# 大模型内部执行流程：
# 1、DeepSeek 收到“北京天气怎么样？”这个问题。
# 2、它发现自己配有一个叫 get_weather 的工具，且用户问的是北京的天气，于是它决定调用工具，传入参数 city="北京"。
# 3、代码在本地自动运行了 get_weather("北京")，得到了返回结果 "北京 今天晴，25°C。"。
# 4、Agent 把这个结果再次发给大模型。
# 5、大模型结合工具返回的结果，生成最终的自然语言回复，并封装在 response 中。
response = agent.invoke({
    "messages" : [{"role": "user", "content": "北京今天天气怎么样？"}]
})

print(response["messages"][-1].content)