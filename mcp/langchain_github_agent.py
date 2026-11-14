import os
import asyncio
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.mcp import MCPAdapter
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL")
)

async def main() :
    async with MCPAdapter({
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
                }
            }
        }
    }) as adapter :
        tools = await adapter.list_tools()
        for tool in tools :
            print(tool.name)

        agent = create_agent(
            model = llm,
            tools = tools,
            system_prompt = "你是一个专业的 GitHub 开发助手，当用户问你github相关的问题时要查询工具回答问题，禁止编造数据"
        )

        result = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": "帮我查一下 langchain-ai/langchain 仓库最近的一个 issue 是什么？"
            }]
        })

        print("ai的回答：")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    # 启动异步引擎
    asyncio.run(main())