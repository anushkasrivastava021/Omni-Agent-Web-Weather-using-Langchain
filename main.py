import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

from langchain.agents import create_react_agent, AgentExecutor

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
search_tool = TavilySearchResults(
    max_results=2,
    search_depth="basic"
)
@tool
def get_weather_data(city: str) -> str:
    # Placeholder function to simulate fetching weather data
    """Fetch weather data for a given city."""

    url=(
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_API_KEY}&query={city}"
    )
    response = requests.get(url)
    data = response.json()
    if "current" not in data:
        return f"Could not retrieve weather data for {city}."
    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%\n"
    )
result=search_tool.invoke("latest news on AI research")
result
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
response=llm.invoke("What year is it?")
response
prompt=hub.pull("hwchase17/react")
tools = [search_tool,get_weather_data]
agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
response=agent_executor.invoke({
    "input": (
        "What is the capital of India."
        "and what is the current weather there?"
    )
})
print(response["output"])