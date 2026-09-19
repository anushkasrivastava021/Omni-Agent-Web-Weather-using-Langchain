import streamlit as st
import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# 1. Environment Setup
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# 2. Tool Definitions
@tool
def get_weather_data(city: str) -> str:
    """Fetch weather data for a given city."""
    url = (
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

# 3. Cache the Agent Initialization
# This prevents the app from rebuilding the LangChain setup every time the user types a message.
@st.cache_resource
def initialize_agent():
    search_tool = TavilySearchResults(max_results=2, search_depth="basic")
    tools = [search_tool, get_weather_data]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest", 
        temperature=0
    )
    
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    
    # Return the executable agent
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

agent_executor = initialize_agent()

# 4. Streamlit User Interface
st.title("🌍 Omni-Agent: Web & Weather")
st.caption("Powered by Gemini, Tavily, and Weatherstack")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Ask me to search the web or check the weather anywhere in the world."}
    ]

# Display historical chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Handle User Input
if prompt := st.chat_input("e.g., What is the capital of India and its current weather?"):
    # Render user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render assistant response with a loading spinner and thought visualizer
    with st.chat_message("assistant"):
        # The callback handler visualizes the ReAct loop (Action/Observation) in the UI
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            # Execute the agent
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            final_output = response["output"]
            
            # Display the final answer
            st.write(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")