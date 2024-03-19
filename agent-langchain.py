import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool, DuckDuckGoSearchRun

import pandas as pd
from pytrends.request import TrendReq

from typing import List, Tuple
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyAraQxg8Jg3o-iCGl9wstPvp-82N_13FFk"



## tools
@tool
def search_youtube(url: str) -> str:
    """Scrape the youtube channels for given url"""
    try:
        self.driver.get(url)
        self.driver.implicitly_wait(10)
        contents_div = self.driver.find_element(By.ID, "contents")
        content = contents_div.text
        return content
    finally:
        self.driver.quit()

@tool
def fetch_data(keywords: str):
    """Fetch the google trend data for given keyword"""
    # Build model
    pytrend = TrendReq()

    # Provide your search terms

    pytrend.build_payload(kw_list=keywords)

    # Get related queries
    related_queries = pytrend.related_queries()
    related_queries_values = related_queries.values()

    # Build lists dataframes
    top = list(related_queries.values())[0]['top']
    rising = list(related_queries.values())[0]['rising']

    # Convert lists to dataframes
    dftop = pd.DataFrame(top)
    dfrising = pd.DataFrame(rising)

    # Join two data frames
    allqueries = pd.concat([dftop, dfrising], axis=1)

    # Function to change duplicates
    cols = pd.Series(allqueries.columns)
    for dup in allqueries.columns[allqueries.columns.duplicated(keep=False)]:
        cols[allqueries.columns.get_loc(dup)] = ([dup + '.' + str(d_idx)
                                                   if d_idx != 0
                                                   else dup
                                                   for d_idx in range(allqueries.columns.get_loc(dup).sum())]
                                                  )
    allqueries.columns = cols

    # Rename to proper names
    allqueries.rename({'query': 'top query', 'value': 'top query value', 'query.1': 'related query', 'value.1': 'related query value'},
                      axis=1, inplace=True)

    # Check your dataset
    #print(allqueries.head(100))
    return allqueries

#search = DuckDuckGoSearchRun()
tools = [search, fetch_data, search_youtube]
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = genai.GenerativeModel(model_name="gemini-pro", tools=tools)
prompt_parts = [
    "Find 10 related queries for Langchain keyword. Pick the related queries from the tool that is given to you,"
                        "When you are picking the related queries, you need to find rising related queries that are not in the top related queries."
]
response = llm.generate_content(prompt_parts)
print(response.text)


# 2nd trial
# llm = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )
# llm_tools = llm.bind(
#     functions=tools
# )
#
#
# def _format_chat_history(chat_history: List[Tuple[str,str]]):
#     buffer = []
#     for human, ai in chat_history:
#         buffer.append(HumanMessage(content=human))
#         buffer.append(AIMessage(content=ai))
#     return buffer
#
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#         "agent_scratchpad": lambda x: format_to_openai_function_messages(
#             x["intermediate_steps"]
#         ),
#     }
#     | prompt
#     | llm_tools
#     | OpenAIFunctionsAgentOutputParser()
# )
#
# class AgentInput(BaseModel):
#     input: str
#     chat_history: List[Tuple[str,str]] = Field(
#         ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
#     )
#
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(input_type=AgentInput)

# 3rd trial
#

#
# chain = prompt | llm
# question = ("Find 10 related queries for Langchain keyword."
#             " When you are picking the related queries, you need to find rising related queries that are not in the top related queries.")
# print(chain.invoke({"question": question}))


