from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_openai import OpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, SearchApiAPIWrapper

from crewai import Agent, Task, Crew, Process

#import openai

from typing import List
from data import GoogleTrendsData
from pydantic import BaseModel, Field

import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json



'''
TO-DO


'''



class YoutubeChannelManager:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=self.chrome_options)

    # We can keep this as the SerpAPI free plan have 100 free api calls per month. when we are out of those 100 calls we can use this one.
    # keyword_list = ["Langchain", "Agents", "Agent Swarms", "Autogen"]
    # google_data = GoogleTrendsData()
    # df = google_data.fetch_data(keyword_list)
    #
    # print(type(df))
    # agent = create_pandas_dataframe_agent(
    #     ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo"),
    #     df,
    #     verbose = True,
    #     agent_type = AgentType.OPENAI_FUNCTIONS
    # )
    #
    # agent.run("What are the most popular 50 search topics? You can get the popularity from top query value column")
    def scrape_youtube_channel(self, website):
        try:
            self.driver.get(website)
            self.driver.implicitly_wait(10)
            contents_div = self.driver.find_element(By.ID, "contents")
            content = contents_div.text
            print(content)
            return content

        finally:
            self.driver.quit()




    ## TOOLS


    def run_crew(self, topic, url):
        content_generator_tool = Tool(
            name="Youtube Channel Scraper",
            func=self.scrape_youtube_channel,
            description="Useful for trying to figure out which content to create in the channel"
        )
        # trend_tool = Tool(
        #     name="Trend Data Finder",
        #     func=GoogleTrendsData.fetch_data(topic),
        #     description="Useful for finding top and rising related queries based on the keyword. Rising related queries are improving their popularity, and Top related queries are popular already.",
        # )

        # duckSearch = DuckDuckGoSearchResults()
        #
        # duck_tool = Tool(
        #     name="DuckDuckSearch",
        #     description="Search the web for finding informations about the given topic to create an outline.",
        #     func = duckSearch.run,
        # )
        search = SearchApiAPIWrapper()
        duck_tool = Tool(
            name="Search",
            description="Search the web for finding informations about the given topic.",
            func=search.run
        )

        google_trend_tool = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())

        ## AGENTS
        related_query_agent = Agent(
            role="Search Engine Optimizer",
            goal="Finding a related query based on the keyword",
            backstory="An expert on Search Engine Optimization which is basically analyzing the Google Trends",
            tools = [google_trend_tool],
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7) # with the new update we need to specify the llm model we use otherwise it uses gpt4
        )

        content_decider_agent = Agent(
            role="Youtube Channel Content Director",
            goal="Figuring out what topics would be interesting for your users.",
            backstory="An expert on working with social media influences and analyzer of Youtube SEO.",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        )

        content_creation_agent = Agent(
            role="Youtube Content Creation Manager",
            goal="Creating interesting titles and outlines by using the tool that is given to find related information on the web.",
            backstory="An expert on Youtube Video Content Creation",
            tools=[duck_tool],
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        )

        result_agent = Agent(
            role="Youtube Video Creator",
            goal="Creating a JSON file for video for the given outline, title and other information about the topic",
            backstory="An expert Director",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        )

        ## TASKS
        related_query_task = Task(
            description=f"Find 10 related queries for {topic} keyword. Pick the related queries from the tool that is given to you,"
                        "When you are picking the related queries, you need to find rising related queries that are not in the top related queries.",
            agent=related_query_agent,
            expected_output=f"10 Related queries for {topic} keyword."
        )

        # need to update this
        content_decider_task = Task(
            description=f"Look at the topics in youtube channel with url this url {url}, "
                        "you can use your tool for scraping the website, you also have the context for related queries which are potential topics"
                        "from the 10 related queries, look at the potential topic from those queries that isn't published in the youtube channel so far and pick one topic from those.",
            agent=content_decider_agent,
            tools = [content_generator_tool],
            context = [related_query_task],
            expected_output="One potential topic for the youtube channel that wasn't in the youtube channel before."
        )

        content_creation_task = Task(
            description = "Create a title and the outline that is interesting for the topic that you have as context. "
                          "When creating the outline use the tool that is given to you to find the related information on the web. ",
            agent = content_creation_agent,
            context = [content_decider_task],
            expected_output = "Video outline and the bullet points for the video."
        )



        # description doesn't seem to be working well, we need to find a way to parse everything into the json right away and not go through any analyze
        # we can even have only one agent that creates the outline, title and parse it as json file to make it easier for the crew.
        result_task = Task(
            description="Create a script for the video for the given title and outline that was passed to you as a context. "
                        "when creating the video, spend same amount of time in each bullet point in the outline"
                        " when outputing the script the video length is evenly distributed among the topics in outline."
                        ,
            agent = result_agent,
            context=[content_creation_task],
            expected_output="Script in one paragraph for desired video."
        )


        ## CREW
        crew = Crew(
            agents=[related_query_agent, content_decider_agent, content_creation_agent, result_agent],
            tasks=[related_query_task,content_decider_task, content_creation_task, result_task],
            verbose=2,
            process = Process.sequential # this ensures that tasks are executed one after each other
        )


        result = crew.kickoff()
        self.save_txt(result, "script")
        return result

    def save_txt(self, output, filename):
        with open(filename, "w") as txt_file:
            txt_file.write(output)

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["SEARCHAPI_API_KEY"] = ""
    os.environ["SERPAPI_API_KEY"] = ""

    manager = YoutubeChannelManager()
    topic = "langchain"
    url = "https://www.youtube.com/@LangChain/videos"
    manager.run_crew(topic, url)
