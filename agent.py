from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_openai import OpenAI
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from crewai import Agent, Task, Crew

from data import GoogleTrendsData

import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options




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
            return content
        finally:
            self.driver.quit()




    ## TOOLS
    #google_trend_tool = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())

    def run_crew(self, topic, url):
        content_generator_tool = Tool(
            name="Youtube Channel Scraper",
            func=self.scrape_youtube_channel,
            description="Useful for trying to figure out which content to create in the channel"
        )
        trend_tool = Tool(
            name="Trend Data Finder",
            func=GoogleTrendsData.fetch_data,
            description="Useful for finding top and rising related queries based on the keyword. Rising related queries are improving their popularity, and Top related queries are popular already.",

        )

        duckSearch = DuckDuckGoSearchResults()
        duck_tool = Tool(
            name="DuckDuckSearch",
            description="Search the web for finding informations about the given topic to create an outline.",
            func = duckSearch.run,
        )


        ## AGENTS
        agent = Agent(
            role="Search Engine Optimizer",
            goal="Finding a related query based on the keyword",
            backstory="An expert on Search Engine Optimization which is basically analyzing the Google Trends",
            tools = [trend_tool]
        )
        agent2 = Agent(
            role="Writer and Digital Content Specialist",
            goal="Creating interesting titles for social media",
            backstory="An expert on working with social media influencers in google and youtube"
        )
        agent3 = Agent(
            role="Youtube Channel Content Director",
            goal="Figuring out what topics would be interesting for your users.",
            backstory="An expert on working with social media influences and analyzer of Youtube SEO."
        )
        agent4 = Agent(
            role="Youtube Video Content Planner",
            goal="Creating an outline for the video from the title that is given, find the related information on the web by using your tool",
            backstory="An expert on Youtube Video Content Creation.",
            tools=[duck_tool],

        )

        ## TASKS
        undiscovered_query_task = Task(
            description=f"Find 10 related queries for {topic} keyword. Pick the related queries from the tool that is given to you,"
                        "When you are picking the related queries, you need to find rising related queries that are not in the top related queries.",
            agent=agent
        )
        # need to update this
        youtube_content_suggestion_task = Task(
            description=f"Look at the topics in youtube channel with url this url {url}, "
                        "you can use your tool for scraping the website, you also have the context for related queries which are potential topics"
                        "from the 10 related queries, look at the potential topics from those queries that isn't published in the youtube channel so far.",
            agent=agent3,
            tools = [content_generator_tool]
        )
        title_generation_task = Task(
            description="Create a title that is interesting from the 10 related queries that you have from first task.",
            agent=agent2
        )
        outline_generation_task = Task(
            description="Create an outline for the potential topics that you want to share in your youtube channel,"
                        "suggestions of topics are the outputs of {agent3} and look at the web with your tool to create an outline"
                        "of what you can talk in your youtube video."
        )


        ## CREW
        crew = Crew(
            agents=[agent,agent2, agent3, agent4],
            tasks=[undiscovered_query_task,title_generation_task, youtube_content_suggestion_task, outline_generation_task],
            verbose=2
        )

        result = crew.kickoff()
        print(result)

        #return result

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""

    # os.environ["SERPAPI_API_KEY"] = ""

    manager = YoutubeChannelManager()
    topic = "langchain"
    url = "https://www.youtube.com/@LangChain/videos"
    manager.run_crew(topic, url)