from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_openai import OpenAI
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from crewai import Agent, Task, Crew, Process

from data import GoogleTrendsData
from pydantic import BaseModel

import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


'''
TO-DO
JSON PARSER AGENT
description doesn't seem to be working well, we need to find a way to parse everything into the json right away and not go through any analyze
we can even have only one agent that creates the outline, title and parse it as json file to make it easier for the crew.

trend_tool gave an error for the tool that is used



'''

# to be sure we need to make sure the schema of the desired json file
class VideoOutlineOutput(BaseModel):
    outline: str
    suggested_topics: list






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
        # this tool gave an error for the tool that is used
        related_query_agent = Agent(
            role="Search Engine Optimizer",
            goal="Finding a related query based on the keyword",
            backstory="An expert on Search Engine Optimization which is basically analyzing the Google Trends",
            tools = [trend_tool],
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7) # with the new update we need to specify the llm model we use otherwise it uses gpt4
        )
        title_generator_agent = Agent(
            role="Writer and Digital Content Specialist",
            goal="Creating interesting titles for social media",
            backstory="An expert on working with social media influencers in google and youtube",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        )
        
        channel_content_agent = Agent(
            role="Youtube Channel Content Director",
            goal="Figuring out what topics would be interesting for your users.",
            backstory="An expert on working with social media influences and analyzer of Youtube SEO.",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        )

        video_outline_agent = Agent(
            role="Youtube Video Content Planner",
            goal="Creating an outline for the video from the title that is given, find the related information on the web by using your tool",
            backstory="An expert on Youtube Video Content Creation.",
            tools=[duck_tool],
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        )

        result_agent = Agent(
            role="Youtube Video Creator",
            goal="Creating a video for the given outline, title and other information about the topic",
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
        channel_content_task = Task(
            description=f"Look at the topics in youtube channel with url this url {url}, "
                        "you can use your tool for scraping the website, you also have the context for related queries which are potential topics"
                        "from the 10 related queries, look at the potential topics from those queries that isn't published in the youtube channel so far and pick one topic from those.",
            agent=channel_content_agent,
            tools = [content_generator_tool],
            context = [related_query_task],
            expected_output="One potential topic for the youtube channel."
        )

        title_generation_task = Task(
            description="Create a title that is interesting from the 10 related queries that you have from first task.",
            agent=title_generator_agent,
            context=[channel_content_task],
            expected_output="Title that is interesting."
        )

        video_outline_task = Task(
            description="Create an outline for the potential topics that you want to share in your youtube channel,"
                        f"suggestions of topics are the outputs of {video_outline_agent} and look at the web with your tool to create an outline"
                        "of what you can talk in your youtube video.",
            agent=video_outline_agent,
            context=[channel_content_task],
            expected_output="Outline for the potential topics"
        )

        # description doesn't seem to be working well, we need to find a way to parse everything into the json right away and not go through any analyze
        # we can even have only one agent that creates the outline, title and parse it as json file to make it easier for the crew.
        result_task = Task(
            description="Create a video for the given topic that was passed to you as a context. "
                        "when creating the video, spend same amount of time in each bullet point in the outline"
                        " so the video length is evenly distributed among the topics in outline."
                        " also parse the information as json in a way that was told to you.",
            agent = result_agent,
            context=[video_outline_task, title_generation_task],
            output_json=VideoOutlineOutput,
            expected_output="JSON File that shows outline of the video and potential topics of the video."
        )




        ## CREW
        crew = Crew(
            agents=[related_query_agent, channel_content_agent, title_generator_agent,video_outline_agent, result_agent],
            tasks=[related_query_task,channel_content_task, title_generation_task, video_outline_task, result_task],
            verbose=2,
            process = Process.sequential # this ensures that tasks are executed one after each other
        )

        result = crew.kickoff()

        return result

        #return result

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""

    # os.environ["SERPAPI_API_KEY"] = ""

    manager = YoutubeChannelManager()
    topic = "langchain"
    url = "https://www.youtube.com/@LangChain/videos"
    manager.run_crew(topic, url)