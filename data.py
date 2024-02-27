import pandas as pd
from pytrends.request import TrendReq


from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

class GoogleTrendsData:
    def __init__(self):
        pass
    def fetch_data(self, keywords):

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
        print(allqueries.head(100))
        return allqueries


    def scrape_youtube_channel(self, website):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode
        chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration

        # Initialize Chrome webdriver
        driver = webdriver.Chrome(options=chrome_options)

        try:
            # Load the website
            driver.get(website)

            # Wait for the page to fully render (adjust the timeout as needed)
            driver.implicitly_wait(10)

            # Find the element with the specified ID
            contents_div = driver.find_element(By.ID,"contents")

            # Get the text content of the element
            content = contents_div.text

            return content

        finally:
            # Close the webdriver
            driver.quit()

