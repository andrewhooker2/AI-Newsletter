import os

from crewai import Agent
from langchain_openai import AzureChatOpenAI
from tools.search_tools import SearchTools

default_llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-PREVIEW"),
    azure_deployment=os.environ.get("AZURE_DEPLOYMENT", "gpt35"),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT", "https://<your-endpoint>.openai.azure.com/"),
    api_key=os.environ.get("AZURE_API_KEY"),
)


class NewsLetterAgents():
    def __init__(self, topic):
        self.topic = topic

    def editor_agent(self):
        return Agent(
            role='Editor',
            goal=f'Oversee the creation of the {self.topic} Newsletter',
            backstory=f"""With a keen eye for detail and a passion for storytelling, you ensure that the newsletter
            not only informs but also engages and inspires the readers. """,
            allow_delegation=True,  # Allowing the work to delegate tasks to other agents
            verbose=True,
            max_iter=5,  # Setting the Agent to only run a set number of times
            llm=default_llm
        )

    def news_fetcher_agent(self):
        return Agent(
            role='NewsFetcher',
            goal=f'Fetch the top {self.topic} news stories for the day',
            backstory=f"""As a digital sleuth, you scour the internet for the latest and most impactful developments
            in the world of {self.topic}, ensuring that our readers are always in the know.""",
            tools=[SearchTools.search_internet],
            verbose=True,
            allow_delegation=True,
            llm=default_llm,
        )

    def news_analyzer_agent(self):
        return Agent(
            role='NewsAnalyzer',
            goal=f'Analyze each {self.topic} news story and generate a detailed markdown summary',
            backstory=f"""With a critical eye and a knack for distilling complex information, you provide insightful
            analyses of {self.topic} news stories, making them accessible and engaging for our audience.""",
            tools=[SearchTools.search_internet],
            verbose=True,
            allow_delegation=True,
            llm=default_llm
        )

    def newsletter_compiler_agent(self):
        return Agent(
            role='NewsletterCompiler',
            goal=f'Compile the analyzed {self.topic} news stories into a final newsletter format',
            backstory=f"""As the final architect of the newsletter, you meticulously arrange and format the content,
            ensuring a coherent and visually appealing presentation that captivates our readers. Make sure to follow
            newsletter format guidelines and maintain consistency throughout.""",
            verbose=True,
            llm=default_llm
        )
