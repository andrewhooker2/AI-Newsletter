import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from agents.agents import NewsLetterAgents
from tasks.task import NewsLetterTasks
from crewai import Crew, Process
from file_io import save_markdown

load_dotenv()

# Azure LLM for production
default_llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-PREVIEW"),
    azure_deployment=os.environ.get("AZURE_DEPLOYMENT", "gpt35"),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT", "https://<your-endpoint>.openai.azure.com/"),
    api_key=os.environ.get("AZURE_API_KEY"),
)

agents = NewsLetterAgents(topic='AI')
tasks = NewsLetterTasks(topic='AI')

editor = agents.editor_agent()
newsFetcher = agents.news_fetcher_agent()
newsAnalyzer = agents.news_analyzer_agent()
newsletterCompiler = agents.newsletter_compiler_agent()

fetch_news_task = tasks.fetch_news_task(newsFetcher)
analyze_news_task = tasks.analyze_news_task(newsletterCompiler, [fetch_news_task])
compile_newsletter_task = tasks.compile_newsletter_task(newsletterCompiler, [analyze_news_task], save_markdown)

crew = Crew(
    agents=[editor, newsFetcher, newsAnalyzer, newsletterCompiler],
    tasks=[fetch_news_task, analyze_news_task, compile_newsletter_task],
    process=Process.hierarchical,
    manager_llm=default_llm
)

results = crew.kickoff()

print(results)
print("=====================================")
