import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.llms import Ollama
from agents.agents import AINewsLetterAgents
from tasks.task import AINewsLetterTasks
from crewai import Crew, Process
from file_io import save_markdown

load_dotenv()

# llm = ChatOpenAI(
#     model="crewai-llama2",
#     base_url="http://localhost:11434/v1",
# )

# Azure LLM for production
default_llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)

# llm = ChatOpenAI(
#     model="gpt-4",
#     base_url="http://localhost:11434/v1",
# )

# Local LLM for testing
ollama_mistral = Ollama(model="crewai-mistral:latest")

agents = AINewsLetterAgents()
tasks = AINewsLetterTasks()

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