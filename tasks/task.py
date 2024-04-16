from datetime import datetime
from crewai import Task

class NewsLetterTasks():
    def __init__(self, topic):
        self.topic = topic

    def fetch_news_task(self, agent):
        return Task(
            description=f'Fetch top {self.topic} news stories from the past 24 hours. The current time is {datetime.now()}.',
            agent=agent,
            async_execution=True,
            expected_output=f"""A list of top {self.topic} news story titles, URLs, and a brief summary for each story from the past 24 hours. 
                Example Output: 
                [
                    {{  'title': '{self.topic} takes spotlight in Super Bowl commercials', 
                    'url': 'https://example.com/story1', 
                    'summary': '{self.topic} made a splash in this year\'s Super Bowl commercials...'
                    }}, 
                    {{...}}
                ]
            """
        )

    def analyze_news_task(self, agent, context):
        return Task(
            description=f'Analyze each {self.topic} news story and ensure there are at least 5 well-formatted articles',
            agent=agent,
            async_execution=True,
            context=context,
            expected_output=f"""A markdown-formatted analysis for each {self.topic} news story, including a rundown, detailed bullet points, 
                and a "Why it matters" section. There should be at least 5 articles, each following the proper format.
                Example Output: 
                '## {self.topic} takes spotlight in Super Bowl commercials\n\n
                **The Rundown:
                ** {self.topic} made a splash in this year\'s Super Bowl commercials...\n\n
                **The details:**\n\n
                - Microsoft\'s Copilot spot showcased its {self.topic} assistant...\n\n
                **Why it matters:** While {self.topic}-related ads have been rampant over the last year, its Super Bowl presence is a big mainstream moment.\n\n'
            """
        )

    def compile_newsletter_task(self, agent, context, callback_function):
        return Task(
            description=f'Compile the {self.topic} newsletter',
            agent=agent,
            context=context,
            expected_output=f"""A complete {self.topic} newsletter in markdown format, with a consistent style and layout.
                Example Output: 
                '# Top stories in {self.topic} today:\\n\\n
                - {self.topic} takes spotlight in Super Bowl commercials\\n
                - Altman seeks TRILLIONS for global {self.topic} chip initiative\\n\\n

                ## {self.topic} takes spotlight in Super Bowl commercials\\n\\n
                **The Rundown:** {self.topic} made a splash in this year\'s Super Bowl commercials...\\n\\n
                **The details:**...\\n\\n
                **Why it matters::**...\\n\\n
                ## Altman seeks TRILLIONS for global {self.topic} chip initiative\\n\\n
                **The Rundown:** OpenAI CEO Sam Altman is reportedly angling to raise TRILLIONS of dollars...\\n\\n'
                **The details:**...\\n\\n
                **Why it matters::**...\\n\\n
            """,
            callback=callback_function
        )