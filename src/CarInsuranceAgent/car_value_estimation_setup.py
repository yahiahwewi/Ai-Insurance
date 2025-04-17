import os
from crewai import LLM, Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from .env file
API_KEY = os.getenv('API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')

@CrewBase
class CarValueEstimatorCrew:

    # YAML paths
    agents_config = 'agents/car_value_estimation_agents.yaml'
    tasks_config = 'tasks/car_value_estimation_tasks.yaml'

    # Set up LLM
    llm = LLM(
        model=LLM_MODEL,
        base_url="https://api.groq.com/openai/v1/",
        api_key=API_KEY,
        temperature=0.4
    )

    # Define agents
    @agent
    def data_validator(self) -> Agent:
        return Agent(config=self.agents_config['data_validator'], llm=self.llm, verbose=True)

    @agent
    def market_analyst(self) -> Agent:
        return Agent(config=self.agents_config['market_analyst'], llm=self.llm, verbose=True)

    @agent
    def condition_adjuster(self) -> Agent:
        return Agent(config=self.agents_config['condition_adjuster'], llm=self.llm, verbose=True)

    @agent
    def final_synthesizer(self) -> Agent:
        return Agent(config=self.agents_config['final_synthesizer'], llm=self.llm, verbose=True)

    # Define tasks
    @task
    def validate_input_task(self) -> Task:
        return Task(config=self.tasks_config['validate_input_task'])

    @task
    def analyze_market_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_market_task'])

    @task
    def adjust_for_condition_task(self) -> Task:
        return Task(config=self.tasks_config['adjust_for_condition_task'])

    @task
    def generate_valuation_report_task(self) -> Task:
        return Task(config=self.tasks_config['generate_valuation_report_task'])

    # Define crew
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.data_validator(),
                self.market_analyst(),
                self.condition_adjuster(),
                self.final_synthesizer()
            ],
            tasks=[
                self.validate_input_task(),
                self.analyze_market_task(),
                self.adjust_for_condition_task(),
                self.generate_valuation_report_task()
            ],
            process=Process.sequential,
            verbose=True
        )
