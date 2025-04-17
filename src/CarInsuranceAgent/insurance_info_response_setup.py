import os
from crewai import LLM, Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
API_KEY = os.getenv('API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')

@CrewBase
class InsuranceInfoResponderCrew:

    # Paths to modular YAML config files
    agents_config = 'agents/insurance_info_response_agents.yaml'
    tasks_config = 'tasks/insurance_info_response_tasks.yaml'

    # Custom LLM setup
    llm = LLM(
        model=LLM_MODEL,
        base_url="https://api.groq.com/openai/v1/",
        api_key=API_KEY,
        temperature=0.4
    )

    # Agents
    @agent
    def intent_classifier(self) -> Agent:
        return Agent(config=self.agents_config['intent_classifier'], llm=self.llm, verbose=True)

    @agent
    def insurance_researcher(self) -> Agent:
        return Agent(config=self.agents_config['insurance_researcher'], llm=self.llm, verbose=True)

    @agent
    def plain_language_explainer(self) -> Agent:
        return Agent(config=self.agents_config['plain_language_explainer'], llm=self.llm, verbose=True)

    @agent
    def next_step_advisor(self) -> Agent:
        return Agent(config=self.agents_config['next_step_advisor'], llm=self.llm, verbose=True)

    # Tasks
    @task
    def classify_question_task(self) -> Task:
        return Task(config=self.tasks_config['classify_question_task'])

    @task
    def research_insurance_info_task(self) -> Task:
        return Task(config=self.tasks_config['research_insurance_info_task'])

    @task
    def explain_in_plain_language_task(self) -> Task:
        return Task(config=self.tasks_config['explain_in_plain_language_task'])

    @task
    def offer_next_steps_task(self) -> Task:
        return Task(config=self.tasks_config['offer_next_steps_task'])

    # Crew definition
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.intent_classifier(),
                self.insurance_researcher(),
                self.plain_language_explainer(),
                self.next_step_advisor()
            ],
            tasks=[
                self.classify_question_task(),
                self.research_insurance_info_task(),
                self.explain_in_plain_language_task(),
                self.offer_next_steps_task()
            ],
            process=Process.sequential,
            verbose=True
        )
