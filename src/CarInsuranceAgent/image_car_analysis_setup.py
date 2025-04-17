import os
from crewai import LLM, Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')


@CrewBase
class ImageCarAnalysisCrew:

    agents_config = 'agents/image_car_analysis_agents.yaml'
    tasks_config = 'tasks/image_car_analysis_tasks.yaml'

    llm = LLM(
        model=LLM_MODEL,
        base_url="https://api.groq.com/openai/v1/",
        api_key=API_KEY,
        temperature=0.4
    )

    @agent
    def image_interpreter(self):
        return Agent(config=self.agents_config['image_interpreter'], llm=self.llm, verbose=True)

    @agent
    def visual_valuator(self):
        return Agent(config=self.agents_config['visual_valuator'], llm=self.llm, multimodal=True, verbose=True)

    @agent
    def coverage_recommender(self):
        return Agent(config=self.agents_config['coverage_recommender'], llm=self.llm, verbose=True)

    @agent
    def response_summarizer(self):
        return Agent(config=self.agents_config['response_summarizer'], llm=self.llm, verbose=True)

    @task
    def interpret_image_task(self):
        return Task(config=self.tasks_config['interpret_image_task'])

    @task
    def estimate_value_task(self):
        return Task(config=self.tasks_config['estimate_value_task'])

    @task
    def recommend_coverage_task(self):
        return Task(config=self.tasks_config['recommend_coverage_task'])

    @task
    def summarize_results_task(self):
        return Task(config=self.tasks_config['summarize_results_task'])

    @crew
    def crew(self):
        return Crew(
            agents=[
                self.image_interpreter(),
                self.visual_valuator(),
                self.coverage_recommender(),
                self.response_summarizer()
            ],
            tasks=[
                self.interpret_image_task(),
                self.estimate_value_task(),
                self.recommend_coverage_task(),
                self.summarize_results_task()
            ],
            process=Process.sequential,
            verbose=True
        )
