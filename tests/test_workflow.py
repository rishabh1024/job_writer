from job_writing_agent.graph.agent_workflow_graph import job_app_graph
from pprint import pprint

if __name__ == "__main__":
    pprint(job_app_graph.get_input_schema().model_json_schema())