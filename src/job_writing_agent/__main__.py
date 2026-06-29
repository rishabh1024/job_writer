"""Package entry point — CLI entrypoint for the job application writer."""

import asyncio
import logging
import sys

from job_writing_agent.run_workflow import JobWorkflow
from job_writing_agent.utils.app_logging import configure_logging
from job_writing_agent.utils.application_cli import start_cli
from job_writing_agent.utils.result_utils import _print_result, _save_result

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """Entry point for the job application writer workflow cli"""

    configure_logging()
    args = start_cli()

    workflow = JobWorkflow(
        resume=args.resume,
        job_description_source=args.jd_source,
        content=args.content_type,
    )

    logger.info("Running workflow...")

    result = asyncio.run(workflow.run_workflow())

    if result and "output_data" in result:
        _print_result(args.content_type, result.get("output_data", ""))
        # Make the save result conditional on the user's preference
        _save_result(args.content_type, result.get("output_data", ""))
        logger.info("Workflow completed successfully.")
        sys.exit(0)

    logger.error("Error running workflow. No output data available.")
    sys.exit(1)
