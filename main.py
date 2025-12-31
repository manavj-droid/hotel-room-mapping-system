"""Main entry point for the hotel room mapping system."""

import sys
from pathlib import Path
from src.pipeline_orchestrator import PipelineOrchestrator
from config.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Execute the main pipeline."""
    try:
        logger.info("Hotel Room Mapping System starting...")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        logger.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
