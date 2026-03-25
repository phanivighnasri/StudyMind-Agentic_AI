import logging

def setup_logger():
    logging.basicConfig(
        filename="rag_agent.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
