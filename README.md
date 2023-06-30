# plainEnglishCoding

Large Language Models (LLMs), including ChatGPT, offer an exciting and powerful way to generate code from plain english prompts.
This project allows Jupyter Notebook or Lab users to generate, run, fix, and explain Python code using plain english prompts and questions in the notebook's cells.
Many other recent Jupyter plugins enable users to ask LLMs coding questions, but the onus is on the user to communicate ny relevant context for their question (e.g., relevant information about packages, functions, or objects already in memory).
This package overcomes this hurdle by communicating contextual information about Python objects in memory with each call to OpenAI's API.

The effective use of this package requires two things from users:
- users must provide their own OpenAI API key, and
- users must provide informative documentation (i.e., docstrings) for all functions and classes that they define.

# Example Use cases

