# plainEnglishCoding

Large Language Models (LLMs), including ChatGPT, offer an exciting and powerful way to generate code from plain english prompts.
This project allows Jupyter Notebook or Lab users to generate, run, fix, and explain Python code using plain english prompts and questions in the notebook's cells.
Many other recent Jupyter plugins enable users to ask LLMs coding questions, but the onus is on the user to communicate any relevant context for their question (e.g., relevant information about packages, functions, or objects already in memory).
This package overcomes this hurdle by communicating contextual information about Python objects in memory with each call to OpenAI's API.

This package aims to make interactions with ChatGPT as seemless as possible from Jupyter Notebook, including
- interactive chats directly in the Notebook
- automatic coding from simple plain English prompts (e.g., "Simulate the Lorenz butterfly attractor and plot the resulting trajectory.") 
- automatic knowledge of all variables in active memory
- ability to fix broken code
- ability to explain code

Users can interact with a Chat for Coding object programmatically.
But the preferred way to use this package is through magic commands including `%%chat`, `%%code`, `%%fix`, and `%%explain`.
For example, executing the following coding cell in a notebook will produce an interactive chat that ends when the user inputs "done"; here is an example:
![chat example!](/images/chatScreenShot.png "Interactive Chat in the Notebook")

The effective use of this package requires two things from users:
- users must provide their own OpenAI API key (add your key to the chatGPT.py file), and
- users must provide informative documentation (i.e., docstrings) for all functions and classes that they define.

# Example Use cases
### Information Look-Up through Chats in the Notebook

# Conclusion

If you enjoy this package, then please check out my [website](https://sites.pitt.edu/~mrfrank/)
