"""
This module contains predefined prompts for use with LLaMA 3 - 70B and Gemini 1.5 Flash models.
"""

# General-purpose prompts
GENERAL_PROMPTS = {
    "introduction": "Write an introduction about the impact of AI on semiconductor research.",
    "summary": "Summarize the following text:\n{input_text}",
    "question": "Answer the following question:\n{question}",
}

# Prompts for specific tasks
TASK_PROMPTS = {
    "generate_ideas": "Generate a list of innovative ideas for improving semiconductor manufacturing processes.",
    "explain_concept": "Explain the concept of {concept} in simple terms.",
    "compare_technologies": "Compare the advantages and disadvantages of {technology1} and {technology2}.",
}

# Prompts for research-related tasks
RESEARCH_PROMPTS = {
    "literature_review": "Provide a literature review on the topic of {topic}.",
    "future_trends": "What are the future trends in {field}?",
    "research_question": "What are the key research questions in the field of {field}?",
}

# Function to retrieve a prompt
def get_prompt(category, key, **kwargs):
    """
    Retrieve a prompt from the specified category and key.
    :param category: The category of the prompt (e.g., 'GENERAL_PROMPTS', 'TASK_PROMPTS').
    :param key: The key of the specific prompt within the category.
    :param kwargs: Additional arguments to format the prompt.
    :return: The formatted prompt string.
    """
    categories = {
        "general": GENERAL_PROMPTS,
        "task": TASK_PROMPTS,
        "research": RESEARCH_PROMPTS,
    }

    if category not in categories:
        raise ValueError(f"Invalid category: {category}. Valid categories are: {list(categories.keys())}")

    prompts = categories[category]
    if key not in prompts:
        raise ValueError(f"Invalid key: {key}. Valid keys for category '{category}' are: {list(prompts.keys())}")

    prompt = prompts[key]
    return prompt.format(**kwargs)

# Example usage
if __name__ == "__main__":
    # Example: Retrieve a general prompt
    prompt = get_prompt("general", "introduction")
    print("General Prompt:\n", prompt)

    # Example: Retrieve a task prompt with formatting
    prompt = get_prompt("task", "explain_concept", concept="quantum computing")
    print("\nTask Prompt:\n", prompt)

    # Example: Retrieve a research prompt with formatting
    prompt = get_prompt("research", "future_trends", field="AI in semiconductors")
    print("\nResearch Prompt:\n", prompt)