cohere_prompt_guideines = (
    lambda description: f"""
Construct a prompt for cohere Large language Model from a user description by following these guidelines :

To write a good prompt in the Cohere Playground, you can follow these guidelines based on the provided article:

Understand the Purpose: Clearly define the purpose of your prompt. Identify what specific output or task you want the generative model to perform.

Be Clear and Specific: Craft your prompt to provide clear instructions or commands to the model. Use imperative verbs like "generate," "write," "list," or "provide" to guide the model's output. The more specific you are, the better the model will understand your requirements.

Use Context and Examples: Consider adding context or examples to your prompt to help ground the model's output. Providing additional information or relevant examples can improve the accuracy and relevance of the generated text.

Iterate and Experiment: Prompt design is a creative process, so feel free to iterate and experiment with different variations of prompts. Test different instructions, formats, or additional details to refine and improve the output.

Consider Prompt Length: Depending on the complexity of the task, you can use both short and long prompts. A concise prompt can sometimes yield satisfactory results, while longer prompts with more instructions and context may be necessary for more specific tasks.


Remember, prompt design is a combination of science and art. While there are guiding principles, it's also essential to be creative and open to exploring different approaches until you achieve the desired outcome.  

User Description :

{description}

Prompt :
"""
)
google_prompt_guidelines = (
    lambda description: f"""
Construct a prompt for Google Generative Ai Large Language Model from a user description by following these guidelines : 

Give clear instructions:
Prompt the model to provide specific guidance and suggestions to enhance the formatting of academic papers. For example:
"Please provide recommendations to improve the structure and organization of the introduction section of an academic paper on [topic]."

Include examples:
Present the model with well-formatted examples of academic paper sections or elements. Request the model to generate similar structures or formats. For instance:
"Based on the provided abstract, generate a concise and well-structured conclusion for an academic paper."

Contextual information:
If needed, provide contextual information that the model can use to tailor its response. For example:
"Given the research question, provide a discussion section that presents the findings and their implications in a clear and coherent manner."

Partial input completion:
Ask the model to complete or revise partial content based on formatting rules. For instance:
"Given the incomplete citation, please generate the full APA citation for the provided scholarly article."

Response formatting:
Guide the model to format its responses appropriately. For example:
"Format your response as a bulleted list, outlining the key steps involved in conducting a literature review for an academic research paper."

User Description : 

{description}

Prompt : 
"""
)
general_prompt_guidelines = (
    lambda description, provider_name, model_name: f"""
Construct a prompt for {provider_name} {model_name} Large Language Model from a user description by following these guidelines : 
To write good prompts for {model_name} models, you can follow these guidelines :

Be clear and specific: Provide clear instructions and constraints in your prompt to guide the model towards generating the desired output. Avoid leaving ambiguous or open-ended prompts that may result in unexpected responses. Specify the task or query explicitly.

Example: Instead of "Classify this post," use "Classify the sentiment of this post as positive, neutral, or negative: 'My cat is adorable '"

Provide sample outputs: If you have specific formatting requirements or want the model to generate outputs in a particular structure, provide examples of the expected output. This helps the model understand the desired format and align its responses accordingly.

Example: Instead of just asking to extract cities and airport codes, provide an example of the expected JSON output structure:

Extract the cities and airport codes from this text as JSON:

Text: "I want to fly from Los Angeles to Miami."
JSON Output: {{
  "Origin": {{
    "CityName": "Los Angeles",
    "AirportCode": "LAX"
  }},
  "Destination": {{
    "CityName": "Miami",
    "AirportCode": "MIA"
  }}
}}

Provide relevant context: If the prompt requires the model to answer questions or perform specific tasks, provide relevant background information or facts to guide the model's understanding. This helps prevent the model from generating fabricated or incorrect responses.

Example: When asking questions about a specific document, include relevant context from that document in your prompt:

Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun... [additional context]

Answer the following question:
Q: Which is the fifth planet from the sun?
A: Jupiter

Q: What’s the mass of Jupiter compared to the Sun?

Refine, refine, refine: Prompt engineering can be an iterative process. Experiment with different techniques, iterate on your prompts, and learn from the initial outputs generated by the model. Use the generated outputs to provide additional context or guidance in subsequent prompts to improve the quality of the responses.

By following these tips and refining your prompts, you can guide {model_name} models to generate more accurate and desired completions.

User Description : {description}
Prompt :
"""
)

anthropic_prompt_guidelines = (
    lambda description: f"""
Construct a prompt for Anthropic Claude Large Language Model from a user description by following these guidelines : 

To get the best results from Claude, follow these prompt engineering best practices: 
1. Provide clear, direct instructions: - Explain the task in a straightforward manner, as if instructing a 
new employee - Give as much context and detail as possible, including any rules or requirements - 
Use numbered steps or bullet points to break down complex tasks 

2. Use examples effectively: - Include 3-5 relevant examples in your prompt, wrapped in <example></example> tags - 
Make examples clear, concise, and representative of desired inputs/outputs - 
Use examples to demonstrate formatting, writing style, level of detail, etc. 

3. Incorporate XML tags for structure: - Use tags like <instructions></instructions> and <input></input> to 
delineate prompt sections - Wrap variable inputs in tags like <product>{{PRODUCT}}</product> - Ask Claude 
to use tags in its output, e.g. <answer></answer>, for easier parsing 4. Try techniques like role prompting 
and chain-of-thought prompting: - Assign a role to Claude (e.g. "You are a master logic solver") 
to improve performance - Instruct Claude to "Think step-by-step" and walk through its reasoning process 

5. Prefill Claude's response: - Provide initial text in the Assistant message to steer the output format or direction - 
Use prefilling to maintain consistency in role-playing scenarios

6. Ask for rewrites when needed: - If Claude's initial response doesn't meet your needs, provide a revised <instructions> 
section - Guide Claude by giving a clear rubric for what an ideal rewrite should include, always test and iterate your 
prompts to unlock Claude's full potential for your use case. 

User Description : {description}
Prompt :
"""
)

perplexityai_prompt_guidelines = (
    lambda description: f"""
Construct a prompt for PerplexityAI Large Language Model from a user description by following these guidelines : 

Clarity and Specificity:
    Clearly state what you need or want to accomplish.
    Be specific about the topic or task you're requesting assistance with.

Anatomy of a Prompt:
    Seamlessly blend your goal and background into a concise sentence.
    Incorporate keywords that guide the AI to better understand and serve your request.
    Ensure the formatting is visually clean for easier processing and better results.

Keyword Inclusion:
    Include relevant keywords that highlight key aspects of your request.
    Keywords help Perplexity understand the context and provide more accurate responses.

Tailored to Audience or Context:
    Craft prompts tailored to your specific role, profession, or context.
    Personalize the prompt to match your needs or the task at hand.

Emphasis on Goal Achievement:
    Clearly articulate your goal or objective within the prompt.
    The prompt should set the stage for what you aim to achieve or obtain through Perplexity's assistance.

Consideration of Constraints or Preferences:
    If applicable, include any constraints, preferences, or specific requirements related to your request.
    This helps Perplexity provide more relevant and tailored responses.

Encapsulation of Information:
    Ensure the prompt encapsulates all necessary information for Perplexity to understand and address your request effectively.
    Avoid ambiguity or vague language to prevent misunderstandings.

Conciseness and Readability:
    Keep the prompt concise and easy to read.
    Avoid unnecessary complexity or verbosity to streamline communication and enhance comprehension.

By following these guidelines, you can create prompts that effectively communicate your needs and enable 
Perplexity to provide accurate and helpful responses.

User Description : {description}
Prompt :
"""
)
