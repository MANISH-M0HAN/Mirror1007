import os
import google.generativeai as genai

# Access your API key as an environment variable.
genai.configure(api_key=os.environ['GENERATIVE_AI_API_KEY'])
# Choose a model that's appropriate for your use case.
model = genai.GenerativeModel('gemini-1.5-flash')

prompt = "Who is Obama?"
response = model.generate_content(prompt)

print(response.text)