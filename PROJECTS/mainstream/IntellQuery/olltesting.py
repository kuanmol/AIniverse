import ollama

# Test prompt
prompt = "What is the purpose of data visualization in analytics?"
response = ollama.chat(model="llama3.1:8b", messages=[
    {"role": "user", "content": prompt}
])

# Print response
print("Prompt:", prompt)
print("Response:", response['message']['content'])