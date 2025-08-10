import openai
import re
import json

class LLMHandler:
    def __init__(self):
        # Set your API key
        self.client = openai.OpenAI(api_key="your-openai-api-key")
    
    def parse_questions(self, questions_text):
        # Extract individual questions
        lines = questions_text.strip().split('\n')
        questions = []
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                questions.append(line.strip())
        
        return questions
    
    def analyze_with_llm(self, data_description, question):
        prompt = f"""
        Given this data: {data_description}
        
        Answer this question: {question}
        
        Provide a precise, numerical answer when possible.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    