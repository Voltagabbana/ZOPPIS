import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64

from prompt_schema import prompt
from parser import path_input
from parser import path_output

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json", # posso richiedere esplicitamente il formato strutturato della risposta
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    #system_instruction="Sei un venditore, quindi i campi vuoti non li sostituisci con Null ma con valori inventati!" # solo come esperimento..
)

doc_path = path_input # Replace with the actual path to your local PDF

with open(doc_path, "rb") as doc_file:
    doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

print("Inizio generazione dell'output...")
response = model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])


output_dir = path_output
output_file_path = os.path.join(output_dir, 'response_output.txt')

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(response.text)
print(f"Risposta scritta nel file: {output_file_path}")
