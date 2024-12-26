import google.generativeai as genai
import os
from dotenv import load_dotenv
from prompt_schema import prompt
import base64
# import absl.logging
# absl.logging.set_verbosity(absl.logging.INFO)
# absl.logging.set_verbosity(absl.logging.ERROR)  # Mostra solo errori gravi

# import grpc
# options = [('grpc.max_send_message_length', 1024*1024*50),  # 50MB
#            ('grpc.max_receive_message_length', 1024*1024*50)]  # 50MB
# channel = grpc.insecure_channel('localhost:50051', options=options)

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

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
doc_path = r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\data\A4M_15122024_006852 copy.pdf" # Replace with the actual path to your local PDF

# Read and encode the local file
with open(doc_path, "rb") as doc_file:
    doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")


response = model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])

print(response.text)