import pandas as pd
import ollama
import os
import google.generativeai as genai
import os
import base64
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cdist


def create_full_text(row, prefix="search_document"):
    return f"{prefix}: " + ", ".join(
        f"{col}: {val}"
        for col, val in row.items() 
        if pd.notna(val)
    )

def get_embedding(text):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Errore durante l'embedding: {e}")
        return None
def count_tokens_simple(text):
    return len(text.split())

def main():
    df_clienti = pd.read_csv(r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\data\clienti.csv",sep = ";")
    df_clienti["full_text"] = df_clienti.apply(create_full_text, axis=1)
    df_clienti_full_text = df_clienti["full_text"]

    path = r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\notebooks\embeddings.parquet"

    if not os.path.exists(path):
        df_clienti_emb = df_clienti_full_text.apply(get_embedding)
        df_clienti_emb = pd.DataFrame(df_clienti_emb)
        df_clienti_emb.to_parquet("embeddings_v2.parquet", engine="pyarrow")
        print("Embeddings salvati")
    else:
        print(f"Il path '{path}' esiste già, uscita dal ciclo.")

    ## argparse
    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    doc_path = r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\data\inputs\otto.pdf"
    with open(doc_path, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")
    prompt = """
    Trascrivi fedelmente il contenuto di questo documento PDF. Non aggiungere spiegazioni, riassunti o commenti.
    Devi solo riportare il testo esattamente come appare nel documento.
    """
    response = model.generate_content([
        {'mime_type': 'application/pdf', 'data': doc_data},
        prompt
    ])
    #print(response.text)
    ordine = response.text

    df_clienti_emb = pd.read_parquet(r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\notebooks\embeddings_v2.parquet")
    query = "search_query: Trovami l'IDCliente (Codice Identificatore Cliente) del seguente ordine: \n" + ordine
    query_emb = get_embedding(query)

    k = 15
    embeddings_series = df_clienti_emb["full_text"]
    query_embedding = query_emb  
    embedding_matrix = np.array(embeddings_series.tolist(), dtype=np.float32)
    print("Shape della matrice degli embedding:", embedding_matrix.shape)
    print("Shape dell'embedding della query:", len(query_embedding))
    similarities = 1 - cdist([query_embedding], embedding_matrix, metric="cosine")[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    print("Top 5 Indici più simili:", top_k_indices)
    print("Similarità corrispondenti:", similarities[top_k_indices])

    rag_results = "\n".join(df_clienti_full_text.loc[top_k_indices].astype(str).tolist())

    rag_results = "\n".join(df_clienti_full_text.loc[top_k_indices].astype(str).tolist())
    prompt_rag = f"""search_query: Trovami l'IDCliente (Codice Identificatore Cliente) del seguente ordine:

    🚀 **IMPORTANTE: INIZIO ORDINE** 🚀

    {ordine}

    **IMPORTANTE: FINE ORDINE**

    "sapendo che tramite RAG ho ottenuto i full text (stringa contenente tutti i dati strutturati tabellari) dei clienti più simili all'ordine in oggetto. Riporto di seguito i dati full text dei clienti più simili all'ordine. Dai importanza sopratutto alla destinazione dell'ordine e alla ragione sociale dell'ordine (l'Indirizzo Dest la RagSoc nella tabella) per scegliere la riga trai top-k che ti propongo! \n"
    
    🚀 **IMPORTANTE: INIZIO ELENCO DATI ESTRATTI TRAMITE RAG** 🚀
    
    {rag_results}

    **IMPORTANTE: FINE ELENCO DATI ESTRATTI TRAMITE RAG**

    "\nRestituiscimi soltanto l'IDCliente, la RagSoc e l'Indirizzo Dest in formato JSON, non voglio alcun tipo di dettaglio aggiuntivo"" 
    """

    print(count_tokens_simple(prompt_rag))

    print(prompt_rag)

    generation_config = {
  "temperature": 0.1,
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

    output_rag = model.generate_content(prompt_rag)
    print(output_rag.text)

if __name__ == "__main__":
    main()