import os
import base64
import argparse

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import google.generativeai as genai
import ollama


def create_full_text(row, prefix="search_document"):
    """Crea una stringa di testo unendo le colonne non nulle."""
    return f"{prefix}: " + ", ".join(
        f"{col}: {val}"
        for col, val in row.items()
        if pd.notna(val)
    )


def get_embedding(text, method="local", task_type="retrieval_document"):
    """Restituisce l'embedding del testo usando il metodo specificato."""
    if method == "local":
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Errore durante l'embedding: {e}")
            return None

    if method == "api":
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type=task_type
            )
            return response["embedding"]
        except Exception as e:
            print(f"Errore durante l'embedding: {e}")
            return None


def count_tokens_simple(text):
    """Conta il numero di token (parole) nel testo."""
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(
        description="Script per ricerca e RAG su dati clienti."
    )
    parser.add_argument(
        "--clienti_csv",
        type=str,
        default=r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\data\clienti.csv",
        help="Path al file CSV dei clienti."
    )
    parser.add_argument(
        "--embeddings_parquet",
        type=str,
        default=r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\embeddings\nomic_search_index.parquet",
        help="Path al file Parquet per gli embeddings."
    )
    parser.add_argument(
        "--doc_pdf",
        type=str,
        default=r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\data\inputs\otto.pdf",
        help="Path al file PDF dell'ordine."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Numero di top risultati da considerare."
    )
    args = parser.parse_args()

    # Caricamento dati e creazione della colonna full_text
    df_clienti = pd.read_csv(args.clienti_csv, sep=";")
    df_clienti["full_text"] = df_clienti.apply(create_full_text, axis=1)
    df_clienti_full_text = df_clienti["full_text"]

    # Gestione del salvataggio degli embeddings
    if not os.path.exists(args.embeddings_parquet):
        df_clienti_emb = df_clienti_full_text.apply(get_embedding)
        df_clienti_emb = pd.DataFrame(df_clienti_emb)
        df_clienti_emb.to_parquet(args.embeddings_parquet, engine="pyarrow")
        print("Embeddings salvati")
    else:
        print(f"Il path '{args.embeddings_parquet}' esiste già, uscita dal ciclo.")

    # Configurazione API e lettura del documento PDF
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    with open(args.doc_pdf, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

    prompt = (
        "Trascrivi fedelmente il contenuto di questo documento PDF. Non aggiungere spiegazioni, "
        "riassunti o commenti. Devi solo riportare il testo esattamente come appare nel documento."
    )
    response = model.generate_content([
        {'mime_type': 'application/pdf', 'data': doc_data},
        prompt
    ])
    ordine = response.text

    # Generazione dell'embedding della query
    df_clienti_emb = pd.read_parquet(args.embeddings_parquet)
    query = "search_query: Trovami l'IDCliente (Codice Identificatore Cliente) del seguente ordine: \n" + ordine
    query_emb = get_embedding(query, task_type="retrieval_query", method="local")

    embeddings_series = df_clienti_emb["full_text"]
    embedding_matrix = np.array(embeddings_series.tolist(), dtype=np.float32)
    print("Shape della matrice degli embedding:", embedding_matrix.shape)
    print("Shape dell'embedding della query:", len(query_emb))

    similarities = 1 - cdist([query_emb], embedding_matrix, metric="cosine")[0]
    top_k_indices = np.argsort(similarities)[::-1][:args.k]
    print("Top 5 Indici più simili:", top_k_indices[:5])
    print("Similarità corrispondenti:", similarities[top_k_indices[:5]])

    rag_results = "\n".join(df_clienti_full_text.loc[top_k_indices].astype(str).tolist())

    prompt_rag = f"""search_query: Trovami l'IDCliente (Codice Identificatore Cliente) del seguente ordine:

🚀 **IMPORTANTE: INIZIO ORDINE** 🚀

{ordine}

**IMPORTANTE: FINE ORDINE**

"sapendo che tramite RAG ho ottenuto i full text (stringa contenente tutti i dati strutturati tabellari) dei clienti più simili all'ordine in oggetto. Riporto di seguito i dati full text dei clienti più simili all'ordine. Dai importanza sopratutto alla destinazione dell'ordine e alla ragione sociale dell'ordine (l'Indirizzo Dest la RagSoc nella tabella) per scegliere la riga trai top-k che ti propongo!

🚀 **IMPORTANTE: INIZIO ELENCO DATI ESTRATTI TRAMITE RAG** 🚀

{rag_results}

**IMPORTANTE: FINE ELENCO DATI ESTRATTI TRAMITE RAG**

"Restituiscimi soltanto l'IDCliente, la RagSoc e l'Indirizzo Dest in formato JSON, non voglio alcun tipo di dettaglio aggiuntivo"
"""
    print("Numero di token nel prompt RAG:", count_tokens_simple(prompt_rag))
    print(prompt_rag)

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",  # Formato strutturato della risposta
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        # system_instruction="Sei un venditore, quindi i campi vuoti non li sostituisci con Null ma con valori inventati!"
    )
    output_rag = model.generate_content(prompt_rag)
    print(output_rag.text)


if __name__ == "__main__":
    main()
