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
        "--doc_file",
        type=str,
        default=r"C:\Users\enduser\Desktop\PythonPostLaurea\ZOPPIS\ZOPPIS\data\inputs\otto.pdf",
        help="Path al file dell'ordine (PDF o file di testo)."
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["pdf", "text"],
        default="pdf",
        help="Tipo di file in input: 'pdf' oppure 'text'."
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
        print(f"Il path '{args.embeddings_parquet}' esiste già, uscita dal ciclo iniziale.")

    # Configurazione API e acquisizione del documento
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # Acquisizione iniziale della variabile "ordine"
    if args.input_type == "pdf":
        model_pdf = genai.GenerativeModel("gemini-2.0-flash")
        with open(args.doc_file, "rb") as doc_file:
            doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")
        prompt_pdf = (
            "Trascrivi fedelmente il contenuto di questo documento PDF. Non aggiungere spiegazioni, "
            "riassunti o commenti. Devi solo riportare il testo esattamente come appare nel documento."
        )
        response_pdf = model_pdf.generate_content([
            {'mime_type': 'application/pdf', 'data': doc_data},
            prompt_pdf
        ])
        ordine = response_pdf.text
    else:  # input_type == "text"
        with open(args.doc_file, "r", encoding="utf-8") as text_file:
            ordine = text_file.read()

    # Loop iterativo per validazione dell'output
    while True:
        # Costruzione della query e calcolo dell'embedding
        query = "search_query: Trovami l'IDCliente (Codice Identificatore Cliente) del seguente ordine: \n" + ordine
        query_emb = get_embedding(query, task_type="retrieval_query", method="local")

        # Calcolo delle similarità e recupero dei top-k documenti
        df_clienti_emb = pd.read_parquet(args.embeddings_parquet)
        embeddings_series = df_clienti_emb["full_text"]
        embedding_matrix = np.array(embeddings_series.tolist(), dtype=np.float32)
        similarities = 1 - cdist([query_emb], embedding_matrix, metric="cosine")[0]
        top_k_indices = np.argsort(similarities)[::-1][:args.k]
        rag_results = "\n".join(df_clienti_full_text.loc[top_k_indices].astype(str).tolist())

        # Costruzione del prompt per il modello RAG
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
        #print(prompt_rag)

        response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt_rag}]) # llama3.2:3b, gemma3:4b
        print(response["message"]["content"])

        # Validazione da terminale
        user_confirm = input("\nIl risultato è corretto? (yes/no): ").strip().lower()
        if user_confirm == "yes":
            print("Ottimo, sono contento di esserti stato d'aiuto 😄")
            break
        else:
            additional_context = input("Scrivimi più dettagli sull'ordine per aiutarti meglio 💪😊: ")
            ordine = ordine + "\n" + additional_context
            print("\n[Contesto aggiornato, rieseguo la procedura!🚀🚀🚀]\n")


if __name__ == "__main__":
    main()
