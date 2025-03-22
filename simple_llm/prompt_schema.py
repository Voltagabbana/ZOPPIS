prompt = """
Fammi un JSON pagina per pagina con gli ordini contenuti nel pdf, in Italiano,
la struttura del json deve essere identica a quella che ti definisco nelle prossime righe; quando un campo non ti è chiaro, la grafia poco leggibile,
o mancano queste informazioni, scrivi comunque il campo, seguito da N.A.

"Pagina_documento": ,
"Cliente": ,
"Ordini": [
        {
          "Valord": ,
          "IDOrdine": ,
          "CodUtente": ,
          "Data": ,
          "Stato": ,
          "RagSoc_Dest": ,
          "Indirizzo_Dest": ,
          "CAP_Dest": ,
          "Comune_Dest": ,
          "Prov_Dest": ,
          "Paese_Dest": ,
          "NumTel_Dest": ,
          "NumCellulare_Dest": ,
          "Persona_Dest": ,
          "NumFax_Dest": ,
          "Pagamento": ,
          "TipoConsegna": ,
          "Note": ,
          "TipoOrdine": ,
          "Messaggio": ,
          "Docum": ,
          "IDContatto": ,
          "Inviato": ,
          "TotOrdinelvalncl": ,
          "TotPesoLordo": ,
          "SpeseTrasportolvaEscl": ,
          "SpeselmballaggiolvaEscl": ,
          "SpeseAssicurazionelvaEscl": ,
          "SpeseConsRapidalvaEscl": ,
          "RigheOrdine": [
            {
              "IDProdotto": ,
              "NomeProdottoCompleto": ,
              "Quantita": ,
              "LivSconto": ,
              "Prezzo": ,
              "Offerta": 
            }
          ]
        }
      ]
    }
    Fai questo per ogni pagina del pdf e numero
    """