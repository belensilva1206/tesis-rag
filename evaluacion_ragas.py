# -*- coding: utf-8 -*-
"""
EVALUACION RAGAS - Gemini como evaluador (via endpoint compatible con OpenAI).
Se ejecuta en el entorno venv-ragas. Lee respuestas_agente.json (generado por
generar_respuestas.py en el entorno venv del framework).
"""
import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (LLMContextPrecisionWithReference, LLMContextRecall,
                           Faithfulness, ResponseRelevancy, AnswerCorrectness)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ARCHIVO_RESPUESTAS = "respuestas_agente.json"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def evaluar():
    if not os.path.exists(ARCHIVO_RESPUESTAS):
        raise FileNotFoundError(
            f"No existe {ARCHIVO_RESPUESTAS}. Primero corre 'python generar_respuestas.py' "
            "en el entorno del framework (venv).")

    with open(ARCHIVO_RESPUESTAS, encoding="utf-8") as f:
        filas = json.load(f)

        evaluador = LangchainLLMWrapper(ChatOpenAI(
        model="gemini-3.6-flash", api_key=GEMINI_API_KEY,
        base_url=BASE_URL, temperature=0.0))
    emb_eval = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="gemini-embedding-001", api_key=GEMINI_API_KEY, base_url=BASE_URL))

    dataset = EvaluationDataset.from_list(filas)
    print(f"Evaluando {len(filas)} respuestas con Gemini...\n")
    resultado = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall(),
                 Faithfulness(), ResponseRelevancy(), AnswerCorrectness()],
        llm=evaluador, embeddings=emb_eval)
    return resultado


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    resultado = evaluar()
    print("\n" + "=" * 60)
    print("RESULTADOS RAGAS")
    print("=" * 60)
    print(resultado)
    try:
        df = resultado.to_pandas()
        df.to_csv("resultados_ragas.csv", index=False, encoding="utf-8-sig")
        print("\nDetalle guardado en resultados_ragas.csv")
    except Exception as e:
        print("No se pudo exportar CSV:", e)