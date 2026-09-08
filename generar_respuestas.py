# -*- coding: utf-8 -*-
"""
============================================================================
GENERAR RESPUESTAS DEL AGENTE  (paso 1 de la evaluacion)
Se ejecuta en el entorno del framework (venv, el que ya funciona).
Produce 'respuestas_agente.json' que luego consume evaluacion_ragas.py.
============================================================================

Uso (en el entorno venv del framework):
    python generar_respuestas.py
"""
import json
from framework_local import construir_sistema, preguntar

# ---------------------------------------------------------------------------
# CONJUNTO DE REFERENCIA DORADO
# Reemplaza cada 'ground_truth' por la respuesta redactada por tu panel de
# abogados (Seccion 4.4.4). Anade tantas preguntas como necesites.
# ---------------------------------------------------------------------------
DATASET_DORADO = [
    {
        "question": "Que se considera un dato personal sensible?",
        "ground_truth": (
            "Son datos personales sensibles aquellos que se refieren a las "
            "caracteristicas fisicas o morales de las personas o a hechos o "
            "circunstancias de su vida privada o intimidad, que revelen origen "
            "etnico o racial, afiliacion politica, sindical o gremial, situacion "
            "socioeconomica, convicciones ideologicas o filosoficas, creencias "
            "religiosas, datos relativos a la salud, al perfil biologico humano, "
            "datos biometricos, y la informacion relativa a la vida sexual, "
            "orientacion sexual e identidad de genero."),
    },
    {
        "question": "Que multas contempla la Ley 21.719?",
        "ground_truth": (
            "La Ley 21.719 establece multas que pueden llegar hasta las 20.000 "
            "UTM o hasta el 4% de los ingresos anuales de la empresa infractora, "
            "segun la gravedad de la infraccion."),
    },
    {
        "question": "En que casos se puede tratar datos personales sensibles?",
        "ground_truth": (
            "El tratamiento de datos personales sensibles solo puede realizarse "
            "cuando el titular manifiesta su consentimiento en forma expresa, o "
            "en los casos especificos que la ley autoriza expresamente."),
    },
]


def main():
    print("Construyendo el sistema (framework)...")
    construir_sistema()

    filas = []
    for i, item in enumerate(DATASET_DORADO, 1):
        print(f"  [{i}/{len(DATASET_DORADO)}] {item['question']}")
        r = preguntar(item["question"])
        filas.append({
            "user_input": item["question"],
            "response": r["respuesta"],
            "retrieved_contexts": [c["texto"] for c in r["chunks"]],
            "reference": item["ground_truth"],
        })

    with open("respuestas_agente.json", "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)
    print(f"\nGuardadas {len(filas)} respuestas en respuestas_agente.json")
    print("Ahora, en el entorno venv-ragas, corre:  python evaluacion_ragas.py")


if __name__ == "__main__":
    main()
