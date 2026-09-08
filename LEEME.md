# Framework RAG — Ley 21.719 (versión local)

## Puesta en marcha

1. Crea y activa el entorno virtual:
   - Windows:  `python -m venv venv` y luego `venv\Scripts\activate`
   - Mac/Linux: `python -m venv venv` y luego `source venv/bin/activate`

2. Instala dependencias:
   `pip install -r requirements.txt`

3. Instala Ollama (app de escritorio) desde https://ollama.com y descarga el modelo:
   `ollama pull qwen3:8b`

4. Copia `.env.ejemplo` a `.env` y pon tu clave real de Gemini.

5. Edita `RUTA_LEYES` en `framework_local.py` para que apunte a la carpeta
   con los PDF de las leyes 19.628 y 21.719.

## Ejecutar

- Sistema + prueba:  `python framework_local.py`
- Evaluación RAGAS:  `python evaluacion_ragas.py`

## Hardware
BGE-M3, el reranker y Qwen3 rinden mejor con GPU NVIDIA. En CPU funcionan
pero lentos; si no tienes GPU, considera dejar la ejecución pesada en Colab.
En `framework_local.py`, si estás en CPU, cambia `use_fp16=True` a `False`.
