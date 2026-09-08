from framework_local import construir_sistema, recuperar_contexto, qwen, SYS_GENERACION, SYS_VERIFICACION
construir_sistema()

pregunta = "¿qué se considera un dato personal sensible?"
ruta, chunks = recuperar_contexto(pregunta, k=5)
contexto = "\n\n".join(f"[{c['ley']} - {c['articulo']}]\n{c['texto']}" for c in chunks)

print("\n===== BORRADOR QUE GENERA QWEN3 =====")
borrador = qwen(f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}", system=SYS_GENERACION)
print(borrador)

print("\n===== QUÉ DEVUELVE EL VERIFICADOR =====")
verif = qwen(f"CONTEXTO:\n{contexto}\n\nRESPUESTA:\n{borrador}", system=SYS_VERIFICACION, formato="json")
print(repr(verif))