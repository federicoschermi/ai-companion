from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph import graph_builder
from ai_companion.settings import settings

app = FastAPI()

# Abilita CORS per permettere l'accesso da qualunque origine (utile per mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint per gestire le richieste di chat dall'app mobile.
    Esso accetta un JSON con:
      - thread_id: ID della sessione (può essere "default" per iniziare)
      - message: Il testo del messaggio dell'utente
    Restituisce un JSON con la risposta generata.
    """
    try:
        body = await request.json()
        thread_id = body.get("thread_id", "default")
        message_text = body.get("message", "")
        if not message_text:
            return Response(content=json.dumps({"error": "Empty message"}), status_code=400)

        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
            # Compila il grafo con il checkpointer per la memoria a breve termine
            graph = graph_builder.compile(checkpointer=short_term_memory)
            # Invia il messaggio al grafo
            await graph.ainvoke(
                {"messages": [HumanMessage(content=message_text)]},
                {"configurable": {"thread_id": thread_id}},
            )
            # Ottieni lo stato aggiornato
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})

        response_text = state.values["messages"][-1].content
        return {"response": response_text}
    except Exception as e:
        logging.exception("Error processing chat endpoint")
        return Response(content=json.dumps({"error": str(e)}), status_code=500)