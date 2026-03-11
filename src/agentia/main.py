import sys
import os
import time
from dotenv import load_dotenv
import gradio as gr

sys.path.extend([
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
])

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware

from agentia.estimation_tool import estimate_property
from agentia.geocoding_tool import geocoding_search, reverse_geocoding
from agentia.tool_bdd import get_database_schema, execute_sql

load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
    if os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7,
    max_tokens=2048,
)

system_prompt = """
Tu es un expert immobilier français.
Tu aides les utilisateurs à estimer le prix de leurs biens immobiliers
en utilisant l'outil 'estimate_property'.
Tu peux aussi utiliser les outils de géocodage 'geocoding_search' et 'reverse_geocoding'
pour trouver des informations précises sur les adresses.
Les données de transactions immobilières sont stockées dans une base de données SQL, tu peux interagir avec elle via les outils 'get_database_schema' et 'execute_sql', elles peuvent être utilisées pour fournir des réponses précises basées sur les données historiques, proche des biens immobiliers similaires.
Réponds en français.
"""

agent = create_agent(
    model=model,
    tools=[estimate_property, geocoding_search, reverse_geocoding, get_database_schema, execute_sql],
    system_prompt=system_prompt,
   
   
   
   
   
   
   
   
   
   
   
   
)

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content)

def format_history(history):
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and not msg.get("metadata"):
            messages.append(AIMessage(content=msg["content"]))
    return messages

with gr.Blocks(title="Agent Immobilier Expert 🏠", fill_height=True) as demo:
    gr.Markdown("# Agent Immobilier Expert 🏠")
    busy = gr.State(False)

    chatbot = gr.Chatbot(
        label="Conversation",
        show_label=False,
        scale=1,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Votre question",
            placeholder="Ex: Estime ma maison de 100m² à Tours...",
            scale=4
        )
        submit_btn = gr.Button("Envoyer", variant="primary", scale=1)

    gr.ClearButton([msg, chatbot], value="Effacer la conversation")

    gr.Examples(
        examples=[
            "Estime ma maison de 100m² avec 4 pièces à Blois.",
            "Quelles sont les coordonnées du MAME à Tours ?"
        ],
        inputs=msg
    )

    def user(user_message, history, is_busy):
        history = history or []
        if is_busy:
            return gr.update(), gr.update(), history, True
        if not user_message or not user_message.strip():
            return gr.update(value=""), gr.update(interactive=True), history, False
        history = history + [{"role": "user", "content": user_message}]
        return gr.update(value="", interactive=False), gr.update(interactive=False), history, True

    def bot(history):
        history = history or []
        langchain_history = format_history(history)

        try:
            final_message_index = None
            tool_message_index = None

            for step in agent.stream(
                {"messages": langchain_history},
                stream_mode="updates"
            ):
                if "model" in step:
                    msg_obj = step["model"]["messages"][-1]

                   
                    if getattr(msg_obj, "tool_calls", None):
                        for tool_call in msg_obj.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            texte_appel = f"**Appel de `{tool_name}`** avec les paramètres : `{tool_args}`\n\n"
                            
                            if tool_message_index is None:
                               
                                history.append({
                                    "role": "assistant",
                                    "content": texte_appel,
                                    "metadata": {
                                        "title": "Étapes de recherche",
                                        "status": "pending"
                                    }
                                })
                                tool_message_index = len(history) - 1
                            else:
                               
                                history[tool_message_index]["content"] += texte_appel
                            yield history
                            
                   
                    else:
                        final_text = extract_text(msg_obj.content)
                        if final_text:
                           
                            if tool_message_index is not None:
                                history[tool_message_index]["metadata"]["status"] = "done"
                                
                            if final_message_index is None:
                                history.append({"role": "assistant", "content": ""})
                                final_message_index = len(history) - 1
                                
                            current = ""
                            for char in final_text:
                                current += char
                                history[final_message_index]["content"] = current
                                time.sleep(0.01)
                                yield history

               
                if "tools" in step:
                    msg_obj = step["tools"]["messages"][-1]
                    tool_output = extract_text(msg_obj.content)
                    
                    if tool_message_index is not None:
                       
                        history[tool_message_index]["content"] += f"> **Résultat :** {tool_output}\n\n---\n\n"
                        yield history

        except Exception as e:
            history.append({
                "role": "assistant",
                "content": f"Désolé, une erreur est survenue : {str(e)}"
            })
            yield history

    def unlock():
        return gr.update(interactive=True), gr.update(interactive=True), False

    gr.on(
        triggers=[msg.submit, submit_btn.click],
        fn=user,
        inputs=[msg, chatbot, busy],
        outputs=[msg, submit_btn, chatbot, busy],
        trigger_mode="once",
        concurrency_limit=1,
        queue=True
    ).then(
        bot,
        inputs=chatbot,
        outputs=chatbot,
        queue=True,
        concurrency_limit=1
    ).then(
        unlock,
        inputs=None,
        outputs=[msg, submit_btn, busy],
        queue=False
    )

    demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
