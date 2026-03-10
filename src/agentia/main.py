import sys
import os
import time
from dotenv import load_dotenv
import gradio as gr

sys.path.extend([os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))])

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
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

system_prompt = (
    """
    Tu es un expert immobilier français.
    Tu aides les utilisateurs à estimer le prix de leurs biens immobiliers
    en utilisant l'outil 'estimate_property'.
    Tu peux aussi utiliser les outils de géocodage 'geocoding_search' et 'reverse_geocoding'
    pour trouver des informations précises sur les adresses.
    Les données de transactions immobilières sont stockées dans une base de données SQL, tu peux interagir avec elle via les outils 'get_database_schema' et 'execute_sql', elles peuvent être utilisées pour fournir des réponses précises basées sur les données historiques, proche des biens immobiliers similaires.
    Réponds en français.
    """
)

# Création de l'agent
agent = create_agent(
    model=model,
    tools=[estimate_property, geocoding_search, reverse_geocoding, get_database_schema, execute_sql],
    system_prompt=system_prompt
)

def format_history(history):
    """Convertit l'historique Gradio en messages LangChain."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

with gr.Blocks(title="Agent Immobilier Expert 🏠", fill_height=True) as demo:
    gr.Markdown("# Agent Immobilier Expert 🏠")
    chatbot = gr.Chatbot(label="Conversation", show_label=False, scale=1)
    
    with gr.Row():
        msg = gr.Textbox(
            label="Votre question",
            placeholder="Ex: Estime ma maison de 100m² à Tours...",
            scale=4
        )
        submit_btn = gr.Button("Envoyer", variant="primary", scale=1)

    clear = gr.ClearButton([msg, chatbot], value="Effacer la conversation")

    gr.Examples(
        examples=[
            "Estime ma maison de 100m² avec 4 pièces à Blois.",
            "Quelles sont les coordonnées du MAME à Tours ?"
        ],
        inputs=msg
    )

    def user(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history):
        langchain_history = format_history(history)
        
        try:
            final_text = ""
            for step in agent.stream({"messages": langchain_history}):
                # Visualisation de l'appel des outils
                if "agent" in step:
                    msg_obj = step["agent"]["messages"][-1]
                    if hasattr(msg_obj, "tool_calls") and msg_obj.tool_calls:
                        for tool_call in msg_obj.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            history.append({
                                "role": "assistant", 
                                "content": f"Réflexion : Utilisation de {tool_name} avec {tool_args}",
                                "metadata": {"title": f"🛠️ Outil : {tool_name}"}
                            })
                            yield history
                
                elif "tools" in step:
                    msg_obj = step["tools"]["messages"][-1]
                    tool_output = msg_obj.content
                    history.append({
                        "role": "assistant", 
                        "content": tool_output,
                        "metadata": {"title": "✅ Résultat de l'outil"}
                    })
                    yield history
                
                if "agent" in step:
                    msg_obj = step["agent"]["messages"][-1]
                    if not (hasattr(msg_obj, "tool_calls") and msg_obj.tool_calls):
                        final_text = msg_obj.content

            if final_text:
                history.append({"role": "assistant", "content": ""})
                for char in final_text:
                    history[-1]["content"] += char
                    time.sleep(0.01) 
                    yield history
            else:
                res = agent.invoke({"messages": langchain_history})
                final_text = res["messages"][-1].content
                history.append({"role": "assistant", "content": ""})
                for char in final_text:
                    history[-1]["content"] += char
                    time.sleep(0.01)
                    yield history

        except Exception as e:
            history.append({"role": "assistant", "content": f"Désolé, une erreur est survenue : {str(e)}"})
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

if __name__ == "__main__":
    demo.launch()
