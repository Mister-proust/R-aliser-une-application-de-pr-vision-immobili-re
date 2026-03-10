import sys
import os
from dotenv import load_dotenv
import gradio as gr

sys.path.extend([os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))])

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from agentia.estimation_tool import estimate_property
from agentia.geocoding_tool import geocoding_search, reverse_geocoding

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
    Réponds toujours en français de manière professionnelle.
    """
)

agent = create_agent(
    model=model,
    tools=[estimate_property, geocoding_search, reverse_geocoding],
    system_prompt=system_prompt
)

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                history_langchain_format.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_langchain_format.append(AIMessage(content=msg["content"]))
        elif isinstance(msg, (list, tuple)):
            user_msg, bot_msg = msg
            if user_msg:
                history_langchain_format.append(HumanMessage(content=user_msg))
            if bot_msg:
                history_langchain_format.append(AIMessage(content=bot_msg))
                
    history_langchain_format.append(HumanMessage(content=message))
    
    response = agent.invoke({"messages": history_langchain_format})
    
    return response["messages"][-1].content

demo = gr.ChatInterface(
    predict,
    title="Agent Immobilier Expert 🏠",
    description="Posez vos questions sur l'estimation de biens et le géocodage d'adresses en France.",
    examples=[
        "Estime ma maison de 100m² avec 4 pièces à Blois.",
        "Quelles sont les coordonnées du 4 Impasse de l'ancienne école normale à Tours ?"
    ]
)

if __name__ == "__main__":
    demo.launch()
