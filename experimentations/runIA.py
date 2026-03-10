import os
import re
import json
import uuid
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import create_agent

# 1. Chargement des variables d'environnement
load_dotenv()

# Configuration LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# 2. Définition du tool
@tool
def estimate(address: str, type_bien: str, surface: int) -> str:
    """
    Estime le prix d'un bien immobilier.
    :param address: L'adresse du bien.
    :param type_bien: Le type (maison ou appartement).
    :param surface: La surface habitable en m2.
    """
    return f"Estimation pour votre {type_bien} à {address} ({surface}m²) : 100 000€"

tools = [estimate]

# 3. Initialisation du modèle Qwen 2.5 officiel via HuggingFace
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",  # Modèle officiel Qwen
    huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    temperature=0.1,
    task="text-generation",
    max_new_tokens=512
)

# 4. Classe personnalisée pour traduire le format Qwen en format LangChain
class QwenChatHuggingFace(ChatHuggingFace):
    def invoke(self, *args, **kwargs):
        msg = super().invoke(*args, **kwargs)
        
        # Interception uniquement si c'est une réponse de l'IA sans tool_calls natifs
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = str(msg.content)
            
            tool_name = None
            tool_args = {}
            
            # Recherche du format XML typique de Qwen : <tool_call> {"name": "...", "arguments": {...}} </tool_call>
            match_xml = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
            # Recherche du format alternatif (Markdown/JSON)
            match_json = re.search(r'```(?:json)?\s*(\{.*?"name".*?"arguments".*?\})\s*```', content, re.DOTALL)

            try:
                if match_xml:
                    data = json.loads(match_xml.group(1))
                    tool_name = data.get("name")
                    tool_args = data.get("arguments", {})
                elif match_json:
                    data = json.loads(match_json.group(1))
                    tool_name = data.get("name")
                    tool_args = data.get("arguments", {})
            except json.JSONDecodeError:
                pass # Si le JSON est mal formé, on laisse l'agent gérer l'erreur de parsing

            # Si on a trouvé un outil valide, on le convertit au format LangGraph
            if tool_name:
                msg.tool_calls = [{
                    "name": tool_name,
                    "args": tool_args,
                    "id": f"call_{uuid.uuid4().hex[:8]}" # LangGraph exige un ID unique
                }]
                msg.content = "" # On vide le texte pour forcer l'exécution de l'outil
                
        return msg

# On applique notre surcouche
chat_model = QwenChatHuggingFace(llm=llm)

# 5. Création de l'agent
system_prompt = "Tu es un assistant utile dans le domaine de l'immobilier. Réponds en français."

agent = create_agent(
    model=chat_model,
    tools=tools,
    system_prompt=system_prompt
)

# 6. Exécution de l'agent
query = "Estime le prix de ma maison située au centre de Blois, elle fait 100m2."

response = agent.invoke(
    {
        "messages": [
            HumanMessage(content=query)
        ]
    }
)

# 7. Affichage des résultats finaux
for message in response["messages"]:
    message.pretty_print()
