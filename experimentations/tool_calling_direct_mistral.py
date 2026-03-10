import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

# Nouvel import officiel remplaçant create_react_agent
from langchain.agents import create_agent

# 1. Chargement des variables du .env AVANT de les assigner
load_dotenv()

# LangSmith Configuration (on s'assure que les variables existent avant de les assigner)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


# 2. Définition du tool avec le décorateur LangChain
@tool
def estimate(address: str, type_bien: str, surface: int) -> str:
    """
    Estime le prix d'un bien immobilier.
    :param address: L'adresse du bien.
    :param type_bien: Le type de bien (maison ou appartement).
    :param surface: La surface en m2.
    :return: L'estimation du prix.
    """
    # Logique d'estimation métier ici
    return f"Estimation pour votre {type_bien} à {address} ({surface}m²) : 100 000€"


# 3. Initialisation du modèle Mistral compatible LangChain
# Remarque : j'utilise mistral-large-latest car c'est le modèle recommandé pour l'usage des tools
model = ChatMistralAI(
    model="mistral-large-latest", 
    temperature=0.7,
    max_tokens=2048,
)


# 4. Création de l'agent avec la nouvelle API
system_prompt = "Tu es un assistant utile dans le domaine de l'immobilier. Réponds en français."

# Utilisation de create_agent et de system_prompt (qui remplace state_modifier)
agent = create_agent(
    model=model,
    tools=[estimate],
    system_prompt=system_prompt
)


# 5. Invocation de l'agent avec la structure de message attendue
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Estime le prix de ma maison située au centre de Blois, elle fait 100m2")
        ]
    }
)


# 6. Affichage des messages générés et de l'appel d'outil
for message in response["messages"]:
    message.pretty_print()
