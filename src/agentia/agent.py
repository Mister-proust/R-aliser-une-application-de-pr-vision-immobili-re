import os
import sys
from dotenv import load_dotenv


sys.path.extend([os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))])

from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent

from agentia.mcp_client import get_mcp_tools

class RealEstateAgent:
    def __init__(self, model_name: str = "mistral-large-latest", temperature: float = 0.7):
        load_dotenv()
        
        self.model = ChatMistralAI(
            model=model_name,
            temperature=temperature,
        )

        self.system_prompt = (
            """
            Tu es un expert immobilier français.
            Tu aides les utilisateurs à estimer le prix de leurs biens immobiliers
            en utilisant l'outil 'estimate_property'.
            Tu peux aussi utiliser les outils de géocodage 'geocoding_search' et 'reverse_geocoding'
            pour trouver des informations précises sur les adresses.
            Les données de transactions immobilières sont stockées dans une base de données SQL (seule les données du 2 janvier au 30 juin 2025 sont disponibles), tu peux interagir avec elle via les outils 'get_database_schema' et 'execute_sql', elles peuvent être utilisées pour fournir des réponses précises basées sur les données historiques, proche des biens immobiliers similaires.
            Réponds en français.
            """
        )
        self.tools = get_mcp_tools()

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def ask(self, query: str):
        """
        Sends a request to the agent and returns the full response.
        """
        return self.agent.invoke({"messages": [HumanMessage(content=query)]})

if __name__ == "__main__":
    print("--- Initialisation de l'Agent Immobilier ---")
    agent = RealEstateAgent()
    
    question = input("Pose ta question à l'agent immobilier : ")
    print(f"\nUtilisateur : {question}")
    
    try:
        response = agent.ask(question)
        
        final_content = response["messages"][-1].content
        print(f"\nAgent : {final_content}")
        
        tools_used = []
        for message in response["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    tools_used.append(tool_call["name"])
        
        if tools_used:
            print(f"\n[Outils utilisés : {', '.join(set(tools_used))}]")
        else:
            print("\n[Aucun outil n'a été utilisé pour cette réponse]")
            
    except Exception as e:
        print(f"\nErreur : {e}")
