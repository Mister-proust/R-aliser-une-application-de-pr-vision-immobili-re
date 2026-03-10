import os
import sys
from dotenv import load_dotenv


sys.path.extend([os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))])

from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent

from agentia.estimation_tool import estimate_property
from agentia.geocoding_tool import geocoding_search, reverse_geocoding 

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
            Réponds en français.
            """
        )

        self.agent = create_agent(
            model=self.model,
            tools=[estimate_property, geocoding_search, reverse_geocoding],
            system_prompt=self.system_prompt
        )

    def ask(self, query: str):
        """
        Envoie une requête à l'agent et retourne la réponse complète.
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
