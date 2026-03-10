# Tester tool ia agentique avec le modèle Qwen2.5-7B-Instruct-Tool-Planning-v0.1 via Hugging Face
# Utilisation de l'API Hugging Face Inference

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

api_key = os.getenv("HF_API_KEY")  
if not api_key:
    raise RuntimeError("HF API key missing. Set HF_API_KEY in your .env file.")

# Initialisation du client Hugging Face
client = InferenceClient(token=api_key)

def get_weather(city: str):
    """
    A function that returns the weather in a given city.
    
    Args:
        city: The city to get the weather for.
    """
    import random
    
    return "sunny" if random.random() > 0.5 else "rainy"

def get_sunrise_sunset_times(city: str):
    """
    A function that returns the time of sunrise and sunset.
    
    Args:
        city: The city to get the sunrise and sunset times for.
    """
    # Simulation - en réalité vous pourriez utiliser une vraie API
    times = {
        "Los Angeles": ["6:30 AM", "6:45 PM"],
        "Paris": ["7:15 AM", "6:30 PM"],
        "Tokyo": ["5:45 AM", "5:30 PM"],
    }
    return times.get(city, ["6:00 AM", "6:00 PM"])

# Dictionnaire pour mapper les noms de fonctions aux fonctions réelles
available_functions = {
    "get_weather": get_weather,
    "get_sunrise_sunset_times": get_sunrise_sunset_times,
}

# Définition des tools au format OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "A function that returns the weather in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sunrise_sunset_times",
            "description": "A function that returns the time of sunrise and sunset at the present moment, for a given city, in the form of a list: [sunrise_time, sunset_time].",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the sunrise and sunset times for."
                    }
                },
                "required": ["city"]
            }
        }
    }
]

history_messages = [
    {"role": "user", "content": "Hi, can you tell me the time of sunrise in Los Angeles?"},
]

# Appel à l'API Hugging Face Inference
print("🤖 Envoi de la requête au modèle...")
response = client.chat_completion(
    model="Qwen/Qwen2.5-7B-Instruct:featherless-ai",
    messages=history_messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=256,
    temperature=0
)

print("\n" + "="*60)
print("📋 RÉPONSE DU MODÈLE")
print("="*60)

# Vérifier si le modèle veut appeler une fonction
if response.choices[0].message.tool_calls:
    print("✅ Le modèle a décidé d'appeler une fonction\n")
    
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        
        print(f"📞 Fonction appelée: {function_name}")
        print(f"📝 Arguments: {function_args}")
        
        # Exécuter la fonction si elle existe
        if function_name in available_functions:
            import json
            args = json.loads(function_args)
            function_to_call = available_functions[function_name]
            function_result = function_to_call(**args)
            
            print(f"✨ Résultat: {function_result}")
            
            # Optionnel: envoyer le résultat au modèle pour obtenir une réponse finale
            print("\n" + "-"*60)
            print("🔄 Envoi du résultat au modèle pour une réponse finale...")
            print("-"*60)
            
            # Ajouter l'appel de fonction et son résultat à l'historique
            history_messages.append(response.choices[0].message)
            history_messages.append({
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_result)
            })
            
            # Deuxième appel pour obtenir la réponse finale
            final_response = client.chat_completion(
                model="Qwen/Qwen2.5-7B-Instruct:featherless-ai",
                messages=history_messages,
                max_tokens=256,
                temperature=0
            )
            
            print(f"\n💬 Réponse finale du modèle:")
            print(f"{final_response.choices[0].message.content}")
        else:
            print(f"⚠️  Fonction {function_name} non trouvée")
else:
    print("💬 Réponse directe (sans appel de fonction):")
    print(response.choices[0].message.content)

print("\n" + "="*60)