# Analyse de Sentiments et Génération d'Images avec LLMs

Ce projet utilise **LangChain** et des **modèles de langage OpenAI/Autres** pour effectuer plusieurs tâches liées à l'intelligence artificielle, notamment :

- L’analyse de sentiments basée sur des aspects spécifiques d’avis clients.
- La génération d’images à partir de descriptions textuelles.
- L’analyse de contenu d’images.

---

## Fonctionnalités

### 1. Analyse de Sentiments
- Analyse des avis sur les ordinateurs portables.
- Détection des aspects : `screen`, `keyboard`, `pad`.
- Attribution d’une polarité pour chaque aspect : `positive`, `negative` ou `neutral`.
- Les aspects non mentionnés sont automatiquement considérés comme `neutral`.
- Sortie organisée en JSON pour faciliter l’exploitation.

Exemple d’avis :"j'ai beaucoup aimé l'écran. La souris n'est pas bonne et le clavier m'a un peu dérangé."

Exemple de sortie :
```json
{
  "category": ["screen", "keybloard", "pad"],
  "polarity": ["positive", "negative", "neutral"]
}
2. Génération d’Images
Création d’images à partir de descriptions textuelles.
Utilisation du modèle gpt-4o-mini avec des outils de génération d’image intégrés.

Exemple :

resp = llm_with_tools.invoke([
    SystemMessage(""),
    HumanMessage(content="je veux une image de Neymar junior da silva au Brasil")
])
3. Analyse d’Images
Permet d’envoyer une image au modèle et d’obtenir une description ou une analyse de son contenu.
Conversion des images en base64 pour l’envoi.

Exemple :

path = '2018965.jpg'
with open(path, 'rb') as file:
    img = base64.b64encode(file.read()).decode('utf-8')

resp = llm5.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Explique-moi ce qu'il y a dans l'image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
    ])
])
Installation
Cloner ce dépôt :
git clone https://github.com/votre-utilisateur/ai-sentiment-image.git
cd ai-sentiment-image
Installer les dépendances :
pip install -r requirements.txt
Créer un fichier .env avec vos clés API :
OPENAI_API_KEY=your_openai_api_key
OLLAMA_API_KEY=your_ollama_api_key
Utilisation
Lancer un notebook Python ou un script pour tester :
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

response = llm.invoke([
    {"role": "system", "content": "you are a helpful assistant."},
    {"role": "user", "content": "c'est quoi l'enset"}
])
print(response.content)
Pour l’analyse de sentiments :
resp = llm6.invoke(input=[
    SystemMessage(content=system_message),
    HumanMessage(content="j'ai beaucoup aimé l'écran. La souris n'est pas bonne et le clavier m'a un peu dérangé.")
])
print(resp.content)
Technologies Utilisées
Python 3.10+
LangChain
OpenAI GPT-4o / GPT-5.2
Ollama, Groq (optionnel pour les modèles alternatifs)
IPython pour l’affichage des résultats
Base64 pour l’envoi d’images
Auteur

Ake Mobio Ivan Junior

Stage en base de données et passionné par les LLMs et l’IA agentique.
Licence

Ce projet est sous licence MIT.
