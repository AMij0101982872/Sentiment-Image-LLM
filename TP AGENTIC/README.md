# AI-Sentiment-Image

Un projet Python exploitant LangChain et plusieurs modèles de langage (OpenAI, Ollama, Groq) pour réaliser des tâches d'intelligence artificielle : tokenisation, analyse de sentiments, génération et analyse d'images.

---

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Technologies utilisées](#technologies-utilisées)
- [Auteur](#auteur)
- [Licence](#licence)

---

## Fonctionnalités

### Tokenisation avec Tiktoken

Analyse et décomposition de prompts en tokens à l'aide du tokenizer de GPT-4o.
```python
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4o")
prompt = "vous etes un experte en IA agentique"
tokens = tokenizer.encode(prompt)

print(len(tokens))
for token in tokens:
    t = tokenizer.decode_single_token_bytes(token).decode("utf-8")
    print(t, end="|")
```

---

### Appels LLM multi-providers

Invocation de modèles de langage via LangChain avec différents providers.

**OpenAI :**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke([
    {"role": "system", "content": "you are a helpful assistant. the output should be in markdown"},
    {"role": "user", "content": "c'est quoi l'enset"}
])
```

**Ollama (local) :**
```python
from langchain_ollama import ChatOllama

llm2 = ChatOllama(model="qwen3.5:cloud")
response = llm2.invoke([
    {"role": "system", "content": "you are a helpful assistant. the output should be in markdown"},
    {"role": "user", "content": "c'est quoi l'enset"}
])
```

**Groq :**
```python
from langchain_groq import ChatGroq

llm3 = ChatGroq(model="openai/gpt-oss-120b")
response = llm3.invoke([
    {"role": "system", "content": "you are a helpful assistant. the output should be in markdown"},
    {"role": "user", "content": "c'est quoi l'enset"}
])
```

---

### Generation d'images

Création d'images à partir de descriptions textuelles via `gpt-4o-mini` et l'outil de génération intégré.
```python
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from IPython.display import Image
import base64

llm4 = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm4.bind_tools([
    {"type": "image_generation", "quality": "high"}
])

resp = llm_with_tools.invoke([
    SystemMessage(""),
    HumanMessage(content="je veux une image de Neymar junior da silva au brasil")
])

Image(base64.b64decode(resp.content_blocks[0]['base64']))
```

---

### Analyse d'images

Description du contenu d'une image encodée en base64, via `gpt-5.2`.
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

llm5 = ChatOpenAI(model="gpt-5.2")

path = '2018965.jpg'
with open(path, 'rb') as file:
    img = base64.b64encode(file.read()).decode('utf-8')

resp = llm5.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Explique-moi ce qu'il y a dans l'image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
    ])
])
```

---

### Analyse de sentiments basee sur les aspects

Identification des aspects `screen`, `keyboard`, `pad` dans un avis client, avec attribution d'une polarité `positive`, `negative` ou `neutral`. Sortie structurée en JSON.
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

system_message = """
vous etre un expert dans l'analyse des sentiments.
Effectuez une analyse de sentiments basée sur les aspects des avis concernant les ordinateurs portables.
Chaque avis peut comporter un ou plusieurs des aspects suivants : screen, keyboard et pad.
Pour chaque avis :
- Identifiez la présence d'au moins un des trois aspects.
- Attribuez une polarité (positive, negative ou neutral) à chaque aspect.
Organisez votre réponse dans un objet JSON :
  - category : [liste des aspects]
  - polarity : [liste des polarités correspondantes]
Si un aspect est absent, supposez que la polarité est neutre.
"""

llm6 = ChatOpenAI(model="gpt-5.2")

resp = llm6.invoke(input=[
    SystemMessage(content=system_message),
    HumanMessage(content="j'ai beaucoup aimé l'écran. La souris n'est pas bonne et le clavier m'a un peu dérangé.")
])

sentiment = json.loads(resp.content.replace("```json", "").replace("```", " "))
print(sentiment['category'])
print(sentiment['polarity'])
```

**Exemple de sortie JSON :**
```json
{
  "category": ["screen", "keyboard", "pad"],
  "polarity": ["positive", "negative", "neutral"]
}
```

---

## Installation

1. Cloner le depot :
```bash
git clone https://github.com/votre-utilisateur/AI-Sentiment-Image.git
cd AI-Sentiment-Image
```

2. Installer les dependances :
```bash
pip install -r requirements.txt
```

3. Creer un fichier `.env` a la racine du projet :
```env
OPENAI_API_KEY=your_openai_api_key
OLLAMA_API_KEY=your_ollama_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Technologies utilisees

| Technologie | Role |
|---|---|
| Python 3.10+ | Langage principal |
| Tiktoken | Tokenisation des prompts |
| LangChain | Orchestration des LLM |
| OpenAI GPT-4o / GPT-5.2 | Modeles principaux |
| Ollama | Execution locale de modeles |
| Groq | Inference rapide |
| IPython | Affichage interactif |
| Base64 | Encodage des images |

---

## Auteur

Ake Mobio Ivan Junior  
Stagiaire en base de donnees — passionne par l'IA agentique

---

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de details.
