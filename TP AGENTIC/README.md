#  AI-Sentiment-Image

Un projet Python exploitant **LangChain** et plusieurs modèles de langage (OpenAI, Ollama, Groq) pour réaliser des tâches d'intelligence artificielle avancées : analyse de sentiments, génération et analyse d'images.

---

##  Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Technologies utilisées](#-technologies-utilisées)
- [Auteur](#-auteur)


---

##  Fonctionnalités

###  Analyse de Sentiments

- Détection des aspects : `screen`, `keyboard`, `pad`
- Attribution d'une polarité : `positive`, `negative` ou `neutral`
- Les aspects non mentionnés sont automatiquement considérés comme `neutral`
- Sortie structurée en JSON

**Exemple d'avis :**
```
j'ai beaucoup aimé l'écran. La souris n'est pas bonne et le clavier m'a un peu dérangé.
```

**Sortie JSON :**
```json
{
  "category": ["screen", "keyboard", "pad"],
  "polarity": ["positive", "negative", "neutral"]
}
```

---

###  Génération d'Images

- Création d'images à partir de descriptions textuelles
- Utilisation du modèle `gpt-4o-mini` avec des outils de génération intégrés

**Exemple :**
```python
resp = llm_with_tools.invoke([
    SystemMessage(""),
    HumanMessage(content="je veux une image de Neymar junior da silva au Brasil")
])
```

---

###  Analyse d'Images

- Analyse et description du contenu d'images
- Envoi des images encodées en base64

**Exemple :**
```python
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

##  Installation


1. **Configurer les variables d'environnement** — créer un fichier `.env` à la racine :
```env
OPENAI_API_KEY=openai_api_key
OLLAMA_API_KEY=ollama_api_key
```

---

##  Utilisation

**Exemple simple avec OpenAI GPT :**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke([
    {"role": "system", "content": "you are a helpful assistant."},
    {"role": "user", "content": "c'est quoi l'enset"}
])
print(response.content)
```

**Exemple pour l'analyse de sentiments :**
```python
resp = llm6.invoke(input=[
    SystemMessage(content=system_message),
    HumanMessage(content="j'ai beaucoup aimé l'écran. La souris n'est pas bonne et le clavier m'a un peu dérangé.")
])
print(resp.content)
```

---

##  Technologies utilisées

| Technologie | Description |
|---|---|
| Python 3.10+ | Langage principal |
| LangChain | Framework d'orchestration LLM |
| OpenAI GPT-4o / GPT-4o-mini | Modèles de langage principaux |
| Ollama | Exécution de modèles en local (optionnel) |
| Groq | Inférence rapide (optionnel) |
| IPython | Affichage interactif |
| Base64 | Encodage des images |

---

## 👤 Auteur

**Ake Mobio Ivan Junior**  


---

