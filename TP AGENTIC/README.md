# AI-Sentiment-Image

Ce projet utilise **LangChain** et différents modèles de langage (**OpenAI, Ollama, Groq**) pour réaliser plusieurs tâches d'IA : analyse de sentiments par aspects, génération d'images et analyse de contenu visuel.

---

## 🚀 Fonctionnalités

### 1. Analyse de Sentiments (ABSA)
Le système analyse les avis clients en se concentrant sur des aspects spécifiques : `screen`, `keyboard`, `pad`.

* **Polarité :** Attribution d'un sentiment (`positive`, `negative` ou `neutral`).
* **Format :** Sortie structurée en JSON pour une intégration facile.

**Exemple d'entrée :**
> *"J'ai beaucoup aimé l'écran. La souris n'est pas bonne et le clavier m'a un peu dérangé."*

**Exemple de sortie JSON :**
```json
{
  "category": ["screen", "keyboard", "pad"],
  "polarity": ["positive", "negative", "neutral"]
}

### 2. Génération d’Images
Création d'images à partir de descriptions textuelles via le modèle `gpt-4o-mini` avec des outils de génération d'image intégrés.

```python
resp = llm_with_tools.invoke([
    SystemMessage(content=""),
    HumanMessage(content="je veux une image de Neymar junior da silva au Brasil")
])
2. Génération d’Images
Création d'images à partir de descriptions textuelles via le modèle gpt-4o-mini avec des outils de génération d'image intégrés.
resp = llm_with_tools.invoke([
    SystemMessage(content=""),
    HumanMessage(content="je veux une image de Neymar junior da silva au Brasil")
])
