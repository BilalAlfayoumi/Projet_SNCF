# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## État du projet

Projet en phase de démarrage : agent conversationnel RAG sur les données open data SNCF (portfolio pour une recherche d'alternance en IA). **Aucun code n'existe encore.** Le seul fichier est `plan-projet-sncf.md`, la feuille de route technique qui fait autorité sur tous les choix d'architecture. La lire avant toute décision structurante, et la mettre à jour si un choix change.

## Architecture cible (résumé du plan)

Deux versions successives, même code cœur :

- **Prototype (priorité actuelle)** : FastAPI (backend) + Streamlit (frontend) + LangGraph (orchestration agent) + FAISS local (vector store) + embeddings HuggingFace (multilingual-e5 ou BGE-M3, corpus en français). LLM via API ou Ollama en local.
- **Version scalable (plus tard)** : mêmes briques conteneurisées (Docker), FAISS remplacé par Qdrant, gateway LLM (LiteLLM/Portkey), Redis, déploiement AWS ECS Fargate. Ne pas anticiper cette version dans le code du prototype, sauf à garder les interfaces propres (ex. abstraction du vector store).

Pipeline d'ingestion avec checkpoint intermédiaire :

```
API open data SNCF → parsing → chunking → JSON intermédiaire (checkpoint) → embedding → upsert FAISS
```

Le checkpoint JSON est un choix délibéré : il évite de tout relancer si l'embedding plante. Ne pas le supprimer pour "simplifier".

## Conventions et contraintes

- Machine de dev : Mac Apple Silicon, pas de CUDA. Pour l'inférence ou l'embedding local : MLX, Ollama, ou modèles HuggingFace CPU/MPS. Jamais de dépendances CUDA-only (bitsandbytes, QLoRA classique).
- LangGraph/LangChain : LCEL et `create_agent` uniquement, pas les API dépréciées (`LLMChain`, `create_tool_calling_agent`, `create_openai_tools_agent`).
- Secrets (clés API SNCF, LLM) : variables d'environnement uniquement, jamais en dur.
- L'évaluation (golden set de 30-50 Q/R + métriques RAGAS) doit être mise en place tôt, avant toute optimisation : chaque changement de prompt, d'embedding ou de chunking se valide en relançant l'éval.

## Ordre de construction prévu

1. Prototype local complet (ingestion → RAG → agent LangGraph → Streamlit) sur un périmètre réduit (2-3 cas d'usage complexes : perturbation + contournement, règles d'abonnement).
2. Évaluation (golden set + RAGAS).
3. Observabilité (tracing LangSmith ou Logfire, logs structurés).
4. Conteneurisation + déploiement démo (Fly.io/Render, CI GitHub Actions lint + tests).
5. Migration AWS scalable (optionnelle).
6. Serveur MCP en bonus final.

## Commandes

Aucune commande de build/test/lint n'existe encore. Quand le squelette Python sera créé, documenter ici l'installation des dépendances, le lancement du backend et du frontend, et l'exécution des tests.
