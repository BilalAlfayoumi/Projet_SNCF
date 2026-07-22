# Plan de projet : agent RAG SNCF

## Objectif du document

Ce document sert de feuille de route technique. Il couvre trois volets : la stack (prototype puis version scalable), les pipelines de chaque brique du système, et le déploiement/MLOps. Chaque section indique le "pourquoi", pas juste le "quoi", pour que les choix restent compréhensibles même après plusieurs semaines.

---

## 1. Stack technique

### 1.1 Version prototype (semaines 1 à 4, objectif : ça marche, sur ta machine)

| Brique | Choix | Pourquoi |
|---|---|---|
| Backend | FastAPI | Rapide à monter, tu le connais déjà, async natif pour les appels LLM |
| Frontend | Streamlit | Zéro friction pour une démo, tu n'as pas à gérer de JS/React |
| Orchestration agent | LangGraph | Déjà exploré, adapté aux architectures multi-nodes |
| Vector store | FAISS (local, fichier) | Pas de serveur à gérer, suffisant pour un corpus de test |
| Embeddings | multilingual-e5 ou BGE-M3 via HuggingFace | Gratuit en local, bon score MTEB, gère le français |
| LLM | API Anthropic/OpenAI/Mistral en clé simple, ou Ollama en local pour ne rien payer pendant le dev | Pas besoin de gateway à ce stade |
| Stockage intermédiaire | JSON local + SQLite si besoin de requêtes simples | Cohérent avec ton pipeline parsing → JSON → embedding déjà pensé |

But de cette phase : valider que le raisonnement de l'agent fonctionne sur des vraies questions (celles qu'on a listées : perturbation + contournement, règles d'abonnement, etc.), sans te soucier de la charge.

### 1.2 Version scalable (cible finale, "des milliers d'utilisateurs")

| Brique | Choix | Pourquoi |
|---|---|---|
| Backend | FastAPI, conteneurisé (Docker) | Même code que le prototype, juste packagé différemment |
| Frontend | Streamlit, mais avec un bémol assumé | Voir encadré ci-dessous |
| Vector store | Qdrant (self-hosted sur AWS ou Qdrant Cloud managé) | FAISS ne gère pas bien la montée en charge concurrente ni les mises à jour incrémentales |
| Gateway LLM | Portkey ou LiteLLM | Rate limiting, fallback entre providers, cache de réponses, observabilité centralisée |
| Cache | Redis | Évite de recalculer une réponse pour une question déjà posée récemment |
| Auth | AWS Cognito ou clé API + rate limit par utilisateur | Nécessaire dès qu'il y a du trafic public |

**Encadré honnête sur Streamlit à l'échelle** : Streamlit crée une session serveur par utilisateur connecté, ce qui consomme de la RAM et du CPU par session active, pas par requête. Pour "des milliers d'utilisateurs simultanés", ce n'est pas l'outil qu'on choisirait en production réelle (on partirait sur du React/Next.js avec une API stateless). Ici, on garde Streamlit pour rester focalisé sur la partie IA (ce qui compte pour ta candidature), mais dans le document ou en entretien, tu peux mentionner que tu as conscience de cette limite et que tu la considères comme un compromis de scope assumé, pas une méconnaissance. Ça montre de la maturité technique plutôt qu'un angle mort.

---

## 2. Pipelines

### 2.1 Pipeline d'ingestion de données

```
API open data SNCF → parsing → chunking → JSON intermédiaire (checkpoint) → embedding → upsert Qdrant/FAISS
```

- **Extraction** : script Python qui appelle l'API SNCF (horaires, perturbations, gares), avec gestion des erreurs et retry.
- **Chunking** : découper les documents (fiches horaires, textes réglementaires) en morceaux cohérents, avec un peu d'overlap pour ne pas couper une info en deux.
- **Checkpoint JSON** : sauvegarde intermédiaire avant l'embedding, comme discuté, pour ne pas tout relancer si l'étape suivante plante.
- **Embedding** : vectorisation des chunks, une seule fois par chunk (pas à chaque requête).
- **Mise à jour** : les perturbations changent en temps réel, donc il faut un mécanisme de rafraîchissement (cron toutes les X minutes pour les données volatiles, ingestion ponctuelle pour les données stables comme les CGV).

### 2.2 Pipeline d'évaluation

- **Dataset de référence ("golden set")** : toi-même tu rédiges 30 à 50 questions/réponses représentatives (dont les cas complexes qu'on a identifiés), pour avoir une base de comparaison.
- **Métriques RAGAS** : faithfulness (la réponse est-elle fidèle aux documents récupérés), answer relevancy (la réponse répond-elle à la question), context precision/recall (le retrieval a-t-il trouvé les bons documents).
- **Test de régression** : à chaque changement de prompt, de modèle d'embedding ou de stratégie de chunking, tu relances l'éval sur le golden set pour vérifier que tu n'as pas dégradé la qualité.

### 2.3 Pipeline d'observabilité et monitoring

- **Tracing** : LangSmith ou Pydantic Logfire pour voir le détail de chaque appel (quel node a été activé, quels documents récupérés, temps de réponse par étape).
- **Logs applicatifs** : structlog (logs structurés en JSON) envoyés vers CloudWatch Logs.
- **Métriques** : latence par requête, taux d'erreur, coût par requête (nombre de tokens × prix), taux de cache hit.
- **Dashboards** : CloudWatch Dashboards pour commencer, Grafana si tu veux aller plus loin.
- **Alertes** : latence anormale, taux d'erreur au-dessus d'un seuil, budget LLM dépassé.

### 2.4 Pipeline sécurité et gateway

- **Gateway LLM (Portkey/LiteLLM)** : centralise les appels vers les providers, permet le rate limiting par utilisateur, le fallback automatique si un provider est en panne, et le cache de réponses identiques.
- **Auth API** : clé API simple en prototype, passage à OAuth/Cognito si tu veux simuler un vrai produit.
- **Secrets** : jamais de clé en dur dans le code, AWS Secrets Manager ou variables d'environnement chiffrées.

#### Guardrails, vue d'ensemble

Un guardrail protège dans deux directions : ce qui rentre dans le modèle (input) et ce qui en sort avant d'atteindre l'utilisateur (output). Il faut aussi distinguer ce qui protège le modèle lui-même (coût, abus, disponibilité) de ce qui protège le client final (contenu affiché, données personnelles). Les quatre quadrants ci-dessous couvrent l'ensemble.

**1. Guardrails côté input (avant que la requête n'atteigne le LLM)**

| Risque | Mécanisme |
|---|---|
| Injection de prompt | Détection de patterns suspects ("ignore les instructions précédentes", tentatives de changement de rôle), ou modèle classificateur dédié (Llama Guard, Prompt Guard) |
| Jailbreak | Même famille de détection, souvent couplée à un prompt système renforcé qui rappelle les limites du rôle de l'agent |
| Questions hors périmètre | Classification d'intention en amont : si la question ne concerne pas la SNCF, réponse de cadrage plutôt que de laisser le LLM improviser |
| Données personnelles dans la requête | Détection de PII (numéro de carte, email, téléphone) avant envoi au LLM, pour éviter qu'elles remontent dans des logs tiers |
| Longueur/format anormal | Limite de taille de requête, pour éviter les tentatives de saturation du contexte |

**2. Guardrails côté output (avant d'afficher la réponse à l'utilisateur)**

| Risque | Mécanisme |
|---|---|
| Hallucination | Vérification de grounding : la réponse doit être appuyée par les documents récupérés (c'est exactement ce que mesure la métrique faithfulness de RAGAS, réutilisable ici en filtre temps réel) |
| Contenu toxique ou inapproprié | Modération de sortie (API de modération du provider, ou modèle dédié) |
| Fuite de données personnelles | Scan de la réponse générée avant affichage, au cas où le LLM aurait reformulé une donnée sensible présente dans les documents source |
| Réponse hors sujet | Vérification de pertinence par rapport à la question posée |
| Format cassé | Validation de structure si tu attends du JSON ou un format précis (Pydantic côté FastAPI, avec retry si le parsing échoue) |

**3. Protection côté modèle (coût, abus, disponibilité)**

- Rate limiting par utilisateur (le gateway LLM le fait nativement).
- Plafond de budget (alerte ou coupure si le coût journalier dépasse un seuil).
- Timeout sur chaque appel, pour éviter qu'une requête bloquée ne monopolise des ressources.
- Fallback vers un modèle moins cher si le modèle principal est indisponible ou trop sollicité.

**4. Protection côté client (ce que l'utilisateur reçoit et son environnement)**

- Sanitization du HTML/Markdown généré avant rendu dans Streamlit, pour éviter l'injection de contenu malveillant dans l'interface.
- Traçabilité des sources citées dans la réponse (afficher d'où vient l'info, ce qui rassure l'utilisateur et permet de vérifier).
- Message de repli clair si un guardrail bloque la requête ou la réponse, plutôt qu'une erreur brute.

**Où placer ces guardrails techniquement** : la plupart des gateways LLM (Portkey, LiteLLM) permettent de brancher des guardrails input/output directement dans leur pipeline de requête, sans que tu aies à les coder à la main dans FastAPI. Pour le prototype, tu peux commencer avec des règles simples (regex, mots-clés, limites de taille) codées directement dans FastAPI, puis migrer vers le gateway une fois que tu passes à la version scalable.

---

## 3. Déploiement et MLOps

### 3.1 Prototype (rapide, peu coûteux)

- Déploiement sur Fly.io ou Render (même logique que ton projet Kardia), backend et frontend dans des services séparés.
- CI basique avec GitHub Actions : lint + tests unitaires à chaque push.

### 3.2 Version scalable sur AWS

```
GitHub Actions → build image Docker → push ECR → deploy ECS Fargate → ALB → utilisateurs
```

- **Conteneurisation** : un Dockerfile pour le backend FastAPI, un pour le frontend Streamlit.
- **Registry** : Amazon ECR pour stocker les images.
- **Orchestration de conteneurs** : ECS Fargate (pas besoin de gérer des serveurs EC2 toi-même) ou App Runner si tu veux encore plus simple.
- **Load balancer** : Application Load Balancer devant le service ECS, pour répartir le trafic et faire du health check.
- **Autoscaling** : règles d'autoscaling ECS basées sur le CPU ou la latence, pour absorber les pics.
- **Base vectorielle** : Qdrant sur un service ECS séparé, ou Qdrant Cloud managé si tu ne veux pas gérer l'infra toi-même.
- **CI/CD complet** : GitHub Actions qui, à chaque merge sur main, build l'image, la pousse sur ECR, puis déclenche un déploiement ECS.
- **Infrastructure as Code** : Terraform pour décrire toute l'infra (VPC, ECS, ALB, Qdrant), en bonus si le temps le permet. Ça montre une vraie maturité DevOps.

### 3.3 MLOps spécifique à la partie IA

- **Versioning des prompts** : garder une trace des versions de prompts système (un simple fichier versionné suffit, pas besoin d'outil dédié au début).
- **Versioning des données** : DVC si tu veux tracker les évolutions du corpus SNCF ingéré.
- **Réévaluation continue** : relancer le golden set d'évaluation après chaque changement de modèle d'embedding, de LLM ou de stratégie de chunking, avant de déployer en production.
- **Rollback** : garder les images Docker précédentes taguées, pour pouvoir revenir en arrière rapidement si une nouvelle version dégrade la qualité.

---

## Progression suggérée

1. Construire le prototype complet en local (ingestion → RAG → agent → Streamlit) sur un périmètre réduit (2-3 cas d'usage complexes).
2. Ajouter l'évaluation (golden set + RAGAS) pour objectiver la qualité avant de scaler quoi que ce soit.
3. Ajouter l'observabilité (tracing + logs) pendant que tu itères, pas après coup.
4. Conteneuriser et déployer la version simple (Fly.io/Render) pour avoir un lien de démo fonctionnel tôt.
5. Si le temps le permet, migrer vers AWS avec la stack scalable, en documentant les différences de choix entre les deux versions (c'est un excellent point à mentionner en entretien : tu sais pourquoi tu passerais de l'un à l'autre).
6. MCP en bonus final, une fois que tout le reste tourne.
