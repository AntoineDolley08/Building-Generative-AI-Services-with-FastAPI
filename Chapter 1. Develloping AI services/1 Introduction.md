# Chapter 1 — Why Generative AI Services Will Power Future Applications

## Shift fondamental : des règles codées aux modèles

- **Avant** : automatisation = coder manuellement des règles métier (ex : détection de spam avec des règles écrites à la main → fastidieux et fragile)
- **Maintenant** : on entraîne un modèle qui comprend les nuances du processus métier et surpasse les règles codées
- **GenAI vs AI traditionnelle** : l'IA traditionnelle fait de la prédiction/classification, la GenAI **produit du contenu multimédia** (texte, code, images, audio, vidéo)

## 7 capacités clés de la GenAI pour les applications futures

### 1. Faciliter le processus créatif
- Le processus créatif est cognitivement coûteux (writer's block, difficulté à visualiser, besoin de recherche préalable)
- La GenAI aide à **connecter des idées** issues d'un vaste corpus de connaissances humaines
- Cas d'usage : visualiser des concepts difficiles à imaginer (ex : description de scène → DALL-E génère l'image)
- **Implication produit** : proposer des suggestions pour aider l'utilisateur à démarrer et prendre de l'élan

### 2. Suggérer des solutions contextuellement pertinentes
- Les problèmes de niche nécessitent beaucoup de recherche et de contexte pour être résolus
- **Le contexte réduit l'espace des solutions possibles**
- Prompts pauvres en contexte → réponses génériques (même logique que les requêtes courtes sur Google)
- **Prompts riches en contexte → réponses pertinentes et spécifiques**
- Exemple concret : les devs passent de Stack Overflow (recherche par mots-clés, espérer trouver le même contexte) aux outils GenAI (description du contexte → solutions générées)
- Stack Overflow a vu ~60% de baisse des questions posées vs 2018
- Les sites Q&A restent précieux pour les discussions, la curation humaine et la vérification des sources

### 3. Personnaliser l'expérience utilisateur
- Le LLM agit comme **assistant personnel** : il pose des questions pour mapper les préférences vers un catalogue
- Exemples : chatbot voyage qui prépare un itinéraire, recommandations adaptées au profil utilisateur
- Éducation : adapter l'explication au niveau de l'élève
- Gaming/VR : générer des narratifs et environnements dynamiques en temps réel selon les choix du joueur

### 4. Réduire les délais de résolution des requêtes clients
- Problème : volume élevé → files d'attente longues, coûts de formation du personnel
- GenAI chatbots vs chatbots traditionnels (règles/scripts) :
  - Meilleure compréhension du contexte conversationnel
  - Réponses dynamiques et personnalisées
  - Gestion des requêtes inattendues
  - Adaptation au feedback utilisateur
- Premier point de contact avant escalade vers un agent humain

### 5. Servir d'interface aux systèmes complexes
- Les utilisateurs non-techniques peuvent interagir en langage naturel avec des systèmes complexes (BDD, outils dev)
- Exemples : gestionnaire d'investissement qui interroge une BDD sans SQL, outil generative fill de Photoshop
- Des startups remplacent des workflows complexes (multi-écrans) par une interface conversationnelle
- ⚠️ Nécessite des **guardrails et mesures de sécurité** (cf. Chapter 9)

### 6. Automatiser les tâches administratives manuelles
- Tâches typiques : traitement de documents à layouts complexes (factures, bons de commande)
- Ces tâches restaient manuelles car chaque document a un layout unique
- Les LLMs peuvent combler les lacunes des automatisations existantes et flagger les cas limites pour revue humaine

### 7. Scaler et démocratiser la génération de contenu
- Productivité : un article de blog passe de jours à heures de production
- Workflow : se concentrer sur le plan et la structure → GenAI remplit les détails
- La GenAI excelle pour les tâches cognitives de bas niveau (résumés, reformulations)
- Ce qui rend un texte intéressant reste le **style et le flow**, pas juste le contenu

## Architecture d'un service GenAI

```
[Client] → [FastAPI Web Server] → [Generative Model]
                ↕                        ↕
         [Contrôle d'accès]     [Sources de données]
         [Routeurs]             (BDD, APIs externes)
```

Le serveur web joue le rôle d'**intermédiaire** :
1. **Enrichit** les prompts utilisateur avec des données contextuelles (BDD, APIs)
2. **Contrôle** les outputs générés (sanity check)
3. **Route** les réponses finales vers l'utilisateur

> 💡 On peut aussi configurer un LLM pour construire des instructions qu'un autre composant exécute (ex : requêtes BDD, appels API).

## Pourquoi FastAPI pour les services GenAI ?

**Besoin** : les services GenAI nécessitent un framework web performant et event-driven.

### Comparaison des frameworks Python

| Framework | Type | Forces | Limites |
|-----------|------|--------|---------|
| **Django** | Full-stack, batteries included | Mature, large communauté, MVC | Support async immature, overhead pour APIs légères |
| **Flask** | Micro-framework | Léger, extensible, leader en downloads | Peu de features par défaut (pas de validation de schéma out-of-the-box) |
| **FastAPI** | Full-stack moderne | Rapide, performant, DX excellente | Plus récent (communauté plus petite mais en forte croissance) |

### Avantages clés de FastAPI

- **Performance** comparable à Gin (Go) ou Express (Node.js)
- **Accès direct à l'écosystème deep learning Python** (impossible avec des frameworks non-Python)
- Features out-of-the-box : validation de données, type safety, documentation auto, serveur web intégré
- Support du **model serving via lifecycle events**
- ~80k GitHub Stars, framework Python à la croissance la plus rapide

> 💡 FastAPI combine la performance d'un framework moderne avec la richesse de l'écosystème Python ML/AI — c'est ce qui le distingue pour les services GenAI.

---

## Freins à l'adoption des services GenAI

### Problèmes liés aux outputs

- **Inexactitude** : les modèles peuvent halluciner (produire des faits inventés mais plausibles)
- **Qualité/originalité limitée** : les modèles recombinent des informations existantes, suivent des patterns génériques et répétitifs
- **Manque de consistance** : difficile de garantir des réponses cohérentes et appropriées pour du customer-facing

> ⚠️ **Hallucinations** = le modèle génère des informations incorrectes présentées comme factuelles. Bloquant pour les cas sensibles (médical, juridique, examens).

### Problèmes d'intégration et sécurité

- **Data privacy** : réticence à connecter les modèles aux systèmes sensibles (BDD internes, systèmes de paiement)
- **Cybersécurité** : risques d'abus/détournement des modèles en production
- **Compatibilité** : intégration complexe avec les systèmes existants (BDD, interfaces web, APIs externes)
- **Expertise requise** : besoin de compétences techniques spécifiques

### Solutions évoquées

| Problème | Solution |
|----------|----------|
| Data privacy, sécurité | Bonnes pratiques de software engineering (couvert dans le livre) |
| Qualité/pertinence des outputs | Optimisation des inputs (prompts) |
| Consistance/cohérence | Fine-tuning des modèles sur des cas d'usage spécifiques |