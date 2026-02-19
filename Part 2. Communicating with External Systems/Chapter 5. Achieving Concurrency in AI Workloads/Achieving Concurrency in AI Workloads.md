# Chapter 5 (partie 1) — Concurrency in AI Workloads : Fondamentaux

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Le problème : servir plusieurs utilisateurs simultanément

Les workloads AI sont coûteux en compute → ils **bloquent** le serveur et empêchent de traiter d'autres requêtes.

### Deux types d'opérations bloquantes

| Type | Cause | Exemples |
|------|-------|----------|
| **I/O-bound** | Attente d'une opération d'entrée/sortie | Requêtes réseau, appels API, lecture/écriture fichiers, requêtes BDD, attente input utilisateur |
| **Compute-bound** | Calcul intensif CPU/GPU | Inférence AI, entraînement de modèles, rendering 3D, simulations |

### 3 stratégies de résolution

| Stratégie | Cible | Approche |
|-----------|-------|----------|
| **System optimization** | I/O-bound | Async programming, multithreading |
| **Model optimization** | Memory/compute-bound | Optimisations mémoire, serveurs d'inférence spécialisés |
| **Queuing system** | Tâches longues | Background tasks, files d'attente |

---

## Concurrency vs Parallelism

### Concurrency (1 core)

Gérer **plusieurs tâches en alternant** entre elles → donne l'**illusion** d'exécution simultanée.

```
Tâche A: ████░░░░████░░░░████
Tâche B:     ████░░░░████░░░░
             ↑ switch  ↑ switch
```

Implémentée via :
- **Async IO** : un seul thread, event loop qui switch entre tâches pendant les temps d'attente I/O
- **Multithreading** : plusieurs threads dans un même process, orchestrés par l'OS

### Parallelism (multi-core)

Exécuter **plusieurs tâches réellement en même temps** sur des cores séparés.

```
Core 1: ████████████████████  (Tâche A)
Core 2: ████████████████████  (Tâche B)
```

Implémenté via :
- **Multiprocessing** : chaque process a son propre core, sa mémoire, ses ressources isolées

### Analogie restaurant

- **Concurrent** : un seul employé prend les commandes ET cuisine, en alternant entre les deux
- **Parallèle** : plusieurs employés, certains prennent les commandes, d'autres cuisinent en même temps

---

## Le GIL (Global Interpreter Lock)

Le GIL empêche **plusieurs threads d'exécuter du code Python simultanément** dans un même process.

### Conséquences

- **Un seul thread actif** à la fois dans un process Python
- Le multithreading Python n'est **pas parallèle** (juste concurrent)
- Le CPU switch entre threads pendant les attentes I/O → utile pour l'I/O-bound
- **Inutile pour le compute-bound** car le process est occupé à calculer, pas à attendre

> Le GIL sera rendu optionnel dans de futures versions de Python (PEP 703 — free-threaded Python).

---

## 3 modèles d'exécution Python

### 1. Synchrone (pas de concurrence)

```
[Tâche 1 ██████████] → [Tâche 2 ██████████] → [Tâche 3 ██████████]
```

- 1 core, 1 thread, séquentiel
- Simple mais lent sous charge

### 2. Concurrent non-parallèle (async ou multithreading)

```
[Tâche 1 ███░░░███] 
[Tâche 2    ███░░░███]    ← même core, switch pendant les I/O
[Tâche 3       ███░░░███]
```

- 1 core, 1+ threads ou event loop
- Optimise le temps d'attente I/O
- Le GIL empêche le vrai parallélisme

### 3. Concurrent et parallèle (multiprocessing)

```
Core 1: [Tâche 1 ██████████]
Core 2: [Tâche 2 ██████████]    ← vrais process séparés
Core 3: [Tâche 3 ██████████]
```

- Multi-cores, multi-process
- Chaque process a sa propre mémoire (pas de GIL partagé)
- Vrai parallélisme

---

## Multithreading vs Multiprocessing

| | Multithreading | Multiprocessing |
|---|---------------|----------------|
| **Mémoire** | Partagée entre threads | Isolée par process |
| **Communication** | Simple (mémoire partagée) | Complexe (IPC nécessaire) |
| **Crash** | Un thread peut affecter les autres | Un process crashé n'affecte pas les autres |
| **Bon pour** | I/O-bound | Compute-bound |
| **Risques** | Deadlocks, race conditions, starvation | Coût mémoire élevé, sync complexe |
| **Coût création** | Modéré | Élevé |

---

## Cas particulier : APIs tierces (OpenAI, Anthropic...)

Quand on utilise une API tierce pour l'inférence :
- L'inférence compute-bound est **déléguée au provider**
- Côté FastAPI, ça devient de l'**I/O-bound** (requêtes réseau)
- → On peut utiliser l'**async programming** pour gérer la concurrence

```
[FastAPI] --HTTP async--> [OpenAI API]  ← compute-bound géré par OpenAI
    ↑                                     
    └── I/O-bound de notre côté → async suffit
```

### Pour les gros modèles self-hosted

Ne pas essayer de scale avec du multiprocessing FastAPI (chaque worker recharge le modèle entier en mémoire).

→ Utiliser des **serveurs d'inférence spécialisés** : vLLM, Ray Serve, NVIDIA Triton.

---

## Tableau récap des stratégies

| Stratégie | Bon pour | Limites |
|-----------|----------|---------|
| **Synchrone** | Apps mono-utilisateur, prototypage | Ne scale pas |
| **Async IO** | I/O-bound (réseau, BDD, fichiers) | Requiert des libs async, erreurs subtiles |
| **Multithreading** | I/O-bound (plus simple qu'async) | GIL, deadlocks, race conditions |
| **Multiprocessing** | Compute-bound, distribution de charge | Mémoire isolée, coût élevé, sync complexe |

---

## Ce qui vient dans la suite du chapitre

1. **Web scraper async** pour enrichir les prompts avec du contenu web
2. **Module RAG** avec vector DB (Qdrant) pour uploader et interroger des documents
3. **Batch image generation** en background tasks
4. **Serveurs d'inférence spécialisés** (vLLM, etc.) pour optimiser les modèles lourds