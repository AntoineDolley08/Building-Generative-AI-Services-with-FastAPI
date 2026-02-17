# Chapter 2 (suite) — Architecture, comparaison frameworks et limitations

## Sécurité & authentification

- FastAPI ne force pas d'implémentation spécifique → set de composants de sécurité à assembler soi-même
- Alternative : plug-in **FastAPI Users** pour auth clé-en-main
- Support SSO avec providers tiers pour environnements enterprise
- Détail au **Chapter 8**

## Au-delà du REST

| Protocole | Usage GenAI | Support FastAPI |
|-----------|-------------|-----------------|
| **REST** (GET/POST/PUT/PATCH/DELETE) | CRUD classique | Natif |
| **WebSocket** | Streaming de tokens en temps réel | Natif |
| **SSE** (Server-Sent Events) | Streaming unidirectionnel serveur→client | Natif |
| **GraphQL** | Requêtes dynamiques, éviter l'over-fetching | Via `strawberry` |

---

## Structures de projet FastAPI

### Approche recommandée : progression flat → nested → modular

> 💡 Si tu ne peux pas justifier l'organisation de tes fichiers à un autre dev, c'est le moment de restructurer.

### 1. Flat — pour démarrer / microservices

```
flat-project/
├── app/
│   ├── main.py
│   ├── routers.py
│   ├── services.py
│   ├── models.py
│   └── database.py
├── requirements.txt
└── .env
```

✅ Simple, focus sur le dev | ❌ Ne scale pas avec la complexité

### 2. Nested — projets moyens (recommandé par la doc officielle)

```
nested-project/
├── app/
│   ├── main.py
│   ├── dependencies.py
│   ├── services/
│   │   ├── users.py
│   │   └── profiles.py
│   ├── models/
│   │   ├── users.py
│   │   └── profiles.py
│   └── routers/
│       ├── users.py
│       └── profiles.py
```

Groupement par **type logique** (tous les models ensemble, tous les routers ensemble).

✅ Organisé | ❌ Couplage ambigu → risque de **shotgun updates** (modifier un truc = cascade de modifs partout)

### 3. Modular — gros projets (inspiré Netflix Dispatch)

```
modular-project/
├── app/
│   ├── modules/
│   │   ├── auth/
│   │   │   ├── routers.py
│   │   │   ├── models.py
│   │   │   ├── dependencies.py
│   │   │   ├── guards.py
│   │   │   └── services.py
│   │   ├── users/
│   │   │   ├── router.py
│   │   │   ├── models.py
│   │   │   ├── services.py
│   │   │   ├── mappers.py
│   │   │   └── pipes.py
│   ├── providers/
│   │   ├── email.py
│   │   └── stripe.py
│   ├── settings.py
│   ├── middlewares.py
│   ├── exceptions.py
│   └── main.py
```

Groupement par **domaine/feature** (tout ce qui concerne `auth` ensemble).

✅ Scalable, maintenable, ajout/suppression facile | ❌ Overkill pour un petit projet

---

## Onion / Layered Design Pattern

Pattern d'architecture en couches concentriques avec **dépendances orientées vers l'intérieur**.

### Les couches (de l'extérieur vers l'intérieur)

```
[Middleware] → [Routers] → [Controllers] → [Services/Providers] → [Repositories] → [Schemas/Models]
   (outer)                                                                              (inner/core)
```

| Couche | Responsabilité |
|--------|---------------|
| **Routers** (APIRouter) | Grouper les controllers, appliquer une logique commune |
| **Controllers** | Gérer requêtes/réponses, orchestrer via injection de dépendances |
| **Services** | Business logic interne, orchestration d'opérations |
| **Providers** | Interface avec systèmes externes (email, paiement, APIs tierces) |
| **Repositories** | Accès données (ORM/SQL), opérations CRUD |
| **Schemas/Models** | Type-safety, validation, structure des données |

### Composants transversaux (cross-layer)

| Composant | Rôle |
|-----------|------|
| **Middleware** | Intercepte request/response avant/après les controllers |
| **Dependencies** | Fonctions réutilisables injectables (cachées par requête) |
| **Pipes** | Transformateurs de données (agrégation, parsing, nettoyage) |
| **Mappers** | Conversion entre schémas (ex : `UserRequest` → `UserInDB`) |
| **Guards** | Protection des controllers (auth/authz) |
| **Exception filters** | Gestion uniforme des erreurs |

### Principe clé : Dependency Inversion

Les modules de haut niveau ne dépendent pas de l'implémentation des modules de bas niveau → ils déclarent ce dont ils ont besoin via le système `Depends()` de FastAPI.

---

## Comparaison des frameworks Python

| | Django | Flask | FastAPI |
|---|--------|-------|---------|
| **Type** | Opinionated, batteries-included | Non-opinionated, micro | Non-opinionated, full-featured |
| **Interface** | WSGI (async depuis v4.2) | WSGI | ASGI |
| **ORM** | Intégré, excellent | À installer | À installer |
| **Validation** | Via forms/serializers | À installer | Pydantic intégré |
| **Auto-doc** | Non | Non | Swagger/OpenAPI intégré |
| **DI system** | Non | Non | `Depends()` intégré |
| **WebSocket** | Channels (extension) | Extension | Natif |
| **Idéal pour** | PWA monolithes | APIs simples | APIs + AI backends |

### ASGI vs WSGI rappel

- **WSGI** (Flask) : synchrone, chaque requête bloque un worker
- **ASGI** (FastAPI) : event loop async + thread pool pour sync → concurrent nativement
- ASGI offre aussi la rétrocompatibilité WSGI

> **Mention** : Quart = contender async inspiré de Flask, mais communauté encore petite.

---

## Limitations de FastAPI pour l'AI

### ⚠️ Points critiques à connaître

| Limitation | Impact | Solution |
|-----------|--------|----------|
| **Pas de partage mémoire modèle** entre workers | Chaque worker charge le modèle entier → bottleneck mémoire | Architecture séparée (model server externe) |
| **Nombre limité de threads** (~40 par défaut via AnyIO) | Scalabilité limitée pour workloads I/O + CPU/GPU | Multiprocessing, process pool |
| **GIL (Global Interpreter Lock)** | L'inférence AI CPU-intensive bloque les autres threads | Multiprocessing (Chapter 5), PEP 703 (GIL optionnel à venir) |
| **Pas de micro-batch inference** | Impossible de batcher les requêtes de prédiction | Serveur modèle dédié |
| **Pas de split CPU/GPU** | CPU bloqué même quand l'inférence tourne sur GPU | Frameworks spécialisés (BentoML) |
| **Conflits de dépendances** | Couplage modèle runtime + libs natives + hardware | Containerisation soignée |

### Architecture recommandée pour workloads lourds

```
[Client] → [FastAPI] ←→ [BentoML (model serving)]
              ↓
    Sécurité, caching,
    business logic
```

**BentoML** : construit sur Starlette (comme FastAPI), conçu pour le ML. Gère le scaling web séparément de l'inférence, avec Runners, gestion de dépendances, et auto-génération de Dockerfiles (CUDA inclus).

---

## Tooling Python recommandé

### Gestion d'environnement
- Simple : `requirements.txt` + `pip`
- Intermédiaire : `uv` ou `Conda`
- Complexe : `Poetry`

### Stack d'outils

| Catégorie | Outils | Rôle |
|-----------|--------|------|
| **Linter** | Flake8, Autoflake | Erreurs de style, imports inutilisés |
| **Formatter** | Black, isort, **Ruff** (remplace tout) | Formatage, tri des imports |
| **Logger** | Loguru | Remplace le logger built-in |
| **Scanner sécurité** | Bandit (code), Safety (dépendances) | Secrets hardcodés, vulnérabilités |
| **Type checker** | Mypy, Pylance (VS Code) | Bugs de typage statique |

> 💡 **Ruff** (écrit en Rust) peut remplacer isort + black + flake8 + bandit → un seul outil ultra-rapide.

### Bonnes pratiques
- Pre-commit hooks pour lint/format/check avant chaque commit
- `.gitignore` pour exclure les fichiers sensibles
- Script CI/CD qui lance les checks automatiquement