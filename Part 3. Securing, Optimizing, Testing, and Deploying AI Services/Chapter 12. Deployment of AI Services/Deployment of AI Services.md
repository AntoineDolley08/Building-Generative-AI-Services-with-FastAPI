# Chapter 12 — Deployment of AI Services

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Options de déploiement](#1-options-de-déploiement)
2. [Containerization avec Docker](#2-containerization-avec-docker)
3. [Docker Storage](#3-docker-storage)
4. [Docker Networking](#4-docker-networking)
5. [Docker Compose](#5-docker-compose)
6. [Optimisation des images Docker](#6-optimisation-des-images-docker)

---

## 1. Options de déploiement

### 4 stratégies

| Stratégie | Principe | Avantages | Inconvénients |
|-----------|----------|-----------|---------------|
| **VM** | Serveur virtuel complet (guest OS + hypervisor) | Accès direct GPU, isolation totale, SSH debug, coûts bas si on-prem | Scale difficile, maintenance lourde (patches, updates), 24/7 = coûts constants |
| **Serverless** | Cloud functions event-driven (AWS Lambda, Azure Functions) | Pay-per-use, scale auto, pas de gestion serveur | Timeout court (~10min), ressources limitées, pas pour model serving |
| **Managed App Platform** | PaaS (Azure App Service, Heroku, Fly.io, Railway) | Deploy rapide, networking/SSL/monitoring intégrés | Plus cher, pas de GPU, vendor lock-in possible |

Plateformes AI dédiées (GPU/model serving) : Azure ML Studio, Google Vertex AI, AWS Bedrock/SageMaker, IBM Watson Studio. Tiers : Hugging Face Inference Endpoints, Weights & Biases, Replicate.
| **Containers** | Docker — app + dépendances packagées dans une unité isolée | Portable, léger, rapide, fiable, scale horizontal | Courbe d'apprentissage Docker, config réseau/permissions |

### Quand utiliser quoi

```
Prototype rapide         → Managed App Platform (Heroku, Railway, Fly.io)
Peu d'users, budget serré → VM (OVH VPS, on-prem)
Event-driven, batch      → Serverless functions
Production scalable      → Containers (Docker) sur VM ou cloud
Model serving GPU        → VM dédiée ou AI platform (Azure ML, Vertex AI, SageMaker)
```

### Serverless : FastAPI sur Azure Functions

```
project/
├── host.json           # config Azure Functions
├── app.py              # FastAPI app classique
├── function.py         # wrapper Azure
└── requirements.txt
```

```python
# app.py — FastAPI standard
app = FastAPI()

@app.post("/generate/text")
async def serve_text(prompt): ...

# function.py — wrapper Azure
import azure.functions as func
from app import app as fastapi_app

app = func.AsgiFunctionApp(
    app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS
)
```

```bash
func start                                          # dev local
func azure functionapp publish <FunctionAppName>    # deploy
```

> ⚠️ Cloud functions = pas adapté pour servir des modèles GenAI (timeout). Utiliser un model provider API à la place.

---

## 2. Containerization avec Docker

### VM vs Container

| | VM | Container |
|---|---|---|
| Isolation | Guest OS complet + hypervisor | Partage le kernel host OS |
| Taille | Plusieurs GB | Quelques MB à ~1 GB |
| Boot | Minutes | Secondes |
| Principe | Abstrait le **hardware** | Abstrait l'**OS kernel** |

### Architecture Docker

```
[Docker Client (CLI/Desktop)]
        ↓ REST API
[Docker Daemon (dockerd)]
    ├── Gère les containers (lifecycle)
    ├── Build/pull/push images
    └── Interagit avec registries (Docker Hub, ACR, ECR)
```

### Image = recette, Container = instance en mémoire

> Une image est **immutable** : on ne peut qu'ajouter, pas modifier. Pour changer → recréer.

### Dockerfile basique

```dockerfile
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

| Instruction | Rôle |
|------------|------|
| `FROM` | Image de base (slim = bon compromis taille/compatibilité) |
| `WORKDIR` | Répertoire de travail dans le container |
| `COPY` | Copie des fichiers host → container |
| `RUN` | Exécute une commande (installe dépendances) |
| `EXPOSE` | Documente le port (ne le publie pas automatiquement) |
| `CMD` | Commande au lancement du container |

### Build et run

```bash
docker build -t genai-service .
docker run -p 127.0.0.1:8000:8000 genai-service
```

### Container Registries

```bash
# Pull une image publique
docker image pull python:3.12-slim

# Tag + push vers un registry
docker build -t genai-service:latest .
docker image tag genai-service:latest docker.io/myrepo/genai-service:latest
docker login
docker image push docker.io/myrepo/genai-service:latest
```

Registries : Docker Hub (public), Azure ACR, AWS ECR, Google Artifact Registry (privés).

> ⚠️ Attention à ne pas écraser un tag existant lors d'un push (ex: `latest` remplacé par une autre image).

### Filesystem en couches (Unionfs)

Chaque instruction Dockerfile crée un **layer** empilé. Les layers sont cachés et réutilisés entre les builds si rien n'a changé.

```
┌─────────────────────────┐
│ Container layer (éphém.) │ ← writable, perdu à l'arrêt
├─────────────────────────┤
│ COPY . .                │ ← code app (volatile)
├─────────────────────────┤
│ RUN pip install         │ ← dépendances (stable)
├─────────────────────────┤
│ COPY requirements.txt   │
├─────────────────────────┤
│ python:3.12-slim        │ ← base image
├─────────────────────────┤
│ Root filesystem         │
└─────────────────────────┘
```

> 10 containers depuis une image de 1 GB ≠ 10 GB de disque. En Linux, les containers sont des processus créés par fork (copie de l'espace mémoire parent). Sans Unionfs, il faudrait copier 1 GB par container. Unionfs partage les layers physiques → chaque container ne consomme que son layer writable. Démarrage rapide + mémoire optimisée.

---

## 3. Docker Storage

### 3 types de montages

| Type | Persistance | Géré par | Use case |
|------|------------|----------|----------|
| **Volumes** | ✅ Persiste après arrêt | Docker | Partager données entre containers, BDD |
| **Bind mounts** | ✅ Persiste (fichiers host) | Host OS | Dev local — code source live-reload |
| **tmpfs** | ❌ RAM uniquement | Host RAM | Cache temporaire, données sensibles, performance |

> Sans volume/mount → les données écrites dans le container sont **perdues** à l'arrêt.

### Volumes

```bash
docker volume create -n data
docker run -v data:/etc/data postgres
```

> ⚠️ Redémarrer un container BDD avec de nouvelles variables d'environnement ne suffit pas toujours à les appliquer. Certains systèmes (PostgreSQL) nécessitent de re-créer le volume pour réinitialiser les credentials admin.

### Bind mounts (dev local)

```bash
docker run -v ./src:/app genai-service
# ⚠️ Modifications dans le container = modifications sur l'host !
```

> `COPY` dans Dockerfile = copie séparée au build. Bind mount = accès direct aux fichiers host.

### tmpfs (mémoire)

```bash
docker run --tmpfs /cache genai-service
```

Use cases : caches modèle, résultats intermédiaires, logs de session, données sensibles temporaires.

> tmpfs ne peut pas être partagé entre containers. Linux uniquement. Permissions reset au restart.

### Permissions filesystem

#### Le problème

Docker tourne en **root** par défaut → les fichiers créés dans les bind mounts appartiennent à root sur l'host → problèmes de permissions. Risque sécurité : si un acteur malveillant accède au container root → accès host en root. Si une image compromise est exécutée → code malveillant avec privilèges root sur l'host.

#### Solution : user non-root

```dockerfile
ARG USERNAME=fastapi
ARG USER_UID=1001
ARG USER_GID=1002

RUN groupadd --gid $USER_GID $USERNAME \
    && adduser \
    --disabled-password \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid $USER_UID \
    --gid $USER_GID \
    $USERNAME

USER $USERNAME    # switcher APRÈS toutes les installations
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> Toujours switcher vers le user non-root **en dernier** dans le Dockerfile, après les `RUN apt-get install` et autres commandes nécessitant des privilèges.

#### Permissions Linux (chmod / chown)

```bash
# Changer le propriétaire
sudo chown -R username:groupname mydir

# Changer les permissions
sudo chmod -R 755 myscripts
```

| Valeur | Symbole | Permissions |
|--------|---------|-------------|
| 7 | `rwx` | Read, Write, Execute |
| 6 | `rw-` | Read, Write |
| 5 | `r-x` | Read, Execute |
| 4 | `r--` | Read only |
| 3 | `-wx` | Write, Execute |
| 2 | `-w-` | Write only |
| 1 | `--x` | Execute only |
| 0 | `---` | Aucune |

`chmod 755` = owner rwx, group r-x, others r-x.

---

## 4. Docker Networking

### Drivers réseau

| Driver | Principe | Use case |
|--------|----------|----------|
| **Bridge** (défaut) | Réseau isolé entre containers sur le même host | Communication inter-containers standard |
| **Host** | Container partage le réseau de l'host (pas d'isolation) | Performance, éviter le port mapping |
| **None** | Aucun réseau | Isolation sécurité, debug, containers éphémères |
| Overlay | Connecte containers sur plusieurs hosts | Docker Swarm, clusters |
| Macvlan | Adresses MAC comme des devices physiques | Legacy, monitoring réseau |
| IPVlan | Contrôle total IPv4/IPv6 | Config réseau avancée |

> Les plus utilisés : **bridge**, **host**, **none**. Les autres = cas avancés.

> ⚠️ Docker interagit avec le kernel pour configurer les règles firewall (iptables/ip6tables sur Linux). Il peut **forcer l'ouverture** de ports fermés par le firewall host. Solution : toujours utiliser `127.0.0.1:` dans le port mapping pour limiter l'accès au localhost.

### Bridge : réseau par défaut

```bash
# Voir les réseaux
docker network ls

# Les containers sur le même bridge peuvent communiquer
# MAIS le bridge par défaut = pas de résolution DNS par nom
```

### User-defined bridge (recommandé)

```bash
# Créer un réseau isolé
docker network create genai-net

# Attacher les containers
docker run --network genai-net --name genai-service genai-service
docker run --network genai-net --name db postgres
```

Avantages vs bridge par défaut :
- **DNS embarqué** : les containers se trouvent par nom (`db`, `genai-service`)
- **Meilleure isolation** : seuls les containers attachés au réseau communiquent
- **Sécurité** : pas de communication accidentelle entre services non liés

> Le DNS embarqué n'est **pas visible** depuis l'host. Pour accéder au container depuis l'host → publier les ports.

> Limite Linux : max 1000 containers par bridge network.

### Publier des ports

```bash
docker run -p 127.0.0.1:8000:8000 myimage
# Syntaxe : <host_port>:<container_port>
```

> ⚠️ Docker peut **forcer l'ouverture** de ports fermés par le firewall host. Toujours utiliser `127.0.0.1:` pour limiter l'accès au localhost.

### Host network

```bash
docker run --net=host genai-service
# → Le container utilise directement le réseau host
# → Plus de port mapping nécessaire
# → Linux uniquement
```

### GPU dans Docker

```bash
# Tester le support GPU
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark

# Lancer un container avec GPU
docker run --gpus=all genai-service
```

```python
# Transférer un modèle HuggingFace sur GPU
from transformers import pipeline

pipe = pipeline("text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_map="cuda")
```

---

## 5. Docker Compose

### Principe

Un fichier `compose.yaml` pour gérer tous les containers, réseaux, volumes et secrets.

### Exemple complet

```yaml
# compose.yaml
services:
  server:
    build: .
    ports:
      - "8000:8000"
    environment:
      ALLOWED_CORS_ORIGINS: $ALLOWED_CORS_ORIGINS
    secrets:
      - openai_api_token
    volumes:
      - ./src/app:/code/app
    networks:
      - genai-net

  db:
    image: postgres:12.2-alpine
    ports:
      - "5433:5432"
    volumes:
      - db-data:/etc/data
    networks:
      - genai-net

volumes:
  db-data:
    name: "my-app-data"

networks:
  genai-net:
    name: "genai-net"
    driver: bridge

secrets:
  openai_api_token:
    environment: OPENAI_API_KEY
```

### Commandes essentielles

```bash
docker compose up       # démarrer tous les services
docker compose down     # arrêter et supprimer (garde volumes/networks)
docker compose logs     # voir les logs
docker compose ps       # statut des services
```

### Watch (hot-reload en dev)

```yaml
services:
  server:
    develop:
      watch:
        - action: sync
          path: ./src
          target: /code
```

```bash
docker compose watch    # sync auto des fichiers modifiés
```

> Plus granulaire que les bind mounts — permet d'ignorer des fichiers/dossiers spécifiques.

### Override pour dev local

```yaml
# compose.yml — base (production)
services:
  server:
    ports: ["8000:8000"]
    command: uvicorn main:app

# compose.override.yml — dev local (auto-mergé)
services:
  server:
    environment:
      - LLM_API_KEY=$LLM_API_KEY
      - DATABASE_URL=$DATABASE_URL
    volumes:
      - ./code:/code
    command: uvicorn main:app --reload
  database:
    image: postgres:latest
    environment:
      - POSTGRES_DB=genaidb
      - POSTGRES_USER=genaiuser
      - POSTGRES_PASSWORD=secretPassword!
    volumes:
      - db_data:/var/lib/postgresql/data
```

`docker compose up` merge automatiquement les deux fichiers.

> Si des volumes/networks sont gérés manuellement (hors Compose), les tagger avec `external: true` dans le compose file pour que Compose ne les gère pas.

### GPU dans Compose

```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 6. Optimisation des images Docker

### Impact des optimisations

| Étape | Build time | Image size |
|-------|-----------|-----------|
| Initial | 353s | 1.42 GB |
| Base image minimale | 38s | 1.38 GB |
| Caching | 24s | 1.38 GB |
| Layer ordering | 18s | 1.38 GB |
| **Multi-stage builds** | **10s** | **34 MB** |

### 1. Base images minimales

```bash
docker image ls
# alpine    3.20     12.1 MB
# python    3.12-alpine    71.4 MB
# python    3.12-slim      186 MB
```

| Image | Taille | Quand |
|-------|--------|-------|
| `python:3.12-slim` | ~186 MB | Build time prioritaire, bonne compatibilité packages |
| `python:3.12-alpine` | ~71 MB | Taille image prioritaire, config supplémentaire requise |

### 2. Éviter les runtimes GPU

GPU inference : ~3 GB NVIDIA + ~1.6 GB PyTorch. Si CPU suffit → **ONNX Runtime** réduit l'image de 5-10 GB à ~0.5 GB.

```bash
pip install transformers[onnx]
python -m transformers.onnx --model=distilbert/distilbert-base-uncased onnx/
```

```python
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
session = InferenceSession("onnx/model.onnx")
inputs = tokenizer("Hello world!", return_tensors="np")  # numpy, pas torch
output = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

### 3. Externaliser les données

Ne pas copier les modèles dans l'image. Utiliser des volumes en dev, du stockage externe en production. En Kubernetes, utiliser des persistent volumes pour le stockage des modèles.

> Si le téléchargement au startup est trop long → les health checks peuvent tuer le container. Configurer des probes plus longues, ou en dernier recours, intégrer le modèle dans l'image.

### 4. Layer ordering et caching

#### ❌ Mauvais ordre (cache invalidé à chaque changement de code)

```dockerfile
COPY . .                          # ← code change souvent
RUN pip install requirements.txt  # ← recalculé à chaque fois !
```

#### ✅ Bon ordre (dépendances cachées, code copié en dernier)

```dockerfile
COPY requirements.txt .
RUN pip install requirements.txt  # ← caché tant que requirements.txt ne change pas
COPY . .                          # ← seul ce layer est recalculé
```

#### Instructions qui créent des layers

| Crée un layer | Ne crée pas de layer |
|--------------|---------------------|
| `COPY`, `ADD`, `RUN`, `ENV` | `WORKDIR`, `ENTRYPOINT`, `LABEL`, `CMD` |

#### Combiner les RUN (cache busting)

```dockerfile
# ❌ 2 layers
RUN apt-get update
RUN apt-get install -y curl

# ✅ 1 layer
RUN apt-get update && apt-get install -y curl
```

#### .dockerignore

```
**/.DS_Store
**/__pycache__
**/.mypy_cache
**/.venv
**/.env
**/.git
```

#### Cache et bind mounts pour les builds

```dockerfile
# Cache le téléchargement du modèle entre les builds
RUN --mount=type=cache,target=/root/.cache/huggingface && \
    pip install transformers && \
    python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('bert-base-uncased')"
```

> `--mount=type=bind` : inclut temporairement des fichiers dans le build context pour un seul `RUN` (pas persisté comme layer). `--mount=type=cache` : cache persistant entre les builds.

#### External cache (CI/CD)

Dans les pipelines CI/CD où les builders sont éphémères, un cache externe accélère drastiquement les builds :

```bash
docker buildx build --cache-from type=registry,ref=user/app:buildcache .
```

> Utiliser `--no-cache` avec `docker build` pour forcer des downloads frais (base images, dépendances) à chaque build si nécessaire.

### 5. Multi-stage builds

Séparer le Dockerfile en stages distincts → le final image ne contient que le nécessaire.

```dockerfile
# Stage 1: Base — télécharge modèle + installe dépendances
FROM python:3.11.0-slim as base
RUN python -m venv /opt/venv
RUN pip install transformers && \
    python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('bert-base-uncased')"
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Stage 2: Production — copie seulement ce qui est nécessaire
FROM base as production
COPY --from=base /opt/venv /opt/venv
COPY --from=base /root/.cache/huggingface /root/.cache/huggingface
WORKDIR /code
COPY . .
EXPOSE 8000
ENV BUILD_ENV=PROD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development — ajoute outils de dev
FROM production as development
COPY --from=production /opt/venv /opt/venv
COPY ./requirements_dev.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements_dev.txt
ENV BUILD_ENV=DEV
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

```bash
docker build --target development -t genai-service:dev .
docker build --target production -t genai-service:prod .
```

### docker init (starter)

```bash
docker init
# → Génère Dockerfile, compose.yaml, .dockerignore avec les best practices
```

---

## Points clés à retenir

1. **Container = méthode de déploiement la plus fiable**. Portable, léger, reproductible. VM si GPU nécessaire et pas de container orchestration
2. **User-defined bridge network** > bridge par défaut. DNS par nom, meilleure isolation, sécurité
3. **Ne jamais tourner en root** dans les containers de production. Créer un user non-root en fin de Dockerfile
4. **Layer ordering** : instructions stables en haut (dépendances), volatiles en bas (code). Sinon le cache est invalidé à chaque changement
5. **Multi-stage builds** : de 1.42 GB / 353s → **34 MB / 10s**. Séparer base, production, development
6. **Volumes pour la persistance**, bind mounts pour le dev local, tmpfs pour le cache temporaire
7. **Docker Compose** simplifie le multi-container. `compose.override.yml` pour les configs dev locales
8. **ONNX Runtime** : si pas besoin de GPU, réduit l'image de 5-10 GB → ~0.5 GB
9. **`.dockerignore`** : toujours exclure `.venv`, `__pycache__`, `.git`, `.env` du build context
10. **`docker compose watch`** > bind mounts pour le hot-reload en dev (plus granulaire)
11. **Publier les ports** avec `127.0.0.1:` pour éviter que Docker force l'ouverture sur le réseau externe
12. **`docker init`** pour démarrer avec les best practices si on part de zéro