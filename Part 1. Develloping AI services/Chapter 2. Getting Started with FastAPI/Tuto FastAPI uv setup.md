# 🚀 Setting Up a FastAPI GenAI Service with `uv`

> Tutorial basé sur le livre *Building Generative AI Services with FastAPI* — adapté pour utiliser [`uv`](https://docs.astral.sh/uv/) au lieu de pip/conda/venv.

## Prérequis

- Python 3.11+
- `uv` installé ([guide d'installation](https://docs.astral.sh/uv/getting-started/installation/))

```bash
# Installer uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou via pip
pip install uv
```

---

## 1. Initialiser le projet

```bash
# Créer le projet avec uv
uv init genai-fastapi-service
cd genai-fastapi-service

# Fixer la version Python
uv python pin 3.12
```

Cela génère la structure suivante :

```
genai-fastapi-service/
├── .python-version    # version Python fixée (3.11)
├── pyproject.toml     # config du projet + dépendances
├── README.md
└── main.py
```

---

## 2. Installer les dépendances

### Core

```bash
uv add "fastapi[standard]" uvicorn openai
```

`fastapi[standard]` installe automatiquement `starlette`, `pydantic`, `uvicorn`, et d'autres dépendances utiles.

### Dev tooling

```bash
uv add --dev ruff mypy loguru bandit pytest
```

> `ruff` remplace à lui seul isort + black + flake8 + autoflake.

### Vérifier les dépendances installées

```bash
uv pip list
```

---

## 3. Configurer le tooling dans `pyproject.toml`

Ajouter la configuration suivante dans le `pyproject.toml` généré par `uv` :

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "SIM", # flake8-simplify
]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 4. Créer le serveur FastAPI minimal

Remplacer le contenu de `main.py` :

```python
# main.py
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# ⚠️ En production, utiliser des variables d'environnement
openai_client = OpenAI(api_key="your_api_key")


@app.get("/")
def root_controller():
    return {"status": "healthy"}


@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    statement = response.choices[0].message.content
    return {"statement": statement}
```

---

## 5. Lancer le serveur

```bash
# Mode développement (hot-reload activé)
uv run fastapi dev

# Le serveur démarre sur http://127.0.0.1:8000
# Swagger UI disponible sur http://127.0.0.1:8000/docs
```

### Tester les endpoints

```bash
# Health check
curl http://127.0.0.1:8000/

# Chat (prompt par défaut)
curl http://127.0.0.1:8000/chat

# Chat avec prompt custom
curl "http://127.0.0.1:8000/chat?prompt=Explain%20ASGI%20in%20one%20sentence"
```

---

## 6. Commandes de dev utiles

```bash
# Formatter + linter
uv run ruff format .
uv run ruff check . --fix

# Type checking
uv run mypy main.py

# Scan de sécurité
uv run bandit -r . -x ./tests

# Tests
uv run pytest

# Tout lancer d'un coup (à mettre dans un script ou Makefile)
uv run ruff format . && uv run ruff check . --fix && uv run mypy main.py && uv run pytest
```

---

## 7. (Optionnel) Makefile pour automatiser

```makefile
.PHONY: dev lint format check test all

dev:
	uv run fastapi dev

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

typecheck:
	uv run mypy main.py

security:
	uv run bandit -r . -x ./tests

test:
	uv run pytest

check: format lint typecheck security

all: check test
```

Utilisation :

```bash
make dev       # lancer le serveur
make check     # format + lint + typecheck + security
make all       # check + tests
```

---

## 8. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.mypy_cache/
.ruff_cache/

# Environment
.env
.venv/

# uv
.python-version

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

---

## Structure finale du projet

```
genai-fastapi-service/
├── main.py
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .env
├── .python-version
├── README.md
└── tests/
    └── test_main.py
```

---

## Commandes `uv` — cheat sheet

| Commande | Description |
|----------|-------------|
| `uv init` | Initialiser un nouveau projet |
| `uv python pin 3.11` | Fixer la version Python |
| `uv add <package>` | Ajouter une dépendance |
| `uv add --dev <package>` | Ajouter une dépendance dev |
| `uv remove <package>` | Supprimer une dépendance |
| `uv run <command>` | Exécuter dans l'environnement du projet |
| `uv sync` | Synchroniser les dépendances depuis `uv.lock` |
| `uv pip list` | Lister les packages installés |
| `uv lock` | Mettre à jour le lockfile |

---

## Ressources

- [uv documentation](https://docs.astral.sh/uv/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [Livre — Building Generative AI Services with FastAPI](https://www.oreilly.com/library/view/building-generative-ai/9781098164843/)