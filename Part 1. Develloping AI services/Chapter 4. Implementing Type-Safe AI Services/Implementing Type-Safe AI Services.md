# Chapter 4 — Implementing Type-Safe AI Services

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Pourquoi la type safety est essentielle](#1-pourquoi-la-type-safety-est-essentielle)
2. [Type annotations et TypeAlias](#2-type-annotations-et-typealias)
3. [Dataclasses vs Pydantic](#3-dataclasses-vs-pydantic)
4. [Pydantic Models en détail](#4-pydantic-models-en-détail)
5. [Field Constraints et Validators](#5-field-constraints-et-validators)
6. [Computed Fields](#6-computed-fields)
7. [Export et sérialisation](#7-export-et-sérialisation)
8. [Pydantic Settings (variables d'environnement)](#8-pydantic-settings-variables-denvironnement)
9. [Intégration avec FastAPI](#9-intégration-avec-fastapi)

---

## 1. Pourquoi la type safety est essentielle

Python est **dynamiquement typé** → les types sont vérifiés au runtime, pas en avance. Ça accélère le prototypage mais crée des problèmes quand le code devient complexe.

**Sans types** :
- Impossible de savoir quels types circulent dans le code
- Les changements de schéma BDD/API cassent le code silencieusement
- Bugs découverts en production au lieu du dev
- Debugging qui prend des heures au lieu de minutes

**Avec types** :
- Les outils (mypy, Pylance) détectent les erreurs **avant l'exécution**
- L'IDE offre autocomplétion et warnings en temps réel
- Les changements de schéma sont immédiatement signalés
- Peut être intégré dans le pipeline CI/CD pour bloquer les déploiements cassés

**FastAPI utilise les types pour** :
- Définir les params de route (path, query, body, headers)
- Convertir les données automatiquement
- Valider les requêtes/réponses
- Générer la documentation OpenAPI/Swagger

---

## 2. Type annotations et TypeAlias

### Annotations de base

```python
def timestamp_to_isostring(date: int) -> str:
    return datetime.fromtimestamp(date).isoformat()

# mypy détecte l'erreur :
timestamp_to_isostring("27 Jan 2025")
# error: incompatible type "str"; expected "int"
```

### TypeAlias — types réutilisables

```python
from typing import Literal, TypeAlias

SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float]

price_table: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}
```

### Annotated — types avec métadonnées (recommandé par FastAPI)

```python
from typing import Annotated, Literal

SupportedModels = Annotated[
    Literal["gpt-3.5-turbo", "gpt-4o"], "Supported text models"
]
PriceTableType = Annotated[
    dict[SupportedModels, float], "Supported model pricing table"
]
```

`Annotated` nécessite au minimum 2 arguments : le type + au moins une métadonnée.

---

## 3. Dataclasses vs Pydantic

### Dataclasses (stdlib Python 3.7+)

Utiles pour : grouper des données, simplifier les signatures de fonctions, éviter le code bloat.

```python
from dataclasses import dataclass

@dataclass
class Message:
    prompt: str
    response: str | None
    model: SupportedModels

@dataclass
class MessageCostReport:
    req_costs: float
    res_costs: float
    total_costs: float

# Simplifie les signatures :
def calculate_costs(message: Message) -> MessageCostReport:
    ...
```

### Limites des dataclasses

| Feature | Dataclasses | Pydantic |
|---------|:-----------:|:--------:|
| Grouper des données | ✅ | ✅ |
| Type hints | ✅ | ✅ |
| Auto parsing (ex: str → datetime) | ❌ | ✅ |
| Validation de champs (email, URL...) | ❌ | ✅ |
| Sérialisation JSON | Basique | ✅ Complet |
| Filtrage de champs (exclude_none...) | ❌ | ✅ |
| Computed fields | ❌ | ✅ |
| Validators custom | ❌ | ✅ |
| Performance (core en Rust) | — | ✅ |

> 💡 FastAPI convertit automatiquement les dataclasses en Pydantic dataclasses sous le capot. Pour un nouveau projet, utiliser Pydantic directement.

---

## 4. Pydantic Models en détail

### Modèle de base

```python
from pydantic import BaseModel
from typing import Literal

class TextModelRequest(BaseModel):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0   # valeur par défaut
```

### Héritage et modèles composés

```python
class ModelRequest(BaseModel):
    prompt: str

class ModelResponse(BaseModel):
    request_id: str
    ip: str | None
    content: str | None
    created_at: datetime = datetime.now()

# Hérite de ModelRequest → inclut automatiquement `prompt`
class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: float = 0.0

class TextModelResponse(ModelResponse):
    tokens: int

# Type réutilisable
ImageSize = Annotated[tuple[int, int], "Width and height in pixels"]

class ImageModelRequest(ModelRequest):
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize
    num_inference_steps: int = 200

class ImageModelResponse(ModelResponse):
    size: ImageSize
    url: str
```

---

## 5. Field Constraints et Validators

### Contraintes avec Field

```python
from pydantic import BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt
from typing import Annotated
from uuid import uuid4

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)]

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    ip: Annotated[str, IPvAnyAddress] | None
    content: Annotated[str | None, Field(min_length=0, max_length=10000)]

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]

class ImageModelRequest(ModelRequest):
    output_size: Annotated[tuple[PositiveInt, PositiveInt], "Image size"]
    num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200
    url: Annotated[str, HttpUrl] | None = None
```

### Types contraints intégrés

| Type | Valide |
|------|--------|
| `EmailStr` | Format email (nécessite `pip install email-validator`) |
| `PositiveInt` | Entier > 0 |
| `HttpUrl` | URL HTTP/HTTPS valide |
| `IPvAnyAddress` | IPv4 ou IPv6 |
| `PostgresDsn` | Connection string Postgres |
| `UUID4` | UUID version 4 |

### Validators custom avec AfterValidator

```python
from pydantic import AfterValidator, validate_call

ImageSize = Annotated[tuple[PositiveInt, PositiveInt], "Image size"]

@validate_call
def is_square_image(value: ImageSize) -> ImageSize:
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid size: {value} - expected 512 or 1024")
    return value

@validate_call
def is_valid_inference_step(num_inference_steps: int, model: SupportedModels) -> int:
    if model == "tinysd" and num_inference_steps > 2000:
        raise ValueError("TinySD: max 2000 inference steps")
    return num_inference_steps

# Types réutilisables avec validators attachés
OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)]
InferenceSteps = Annotated[
    int,
    AfterValidator(lambda v, values: is_valid_inference_step(v, values["model"])),
]

class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize                    # validé par is_square_image
    num_inference_steps: InferenceSteps = 200  # validé par is_valid_inference_step
```

> Alternatives : `@field_validator` (un seul champ) et `@model_validator` (plusieurs champs) en décorateurs.

---

## 6. Computed Fields

Champs calculés automatiquement à partir d'autres champs :

```python
from pydantic import computed_field

class TextModelResponse(ModelResponse):
    model: SupportedModels
    price: Annotated[float, Field(ge=0, default=0.01)]

    @property
    @computed_field
    def tokens(self) -> int:
        return count_tokens(self.content)

    @property
    @computed_field
    def cost(self) -> float:
        return self.price * self.tokens
```

Les computed fields apparaissent dans `.model_dump()` et dans les réponses FastAPI automatiquement.

---

## 7. Export et sérialisation

### Vers dict ou JSON

```python
response = TextModelResponse(content="Hello", ip=None)

response.model_dump(exclude_none=True)
# {'content': 'Hello', 'cost': 0.01, 'tokens': 1, ...}  (sans les champs None)

response.model_dump_json(exclude_unset=True)
# '{"ip":null,"content":"Hello","tokens":1,"cost":0.01}'
```

### Options de filtrage

| Option | Exclut |
|--------|--------|
| `exclude_none=True` | Champs avec valeur `None` |
| `exclude_unset=True` | Champs non explicitement définis |
| `exclude_defaults=True` | Champs restés à leur valeur par défaut |

> Utile pour les filtres de recherche : exclure les paramètres non fournis par le client.

---

## 8. Pydantic Settings (variables d'environnement)

```bash
uv add pydantic-settings
```

### Fichier `.env`

```env
APP_SECRET=asdlkajdlkajdklaslkldjkasldjkasdjaslk
DATABASE_URL=postgres://sa:password@localhost:5432/cms
CORS_WHITELIST=["https://xyz.azurewebsites.net","http://localhost:3000"]
```

### Settings class

```python
from pydantic import Field, HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    port: Annotated[int, Field(default=8000)]
    app_secret: Annotated[str, Field(min_length=32)]
    pg_dsn: Annotated[
        PostgresDsn,
        Field(alias="DATABASE_URL", default="postgres://user:pass@localhost:5432/db"),
    ]
    cors_whitelist_domains: Annotated[
        set[HttpUrl],
        Field(alias="CORS_WHITELIST", default=["http://localhost:3000"]),
    ]

settings = AppSettings()
```

**Fonctionnement** :
1. Cherche les valeurs fournies dans le code (defaults)
2. Si non trouvées → lit les **variables d'environnement** (mapping auto : `app_secret` → `APP_SECRET`)
3. `alias` permet de mapper un nom de champ différent du nom de la variable env
4. Validation Pydantic appliquée (format Postgres, URLs valides, longueur min du secret...)

**Changer de fichier env** :
```python
test_settings = AppSettings(_env_file="test.env")
```

---

## 9. Intégration avec FastAPI

### Endpoint typé avec Pydantic

```python
@app.post("/generate/text")
def serve_text_controller(
    request: Request, body: TextModelRequest = Body(...)
) -> TextModelResponse:
    if body.model not in ["tinyLlama", "gemma2b"]:
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

**Ce que FastAPI fait automatiquement** :
- Valide le body JSON contre `TextModelRequest` (erreur 422 si invalide)
- Sérialise `TextModelResponse` en JSON (avec computed fields)
- Génère la doc Swagger avec les schémas Pydantic
- mypy vérifie que le return type correspond à `TextModelResponse`

### Réponse automatique en cas d'erreur de validation

```json
{
    "detail": [{
        "type": "literal_error",
        "loc": ["body", "model"],
        "msg": "Input should be 'tinyllama' or 'gemma2b'",
        "input": "gpt-4o",
        "ctx": {"expected": "'tinyllama' or 'gemma2b'"}
    }]
}
```

---

## Récap — Cheat sheet

### Quand utiliser quoi

| Besoin | Outil |
|--------|-------|
| Grouper des données simples | `dataclass` |
| Validation + sérialisation | `BaseModel` (Pydantic) |
| Contraintes sur les champs | `Field()` + `Annotated` |
| Validation custom | `@field_validator` / `AfterValidator` |
| Champs calculés | `@computed_field` |
| Variables d'environnement | `BaseSettings` (pydantic-settings) |
| Types réutilisables | `TypeAlias` ou `Annotated` |

### Dépendances

```bash
uv add pydantic pydantic-settings loguru tiktoken
```

### Points clés

1. **Toujours typer** les fonctions, paramètres et retours → détection précoce des bugs
2. **Pydantic > dataclasses** pour FastAPI (validation, sérialisation, computed fields)
3. **Field constraints** (`ge`, `le`, `min_length`, `max_length`) protègent contre les inputs invalides
4. **AfterValidator** pour la logique de validation custom (ex : taille d'image carrée)
5. **BaseSettings** pour charger et valider les variables d'environnement de manière type-safe
6. **Computed fields** encapsulent la logique de calcul dans le modèle (tokens, coût)
7. FastAPI convertit automatiquement les dataclasses en Pydantic → migration possible sans réécriture