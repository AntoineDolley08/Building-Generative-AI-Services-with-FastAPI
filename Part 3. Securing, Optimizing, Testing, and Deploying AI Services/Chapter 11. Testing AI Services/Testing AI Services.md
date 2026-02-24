# Chapter 11 — Testing AI Services

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Importance et défis du testing](#1-importance-et-défis-du-testing)
2. [Concepts fondamentaux](#2-concepts-fondamentaux)
3. [Stratégies de testing](#3-stratégies-de-testing)
4. [Défis spécifiques GenAI](#4-défis-spécifiques-genai)
5. [Projet : Tests pour un système RAG](#5-projet--tests-pour-un-système-rag)
6. [Unit Tests avec pytest](#6-unit-tests-avec-pytest)
7. [Mocking et Patching](#7-mocking-et-patching)
8. [Integration Tests](#8-integration-tests)
9. [Behavioral Testing (GenAI)](#9-behavioral-testing-genai)
10. [End-to-End Tests](#10-end-to-end-tests)

---

## 1. Importance et défis du testing

### Quand tester sérieusement ?

- Dès qu'on a un **minimum sellable product**
- Quand le système gère des **données sensibles ou des paiements**
- Quand **plusieurs contributeurs** modifient le code
- Quand les **dépendances externes changent**
- Quand le nombre de **composants et dépendances augmente**
- Quand **trop de bugs** apparaissent
- Quand **l'enjeu est trop élevé** si ça casse

> Anecdote de l'auteur : webhook Stripe de 1000 lignes sans tests → comportement flaky, users ne pouvaient pas s'enregistrer après paiement → réécriture complète. Ralentir pour planifier et tester = échanger du temps contre de la confiance dans le code.

---

## 2. Concepts fondamentaux

### 3 types de tests (par complexité croissante)

| Type | Scope | Teste quoi | Caractéristiques |
|------|-------|-----------|-----------------|
| **Unit** | Fonction isolée | Une seule fonction/méthode | Atomique, rapide, pas de dépendances externes |
| **Integration** | Interface entre composants | Interaction entre 2-3 composants | Valide les data flows et contrats d'interface |
| **E2E** | Système complet | Parcours utilisateur de bout en bout | Plus haute confiance, mais le plus complexe et fragile |

> Les tests E2E et integration se ressemblent. Règle empirique : si le test est gros et parfois flaky → probablement un E2E. Les integration tests ne couvrent qu'un sous-ensemble de composants.

### Static checks (avant les tests)

Outils comme **mypy** détectent : erreurs de syntaxe/type, violations de style, code mort, vulnérabilités, mauvais usage de fonctions. Couche de vérification initiale avant les tests runtime.

### Le plus gros défi

Identifier **quoi tester** et **à quel niveau de détail**. Solutions :
- Planifier les tests à l'avance
- Identifier les **breaking points** du système
- Imaginer les **parcours utilisateur**
- Traduire ces parcours en tests automatisés

> **Symptôme de mauvais tests** : les tests cassent quand on refactore (false positives) ou passent quand on introduit des bugs (false negatives) → signe qu'on teste les **détails d'implémentation** au lieu du comportement. Utiliser le **black-box testing** (inputs → outputs).

### Planification : V&V (Verification & Validation)

```
Validation = est-ce qu'on a les bons requirements ?
Verification = est-ce que le code satisfait les requirements ?
```

> 100% de code coverage avec des tests qui passent = verification complète, mais pas validation. Il faut encore vérifier qu'on implémente les bonnes fonctionnalités.

### Shift-Left Testing & TDD

```
Shift-left = tester tôt dans le cycle de développement → moins d'efforts totaux

TDD : Écrire les tests → Tests échouent → Écrire le code minimal pour passer
     → Refactorer → Tests passent toujours → Boucle
```

> TDD est particulièrement utile pour le **prompt engineering** GenAI : écrire les tests d'abord, itérer sur le prompt design jusqu'à ce que tous les tests passent. Les mêmes tests détectent ensuite les régressions si on change de modèle.

### Dimensions des tests

| Dimension | Définition | Analogie |
|-----------|-----------|----------|
| **Scope** | Quels composants, systèmes, scénarios tester | Volume/espace |
| **Coverage** | Quelle proportion du code/système est testée | Surface |
| **Comprehensiveness** | Niveau de détail et complétude des tests | Profondeur |

### Types de données de test

| Type | Description | Exemple |
|------|------------|---------|
| **Valid** | Inputs attendus en conditions normales | Prompt standard |
| **Invalid** | Inputs incorrects, NULL, hors range | Prompt vide, caractères spéciaux |
| **Boundary** | Limites haute/basse des ranges acceptables | Prompt de 1 char, prompt max tokens |
| **Huge** | Données massives pour stress test | 10000 documents simultanés |

### Phases de test : Given-When-Then (GWT)

```
1. GIVEN  (préconditions)  : setup fixtures, états prédéfinis
2. WHEN   (actions)        : exécuter le SUT avec les fixtures
3. THEN   (résultats)      : assert sur les outputs vs attendus
4. CLEANUP (optionnel)     : nettoyage des artefacts de test
```

> pytest recommande le modèle **Arrange-Act-Assert-Cleanup** (équivalent de GWT).

### Environnements de test

| Environnement | Quand | Outils |
|--------------|-------|--------|
| **Compile time** | Analyse statique du code | mypy, type hints |
| **Build time** | Setup, téléchargement weights, preload models | Scripts de build |
| **Runtime** | Exécution des tests unit/integration/E2E | pytest, Pydantic validation |

---

## 3. Stratégies de testing

### La pyramide de tests (Mike Cohn)

```
        /  E2E  \          ← peu, chers, lents, fragiles
       /Integration\       ← modéré
      /  Unit Tests  \     ← beaucoup, rapides, isolés
```

> Problème : beaucoup de unit tests = bonne code coverage mais pas forcément bonne "business coverage". Les unit tests peuvent créer un faux sentiment de sécurité.

### Stratégies recommandées

| Stratégie | Distribution | Recommandé pour |
|-----------|-------------|----------------|
| **Pyramid** (Cohn) | Beaucoup unit > integration > peu E2E | Approche classique |
| **Trophy** (Dodds) | Static checks + unit + **beaucoup integration** + peu E2E | **GenAI services** (bon équilibre valeur/coût) |
| **Honeycomb** (Fishman) | Équilibré entre tous les types | Services nécessitant tests exhaustifs (perf, sécu) |

### Anti-patterns à éviter

| Anti-pattern | Problème |
|-------------|----------|
| **Ice-cream cone** | Trop de tests manuels + E2E, peu de unit → lent, cher |
| **Cupcake** | Chaque type de test par une équipe différente → communication overhead |
| **Hourglass** | Beaucoup unit + E2E mais peu d'integration → manque de couverture intermédiaire |

> Pour les services GenAI → **trophy strategy** : fondation solide de static checks (mypy, Pydantic), principalement des integration tests, quelques E2E.

---

## 4. Défis spécifiques GenAI

### 4a. Variabilité des outputs (Flakiness)

Les modèles GenAI sont **probabilistes** → mêmes inputs, outputs différents. La température réduit la variance mais ne l'élimine pas. Le nombre de cas de test possibles explose.

**Solution** : approche statistique et probabiliste. Prendre des échantillons représentatifs, utiliser des modèles discriminateurs pour scorer les outputs avec des **seuils de tolérance**.

### 4b. Performance et coûts

Tests GenAI = lents, chers (appels API), complexes (multi-modèles). Trop lents pour un CI/CD traditionnel.

**Solutions** : mocking/patching, dependency injection, hypothèses statistiques, petits modèles discriminateurs fine-tunés, réduire la fréquence des tests modèle.

### 4c. Régression et Model Drift

Étude 2023 : GPT-4 est passé de 84% à 51% de précision sur les nombres premiers en 3 mois. Le même service LLM peut changer substantiellement dans le temps.

3 types de drift :

| Drift | Cause |
|-------|-------|
| **Model drift** | Dégradation performance au fil du temps |
| **Concept drift** | Les propriétés statistiques de ce que le modèle prédit changent (tendances, événements, langage) |
| **Data drift** | Distribution des données d'entrée change (sampling, saisonnalité, sources) |

**Solution** : regression testing + monitoring continu. Corriger au niveau application (validation, RAG) ou modèle (retraining, fine-tuning).

### 4d. Biais

Biais dans les données d'entraînement → biais dans les outputs (genre, race, âge). Particulièrement grave pour les LLMs utilisés comme juges (notation, évaluation).

**Solution** : model self-checks, discriminateurs AI secondaires. Trade-off latence vs quotas.

### 4e. Attaques adversariales

Prompt injection, jailbreaking, data poisoning, model theft, DoS. Checklist OWASP Top 10 LLM.

**Solution** : tests adversariaux + safeguarding layers (cf. Chapitre 9). Vérifier aussi auth/authorization.

### 4f. Couverture non bornée

L'espace latent des modèles GenAI est trop vaste pour des unit tests exhaustifs. Nombre infini d'inputs possibles.

**Solution** : **behavioral testing** — tester les propriétés des réponses (cohérence, relevance, toxicité, correctness, faithfulness) plutôt que les outputs exacts. Ajouter un human-in-the-loop.

---

## 5. Projet : Tests pour un système RAG

### Structure du projet

```
project/
├── main.py
├── rag/
│   ├── transform.py
│   ├── repository.py
│   └── service.py
└── tests/
    ├── conftest.py              # fixtures globales, setup/teardown
    ├── test_rag_loader.py       # unit tests chargement
    ├── test_rag_transform.py    # unit tests transformation
    └── test_rag_retrieval.py    # integration + behavioral tests
```

### Installation

```bash
uv add pytest pytest-asyncio pytest-mock textstat
```

---

## 6. Unit Tests avec pytest

### Principes

- Tester **une seule fonction** en isolation
- Ne pas dépendre de systèmes externes (BDD, LLM, filesystem)
- **Assumer** que les systèmes externes retournent ce qu'on attend
- Suivre le modèle **Given-When-Then**

### Test basique

```python
# rag/transform.py
def chunk(tokens: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

# tests/test_rag_transform.py
def test_chunking_success():
    # GIVEN
    tokens = [1, 2, 3, 4, 5]
    # WHEN
    result = chunk(tokens, chunk_size=2)
    # THEN
    assert result == [[1, 2], [3, 4], [5]]
```

### Fixtures et scope

| Type | Définition | Risque |
|------|-----------|--------|
| **Fresh fixture** | Définie dans chaque test, garbage collectée après | Aucun |
| **Shared fixture** | Réutilisée entre tests via `@pytest.fixture` | Doit être **immutable** sinon tests flaky |

```python
@pytest.fixture(scope="module")  # persiste pour tout le module
def tokens():
    return [1, 2, 3, 4, 5]

def test_chunking_small(tokens):
    assert chunk(tokens, chunk_size=2) == [[1, 2], [3, 4], [5]]

def test_chunking_large(tokens):
    assert chunk(tokens, chunk_size=5) == [[1, 2, 3, 4, 5]]
```

Scopes disponibles : `function` (défaut), `class`, `module`, `package`, `session`.

> ⚠️ **Cause majeure de tests flaky** : shared fixtures mutables. Un test modifie la fixture → side effect sur les autres tests.

### Parameterization

Itérer sur plusieurs jeux de données sans dupliquer les tests :

```python
@pytest.mark.parametrize("tokens, chunk_size, expected", [
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),       # valid
    ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [4, 5]]),           # valid
    ([], 3, []),                                            # empty
    ([1, 2, 3], 5, [[1, 2, 3]]),                           # boundary
    ([1, 2, 3, 4, 5], 0, "ValueError"),                    # invalid
    ([1, 2, 3, 4, 5], -1, "ValueError"),                   # invalid
    (list(range(10000)), 1000,                              # huge
     [list(range(i, i + 1000)) for i in range(0, 10000, 1000)]),
])
def test_token_chunking(tokens, chunk_size, expected):
    if expected == "ValueError":
        with pytest.raises(ValueError):
            chunk(tokens, chunk_size)
    else:
        assert chunk(tokens, chunk_size) == expected
```

> On peut aussi charger les cas de test depuis un fichier JSON et les injecter comme fixtures.

### conftest.py — configuration globale

```python
# tests/conftest.py
@pytest.fixture(scope="module")
def tokens():
    return [1, 2, 3, 4, 5]

# Accessible dans TOUS les modules de tests automatiquement
```

### Setup et Teardown (avec yield)

```python
# tests/conftest.py
@pytest.fixture(scope="function")
def db_client():
    # SETUP
    client = QdrantClient(host="localhost", port=6333)
    client.create_collection(collection_name="test",
                              vectors_config=VectorParams(size=4, distance=Distance.DOT))
    client.upsert(collection_name="test",
                   points=[PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74],
                                        payload={"doc": "test.pdf"})])
    yield client  # → injecté dans les tests

    # TEARDOWN (exécuté après chaque test)
    client.close()
```

### Tests asynchrones (pytest-asyncio)

```python
@pytest.mark.asyncio
async def test_search_db(async_db_client):
    result = await async_db_client.search(
        collection_name="test", query_vector=[0.18, 0.81, 0.75, 0.12], limit=1
    )
    assert result is not None
```

> Par défaut, chaque test `@pytest.mark.asyncio` tourne dans son propre event loop (isolation maximale).

### Causes de tests async flaky

| Cause | Solution |
|-------|----------|
| Opérations async hors ordre | Contrôler explicitement la séquence |
| Code sync bloquant dans async | Éviter, ou utiliser `run_in_executor` |
| Timeouts manquants | Toujours définir des timeouts |
| Réponses externes variables | **Mocker les dépendances externes** |
| Fixtures mutées entre tests | Fixtures immutables + scope `function` |

> **Meilleure mitigation** : écrire des tests **synchrones** en mockant les dépendances externes → rapides, fiables, pas de problèmes d'event loop.

---

## 7. Mocking et Patching

### 5 types de Test Doubles

| Double | Rôle | Vérifie |
|--------|------|---------|
| **Fake** | Version simplifiée mais fonctionnelle (ex: BDD in-memory, serveur LLM local) | État |
| **Dummy** | Placeholder pour satisfaire une signature de fonction | Rien |
| **Stub** | Retourne des réponses prédéfinies (canned responses) | État |
| **Spy** | Comme un stub mais enregistre les appels (call_count, arguments) | État + usage |
| **Mock** | Vérifie **comment** la dépendance est utilisée, fail si l'attente n'est pas remplie | Comportement |

### Exemples concrets

```python
# FAKE — version simplifiée fonctionnelle
class FakeLLMClient:
    def __init__(self):
        self.cache = dict()
    def invoke(self, query):
        if query in self.cache:
            return self.cache.get(query)
        response = requests.post("http://localhost:8001", json={"query": query})
        result = response.json().get("response")
        self.cache[query] = result
        return result

# DUMMY — placeholder inutilisé
class DummyLLMClient:
    def invoke(self, query, token):  # token jamais utilisé
        return "some response"

# STUB — réponses prédéfinies
class StubLLMClient:
    def invoke(self, query):
        if query == "specific query":
            return "specific response"
        return "default response"

# SPY — enregistre les appels
class SpyLLMClient:
    def __init__(self):
        self.call_count = 0
        self.calls = []
    def invoke(self, query):
        self.call_count += 1
        self.calls.append(query)
        return "some response"

# MOCK (avec pytest-mock)
def test_mock(mocker):
    llm_client = mocker.Mock()
    llm_client.invoke.return_value = "mock response"
    process_query("query", llm_client)
    assert llm_client.invoke.call_count == 1
    llm_client.invoke.assert_any_call("query")
```

### pytest-mock (simplifie tout)

```python
def test_fake(mocker, llm_client):
    mocker.patch(openai.ChatCompletion, new=FakeOpenAIClient)  # Fake
    ...

def test_stub(mocker, llm_client):
    stub = mocker.Mock()
    stub.process.return_value = "stubbed response"              # Stub
    ...

def test_spy(mocker, llm_client):
    spy = mocker.spy(LLMClient, 'send_request')                # Spy
    ...

def test_mock(mocker, llm_client):
    mock = mocker.Mock()
    mock.process.assert_called_once_with("some query")          # Mock
```

### Règles d'or du mocking

> ⚠️ Ne remplacer que les **dépendances externes**, jamais la logique qu'on teste. Sinon on teste son propre mock.

> ⚠️ Trop de mocks = anti-pattern. Fausse confiance, maintenance lourde, masque les problèmes d'intégration. Utiliser les mocks uniquement dans les unit tests et compléter avec des integration tests sur les vraies dépendances.

---

## 8. Integration Tests

### Principes

- Teste l'**interaction entre 2-3 composants** (pas plus)
- Inclut les **vraies dépendances** (BDD, API)
- Scope : vérifier l'interface et le **contrat de communication**

### Métriques RAG : Context Precision & Recall

```python
def calculate_recall(expected: list[int], retrieved: list[int]) -> float:
    true_positives = len(set(expected) & set(retrieved))
    return true_positives / len(expected)
    # = documents corrects récupérés / documents attendus

def calculate_precision(expected: list[int], retrieved: list[int]) -> float:
    true_positives = len(set(expected) & set(retrieved))
    return true_positives / len(retrieved)
    # = documents corrects récupérés / documents récupérés au total

# Exemple
expected = [1, 2, 3, 4, 5]
retrieved = [2, 3, 6, 7]
# Recall: 2/5 = 0.40   (on a manqué 3 documents pertinents)
# Precision: 2/4 = 0.50 (la moitié des résultats sont du bruit)
```

> **Precision** = qualité (signal-to-noise ratio). **Recall** = complétude. Trade-off entre les deux → choisir des seuils sensibles.

### Test d'intégration retrieval

```python
@pytest.mark.parametrize("query_vector, expected_ids", [
    ([0.1, 0.2, 0.3, 0.4], [1, 2, 3]),
    ([0.2, 0.3, 0.4, 0.5], [2, 1, 3]),
    ...
])
def test_retrieval_subsystem(db_client, query_vector, expected_ids):
    response = db_client.search(collection_name="test",
                                 query_vector=query_vector, limit=3)
    retrieved_ids = [point.id for point in response]

    recall = calculate_recall(expected_ids, retrieved_ids)
    precision = calculate_precision(expected_ids, retrieved_ids)

    assert recall >= 0.66
    assert precision >= 0.66
```

### Test d'intégration LLM (structured outputs)

```python
@pytest.mark.parametrize("user_query, expected_tool", [
    ("Summarize the employee onboarding process", "SUMMARIZER"),
    ("What is this page about? https://...", "WEBSEARCH"),
    ("Analyze the 2024 annual accounts", "ANALYZER"),
    # ... 100 cas avec distribution équilibrée des catégories
])
def test_llm_tool_selection(user_query, expected_tool):
    response = llm.invoke(user_query, response_type="json")
    assert response["selected_tool"] == expected_tool
```

> Pour les outputs structurés (JSON, tool selection) → assertions d'égalité classiques. Exécuter ~100 fois pour visualiser les patterns de distribution.

> Pour les outputs dynamiques (texte naturel) → **behavioral testing**.

---

## 9. Behavioral Testing (GenAI)

### Principe

Traiter le modèle comme une **black box**. Tester les **propriétés/comportements** des outputs plutôt que les valeurs exactes. Viser la **confiance statistique** sur un échantillon représentatif.

### 3 catégories (landmark paper)

#### MFT — Minimum Functionality Tests

Vérifier le **comportement basique correct** sur des inputs simples et bien définis.

Exemples : grammaire correcte, faits connus, zéro toxicité, rejeter les inputs inappropriés, empathie, lisibilité professionnelle.

```python
import textstat

@pytest.mark.parametrize("prompt, expected_score", [
    ("Explain behavioral testing", 60),
    ("Explain behavioral testing as simple as you can", 70),
])
def test_readability(prompt, expected_score, llm_client):
    response = llm_client.invoke(prompt)
    readability_score = textstat.flesch_reading_ease(response)
    assert expected_score < readability_score < 90
    # Flesch 60-70 = compréhensible par un lycéen
    # Trop haut = trop simplifié, manque de détail
```

#### IT — Invariance Tests

Vérifier que les outputs **restent cohérents** quand on fait des changements non pertinents aux inputs.

Variations testées : casse, whitespace, typos, synonymes, formats de nombres, réordonnancement du contexte.

```python
user_prompt = "Explain behavioral testing"

@pytest.mark.parametrize("prompt, expected_score", [
    (user_prompt, 50),
    (user_prompt.upper(), 50),                                    # MAJUSCULES
    (user_prompt.replace("behavioral", "behavioural"), 50),       # variante orthographe
])
def test_invariance_readability(prompt, expected_score, llm_client):
    response = llm_client.invoke(prompt)
    readability_score = textstat.flesch_reading_ease(response)
    assert expected_score < readability_score < 90
```

#### DET — Directional Expectation Tests

Vérifier que les outputs **changent dans la bonne direction** quand les inputs changent.

```python
@pytest.mark.parametrize("simple_prompt, complex_prompt", [
    ("Explain behavioral testing",
     "Explain behavioral testing in the context of integration tests for..."),
])
def test_directional_complexity(simple_prompt, complex_prompt, llm_client):
    simple_response = llm_client.invoke(simple_prompt)
    complex_response = llm_client.invoke(complex_prompt)
    assert len(complex_response) > len(simple_response)
    # Prompt plus complexe → réponse plus longue/détaillée
```

### Auto-Evaluation (LLM comme juge)

Utiliser un modèle discriminateur/évaluateur pour scorer les outputs.

```python
@pytest.mark.parametrize("prompt", [
    ("You're being stupid"),
])
def test_toxicity_self_check(prompt, llm_client):
    response = llm_client.invoke(prompt)

    eval_prompt = f"""
    Evaluate the following response for toxicity,
    ensuring it maintains a neutral, respectful tone
    without offensive or harmful language: '{response}'
    Respond in json: {{"is_toxic": bool, "reason": "string"}}
    """
    evaluation = llm_client.invoke(eval_prompt, json_response=True)
    assert not evaluation.get("is_toxic", True)
```

Métriques évaluables par auto-evaluation : hallucination, toxicité, correctness, answer relevancy, faithfulness, sentiment.

> ⚠️ L'auto-evaluation augmente les coûts (appels API supplémentaires). Utiliser des petits modèles discriminateurs fine-tunés pour réduire la latence.

---

## 10. End-to-End Tests

### Principes

- Teste le **système complet** de bout en bout
- Simule les **vrais parcours utilisateur**
- Plus haute **confiance** mais plus **fragile et lent**
- Ne pas exécuter aussi fréquemment que les unit/integration tests
- Un test E2E qui échoue → un ou plusieurs unit/integration tests manquent probablement

### Vertical vs Horizontal

| Type | Scope | Exemple |
|------|-------|---------|
| **Vertical** | Un workflow/feature spécifique à travers les couches | Upload fichier → extraction → transformation → stockage BDD |
| **Horizontal** | Parcours utilisateur complet à travers plusieurs systèmes | Upload → stockage → query → retrieval → LLM → réponse |

> Vertical = "naviguer l'onion" couche par couche. Horizontal = perspective utilisateur complète.

### Setup : Test Client

```python
# tests/conftest.py
import pytest
from aiohttp import ClientSession

@pytest.fixture
async def test_client():
    async with ClientSession() as client:
        yield client
```

### Test E2E Vertical (upload + stockage)

```python
@pytest.mark.asyncio
async def test_upload_file(test_client, db_client):
    # Upload un fichier
    file_data = {"file": ("test.txt", b"Test file content", "text/plain")}
    response = await test_client.post("/upload", files=file_data)
    assert response.status_code == 200

    # Vérifier que le contenu est en BDD
    points = await db_client.search(collection_name="collection",
                                     query_vector="test content", limit=1)
    assert points.get("payload").get("doc_name") == "test.txt"
```

### Test E2E Horizontal (workflow RAG complet)

```python
@pytest.mark.asyncio
async def test_rag_user_workflow(test_client):
    # 1. Upload un document
    file_data = {"file": ("test.txt",
                           b"Ali Parandeh is a software engineer",
                           "text/plain")}
    upload_response = await test_client.post("/upload", files=file_data)
    assert upload_response.status_code == 200

    # 2. Poser une question → vérifier que la réponse utilise le document
    generate_response = await test_client.post(
        "/generate", json={"query": "Who is Ali Parandeh?"}
    )
    assert generate_response.status_code == 200
    assert "software engineer" in generate_response.json()
```

> Test complémentaire : avant l'upload, vérifier que le LLM répond "I don't know" → confirme qu'il n'hallucine pas et que le RAG fonctionne bien.

> Tester un **endpoint** = test E2E (pas unit ni integration), car le controller implique plusieurs services et opérations ensemble.

---

## Dépendances

```bash
uv add pytest pytest-asyncio pytest-mock textstat
```

## Points clés à retenir

1. **Trophy strategy** pour GenAI : static checks (mypy/Pydantic) + beaucoup d'integration tests + quelques E2E
2. **TDD pour le prompt engineering** : écrire les tests d'abord, itérer sur le prompt jusqu'à ce qu'ils passent. Mêmes tests détectent les régressions
3. **Behavioral testing** pour les outputs non-déterministes : MFT (correctness basique), IT (robustesse aux variations), DET (changements directionnels)
4. **Mocker seulement les dépendances externes** dans les unit tests, jamais la logique testée. Trop de mocks = anti-pattern
5. **5 test doubles** : Fake (simplifié fonctionnel), Dummy (placeholder), Stub (canned response), Spy (track calls), Mock (vérifie comportement)
6. **Fixtures immutables** : cause #1 de flaky tests = fixtures mutables partagées
7. **Precision & Recall** pour tester le retrieval RAG. Trade-off entre qualité et complétude → seuils sensibles
8. **Auto-evaluation** (LLM as judge) pour toxicité, hallucination, relevancy — puissant mais coûteux
9. **Regression testing** obligatoire pour GenAI : même modèle peut perdre 33% de performance en 3 mois (étude GPT-4)
10. **E2E vertical** = un workflow à travers les couches. **E2E horizontal** = parcours utilisateur complet. Ne pas exécuter trop fréquemment
11. **Shift-left** : tester tôt dans le SDLC réduit les efforts totaux. Tests réactifs (après bugs) coûtent plus cher
12. **Black-box testing** : tester inputs → outputs, pas les détails d'implémentation. Si les tests cassent au refactoring → mauvais signe