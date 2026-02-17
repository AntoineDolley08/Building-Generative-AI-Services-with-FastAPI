# Chapter 3 — AI Integration and Model Serving

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Language Models (Transformers)](#1-language-models-transformers)
2. [Audio Models (Bark)](#2-audio-models-bark)
3. [Vision Models (Stable Diffusion)](#3-vision-models-stable-diffusion)
4. [Video Models](#4-video-models)
5. [3D Models (Shap-E)](#5-3d-models-shap-e)
6. [Stratégies de Model Serving](#6-stratégies-de-model-serving)
7. [Middleware pour le monitoring](#7-middleware-pour-le-monitoring)
8. [Récap & cheat sheets](#8-récap--cheat-sheets)

---

## 1. Language Models (Transformers)

### Transformers vs RNNs

**RNNs (ancien paradigme)** :
- Traitent le texte **séquentiellement** (token par token)
- Maintiennent un **state vector** (mémoire) transportant l'info d'un token au suivant
- **Problème** : plus on avance dans la séquence, plus l'impact des premiers tokens diminue → perte de contexte
- Entraînement **non parallélisable** sur GPU

**Transformers (paradigme actuel)** — Paper : *"Attention Is All You Need"* (Vaswani et al., 2017)
- Pas de mémoire cachée (state vector)
- **Self-attention** : modélise les relations entre **tous les mots** (pas juste les voisins)
- Traitement **non-séquentiel** → parallélisable sur GPU → scalable

```
RNN :    mot1 → mot2 → mot3 → mot4  (séquentiel, perd le contexte)
Transformer : mot1 ↔ mot2 ↔ mot3 ↔ mot4  (tous connectés entre eux)
```

### Attention heads

- Blocs spécialisés capturant les **relations pairwise** entre mots (attention maps)
- Plusieurs heads par couche → analyse sous **différents angles simultanément**
- Plus de heads/couches = meilleure compréhension des patterns complexes

### Pipeline de traitement du texte

**1. Tokenization** — Texte → tokens → IDs numériques

```
"FastAPI is great" → ["Fast", "API", " is", " great"] → [12043, 7112, 374, 2294]
```

**2. Embedding** — Tokens → vecteurs denses de floats capturant le sens sémantique

```
token "Fast" → [0.12, -0.34, 0.56, ..., 0.78]  (vecteur de dimension n)
```

Après entraînement, les mots de sens similaire ont des vecteurs proches.

**3. Positional Encoding** — Ajoute l'info d'ordre des mots (les transformers traitent tout en parallèle)

```
embedding_final = token_embedding + positional_embedding
```

**4. Cosine Similarity** — Mesure la similarité entre deux mots via l'angle entre vecteurs. Petit angle = sens similaire.

### Génération autoregressive

Le transformer prédit le **prochain token** basé sur tous les précédents, en boucle jusqu'à `<stop>` / `<eos>`.

```
Input: "How to set up"
→ prédit "a" → "How to set up a"
→ prédit "FastAPI" → "How to set up a FastAPI"
→ prédit <eos> → stop
```

### Context Window

Nombre max de tokens en mémoire.

| Modèle | Context Window |
|--------|---------------|
| GPT-4o-mini | ~128K tokens (~300 pages) |
| Magic.Dev LTM-2-mini | 100M tokens (~750 romans) |
| Autres modèles | Centaines de milliers de tokens |

Trade-offs : window courte → perte d'info | window longue → plus cher, plus lent sous charge.

### Paramètres d'inférence clés

| Paramètre | Rôle |
|-----------|------|
| `max_new_tokens` | Nombre max de tokens à générer |
| `do_sample` | `True` = sampling aléatoire, `False` = greedy (le plus probable) |
| `temperature` | Bas = précis/déterministe, Haut = créatif/aléatoire |
| `top_k` | Restreint aux K tokens les plus probables |
| `top_p` | Nucleus sampling : garde les tokens couvrant P% de probabilité |

### 3 variantes de Transformers

| Variante | Spécialisation | Tâches |
|----------|---------------|--------|
| **Encoder-Decoder** | Séquence → séquence | Traduction, résumé, Q&A |
| **Encoder-only** | Compréhension du sens | Sentiment analysis, NER, classification |
| **Decoder-only** | Prédiction du prochain token | Chatbots, génération de texte |

> 💡 Les chatbots (GPT, Llama, Mistral) sont des **decoder-only** transformers.

### Exemple : servir TinyLlama avec FastAPI

```
[Streamlit UI] → HTTP GET → [FastAPI] → [TinyLlama 1.1B]
   client.py                  main.py      models.py
```

```python
# models.py
import torch
from transformers import Pipeline, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16, device=device
    )
    return pipe

def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    predictions = pipe(
        prompt, temperature=temperature,
        max_new_tokens=256, do_sample=True, top_k=50, top_p=0.95,
    )
    return predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
```

```python
# main.py
@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    pipe = load_text_model()
    return generate_text(pipe, prompt)
```

> ⚠️ Le modèle est chargé/déchargé à chaque requête → anti-pattern. Solution : lifespan (section 6).

### Hardware pour LLMs open-source

| Taille modèle | Hardware nécessaire |
|---------------|-------------------|
| < 3B (TinyLlama) | CPU possible, GPU recommandé (~3 GB RAM) |
| < 30B | 1x GPU consumer (RTX 4090, 24 GB VRAM) |
| 70B (quantisé) | GPU 64 GB VRAM ou multi-GPU |
| 405B-480B (Llama 3.1, Arctic) | 8x H100 (640 GB VRAM total) |

> La plupart des organisations utilisent des modèles légers (≤ 3B) ou des APIs (OpenAI, Anthropic, Cohere, Mistral).

---

## 2. Audio Models (Bark)

Bark (Suno AI) : transformer capable de générer parole multilingue, musique, bruits de fond, effets sonores.

### Pipeline de synthèse (4 modèles chaînés)

```
Texte → [1. Semantic] → [2. Coarse Acoustics] → [3. Fine Acoustics] → [4. Encodec] → Audio WAV
```

| Étape | Modèle | Type | Rôle |
|-------|--------|------|------|
| 1 | **Semantic** | Autorégressif causal | Capture le sens sémantique |
| 2 | **Coarse Acoustics** | Autorégressif causal | Features audio brutes |
| 3 | **Fine Acoustics** | Auto-encoder non-causal | Raffine les détails audio |
| 4 | **Encodec** | Décodeur | Décode en waveform final |

### Code clé

```python
# models.py
from transformers import AutoProcessor, AutoModel

def load_audio_model():
    processor = AutoProcessor.from_pretrained("suno/bark-small", device=device)
    model = AutoModel.from_pretrained("suno/bark-small", device=device)
    return processor, model

def generate_audio(processor, model, prompt, preset):
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    return output, sample_rate
```

```python
# utils.py — conversion audio array → buffer streamable
import soundfile
from io import BytesIO

def audio_array_to_buffer(audio_array, sample_rate):
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format="wav")
    buffer.seek(0)
    return buffer
```

```python
# main.py
@app.get("/generate/audio", response_class=StreamingResponse)
def serve_audio_controller(prompt: str, preset: VoicePresets = "v2/en_speaker_1"):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(audio_array_to_buffer(output, sample_rate), media_type="audio/wav")
```

### Concepts clés

- **StreamingResponse** : pour contenus volumineux (audio, vidéo). Le client consomme au fur et à mesure.
- **Buffer mémoire (BytesIO)** > fichier sur disque pour la latence, mais trade-off avec la RAM.
- **Voice Presets** : typés avec `Literal["v2/en_speaker_1", "v2/en_speaker_9"]`

---

## 3. Vision Models (Stable Diffusion)

### Comment fonctionne Stable Diffusion

```
[Image] → Encode → [Latent Space (bruit blanc)] → Denoise (N steps) → Decode → [Nouvelle Image]
                            ↑
                    [Text Encoder] contrôle la génération via le prompt
```

**Entraînement (forward diffusion)** : images encodées → ajout progressif de bruit → le modèle apprend à retirer le bruit.

**Inférence (reverse diffusion)** : bruit aléatoire → débruitage itératif guidé par le prompt textuel → image générée.

> Plus d'inference steps = meilleure qualité, mais plus lent/coûteux.

### Code clé

```python
# models.py
from diffusers import DiffusionPipeline

def load_image_model():
    return DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", torch_dtype=torch.float32, device=device
    )

def generate_image(pipe, prompt):
    return pipe(prompt, num_inference_steps=10).images[0]  # PIL Image
```

```python
# utils.py
def img_to_bytes(image, img_format="PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=img_format)
    return buffer.getvalue()
```

```python
# main.py
@app.get("/generate/image", response_class=Response)
def serve_image_controller(prompt: str):
    pipe = load_image_model()
    output = generate_image(pipe, prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")
```

### Limitations SD open-source

| Limitation | Description |
|-----------|-------------|
| Cohérence | Ne reproduit pas tous les détails du prompt |
| Taille output | Tailles fixes (512×512 ou 1024×1024) |
| Composabilité | Contrôle limité de la composition |
| Photoréalisme | Détails qui trahissent la génération IA |
| Texte lisible | Certains modèles échouent |

### LoRA (Low-Rank Adaptation)

Technique de fine-tuning efficace : ajoute un **minimum de paramètres entraînables** par couche, les paramètres originaux restent figés. Réduit drastiquement la mémoire GPU nécessaire.

---

## 4. Video Models

Générer 1 seconde de vidéo = des dizaines de frames → **GPU quasi obligatoire**.

### Pipeline image-to-video (Stability AI)

```
[Image PIL] → resize(1024×576) → [Stable Video Diffusion] → [Frames PIL] → [av/ffmpeg] → MP4 stream
```

```python
# models.py
from diffusers import StableVideoDiffusionPipeline

def load_video_model():
    return StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16, variant="fp16", device=device,
    )

def generate_video(pipe, image, num_frames=25):
    image = image.resize((1024, 576))
    generator = torch.manual_seed(42)
    return pipe(image, decode_chunk_size=8, generator=generator, num_frames=num_frames).frames[0]
```

### Export frames → MP4

```python
import av

def export_to_video_buffer(images):
    buffer = BytesIO()
    output = av.open(buffer, "w", format="mp4")
    stream = output.add_stream("h264", 30)        # H.264, 30 FPS
    stream.pix_fmt = "yuv444p"                     # full color resolution
    stream.options = {"crf": "17"}                 # quasi-lossless
    for image in images:
        output.mux(stream.encode(av.VideoFrame.from_image(image)))
    output.mux(stream.encode(None))                # flush
    return buffer
```

```python
# main.py — premier endpoint POST avec File upload
@app.post("/generate/video", response_class=StreamingResponse)
def serve_video_controller(image: bytes = File(...), num_frames: int = 25):
    image = Image.open(BytesIO(image))
    model = load_video_model()
    frames = generate_video(model, image, num_frames)
    return StreamingResponse(export_to_video_buffer(frames), media_type="video/mp4")
```

### OpenAI Sora — Vision Transformer + Diffusion

Combine **Transformer** (scalabilité, dépendances long-range) + **Diffusion** (qualité, contrôle fin).

```
LLM : prédit le prochain TOKEN dans une séquence texte
Sora : prédit le prochain PATCH dans une séquence vidéo
```

Innovations : **3D U-Net** (3e dimension = temps), compression en **space-time patches**, génération en taille d'écran native.

**Capacités émergentes** :

| Capacité | Description |
|----------|-------------|
| 3D consistency | Objets cohérents quand la caméra bouge |
| Object permanence | Objets persistants hors-champ |
| World interaction | Actions affectent l'environnement |
| World simulation | Simule des mondes avec règles physiques |

---

## 5. 3D Models (Shap-E)

### Vocabulaire 3D

- **Vertices** : points (x, y, z) | **Edges** : segments entre vertices | **Faces** : polygones | **Mesh** : ensemble vertices + edges + faces

### Pipeline Shap-E

```
[Prompt] → [Encoder → Implicit Functions] → [NeRF rendering] → [SDF → Mesh] → OBJ file
```

| Composant | Rôle |
|-----------|------|
| **Implicit functions** | Définissent surfaces/volumes en continu |
| **NeRF** | Construit la scène 3D : coordonnée + direction → densité + couleur RGB |
| **SDF** | Convertit en mesh. Distance : négatif=intérieur, 0=surface, positif=extérieur |

```python
# models.py
from diffusers import ShapEPipeline

def generate_3d_geometry(pipe, prompt, num_inference_steps):
    return pipe(
        prompt, guidance_scale=15.0,
        num_inference_steps=num_inference_steps, output_type="mesh",
    ).images[0]
```

```python
# main.py — avec header Content-Disposition pour forcer le téléchargement
@app.get("/generate/3d", response_class=StreamingResponse)
def serve_3d_controller(prompt: str, num_inference_steps: int = 25):
    model = load_3d_model()
    mesh = generate_3d_geometry(model, prompt, num_inference_steps)
    response = StreamingResponse(mesh_to_obj_buffer(mesh), media_type="model/obj")
    response.headers["Content-Disposition"] = f"attachment; filename={prompt}.obj"
    return response
```

---

## 6. Stratégies de Model Serving

### Stratégie 1 : Model Agnostic (load/unload par requête)

```
Requête → Load model → Inférence → Unload model → Réponse
```

✅ Swap dynamique de modèles | ❌ Très lent, FIFO blocking | **Prototypage uniquement, jamais en prod.**

### Stratégie 2 : ⭐ Preload avec Lifespan (recommandé)

```
App startup → Load model → [Requête 1, 2, 3...] → App shutdown → Unload + cleanup
```

```python
from contextlib import asynccontextmanager

models = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    models["text2image"] = load_image_model()  # startup
    yield                                       # handle requests
    models.clear()                              # shutdown/cleanup

app = FastAPI(lifespan=lifespan)

@app.get("/generate/image")
def serve_image(prompt: str):
    output = generate_image(models["text2image"], prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")
```

✅ Pas de reload, réponses rapides | ❌ RAM/VRAM occupée en permanence

> Legacy : `@app.on_event("startup")` / `@app.on_event("shutdown")` — déprécié.

### Stratégie 3 : Serving externe (FastAPI = couche logique)

FastAPI gère auth, coordination, monitoring. Le modèle tourne **ailleurs**.

**Option A — BentoML (self-hosted)** :

```python
# bento.py
@bentoml.service(resources={"cpu": "4"}, traffic={"timeout": 120})
class Generate:
    def __init__(self):
        self.pipe = load_image_model()

    @bentoml.api(route="/generate/image")
    def generate(self, prompt: str):
        return self.pipe(prompt, num_inference_steps=10).images[0]
```

```python
# main.py (FastAPI = client HTTP)
async def serve_bentoml_controller(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/generate", json={"prompt": prompt})
    return Response(content=response.content, media_type="image/png")
```

**Option B — API providers (OpenAI, Anthropic, etc.)** :

```python
openai_client = OpenAI()

@app.get("/generate/openai/text")
def serve_openai(prompt: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

**Option C — LangChain (abstraction multi-providers)** :

```python
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI

llm = OpenAI()
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

@app.get("/generate/text")
def generate(query: str):
    return llm_chain.run(query)
```

### Tableau comparatif des stratégies

| Stratégie | Quand l'utiliser | Performance |
|-----------|-----------------|-------------|
| **Model Agnostic** | Prototypage, swap de modèles | ❌ Lent |
| **Lifespan Preload** | Production, modèle unique | ✅ Rapide |
| **Externe (BentoML)** | Gros modèles, GPU dédié | ✅ Scalable |
| **Externe (API provider)** | Pas de GPU, budget API ok | ✅ Simple |
| **Externe (LangChain)** | Multi-providers, flexibilité | ✅ Flexible |

> ⚠️ Serving externe = données partagées avec le provider. Data privacy critique → self-host ou cloud managé (Azure OpenAI).

---

## 7. Middleware pour le monitoring

```python
@app.middleware("http")
async def monitor_service(req: Request, call_next) -> Response:
    request_id = uuid4().hex
    start_time = time.perf_counter()

    response = await call_next(req)

    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = request_id

    # Log (en prod → BDD, cf Chapter 7)
    log_to_csv(request_id, req.url, req.client.host, response_time, response.status_code)
    return response
```

| Champ | Source |
|-------|--------|
| Request ID | `uuid4()` |
| Datetime | `datetime.now(UTC)` |
| Endpoint | `req.url` |
| Client IP | `req.client.host` |
| Response time | `perf_counter()` delta |
| Status code | `response.status_code` |

Points importants :
- S'exécute **avant et après** chaque handler → pas besoin de logger dans chaque endpoint
- En prod : persister en BDD (pas CSV, containers éphémères)
- Logger les bodies → attention data privacy et performance

---

## 8. Récap & cheat sheets

### Response patterns FastAPI

| Contenu | Method | Input | Response type | Media type | Lib |
|---------|--------|-------|--------------|------------|-----|
| Texte/JSON | GET | Query params | `return {...}` | `application/json` | — |
| Image | GET | Query params | `Response(bytes)` | `image/png` | Pillow |
| Audio | GET | Query params | `StreamingResponse(buffer)` | `audio/wav` | soundfile |
| Vidéo | POST | File upload | `StreamingResponse(buffer)` | `video/mp4` | av/ffmpeg |
| 3D | GET | Query params | `StreamingResponse(buffer)` | `model/obj` | open3d |

### Dépendances (toutes)

```bash
# Core
uv add "fastapi[standard]" uvicorn openai

# ML/AI
uv add transformers torch diffusers

# Audio/Video/3D
uv add soundfile av open3d python-multipart pillow

# Optionnel
uv add accelerate         # optimise l'usage mémoire CPU
uv add bentoml            # serving externe
uv add langchain langchain-openai  # abstraction multi-providers
uv add streamlit          # UI de prototypage
uv add httpx              # client HTTP async
```

### Points clés du chapitre

1. **Tokenization → Embedding → Positional Encoding → Attention → Prédiction autoregressive** = pipeline complet des LLMs
2. **Stable Diffusion** = encode → noise → denoise guidé par texte → decode
3. **Sora** = Transformer + Diffusion avec 3D U-Net et space-time patches
4. **Shap-E** = fonctions implicites + NeRF + SDF pour la 3D
5. **Lifespan preload** = LE pattern de production pour le model serving
6. **FastAPI comme couche logique** + serving externe (BentoML/API) pour les gros modèles
7. **Middleware** = monitoring centralisé sans toucher aux handlers