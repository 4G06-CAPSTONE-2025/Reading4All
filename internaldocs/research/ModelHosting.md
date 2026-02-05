# Research: Model Hosting / Storage Options

## 1. Hugging Face 
Hugging Face is a widely used ML platform that provides:
- A model & dataset hub that can store model weights/checkpoints, datasets, versioning
- Spaces (host demos / lightweight apps)
- Inference Endpoints (managed deployment of a model behind a production endpoint)

Pricing / product info is documented by Hugging Face.

### How it can be used for alt-text generation

**Typical patterns**
- Store model artifacts on the Hub (public or private repo)
- Deploy either:
    - Spaces (good for demos / prototypes)
    - Inference Endpoints (managed REST endpoint for production-ish usage)

**Inference Endpoints**
- Managed endpoint, billed based on compute/hour and replicas; billed monthly; cost calculated by minute.

**Integration with backend**
- Very easy: backend makes an HTTP request to the Endpoint/Space URL.
- Authentication via HF tokens if private.

### Cost profile 
- Great for min viable product because time-to-deploy is low.
- Can become expensive if you need always-on GPU or higher throughput, because you’re paying for managed compute (similar to cloud, plus managed service overhead).
- HF Endpoints explicitly describe pay-as-you-go hourly compute (including examples of low CPU core pricing and GPU pricing ranges).

### Pros
- Fast setup (especially for demos)
- Built-in model/dataset hosting + versioning
- Managed deployments (Inference Endpoints)

### Cons

- For longer terms, managed endpoints can cost more than self-hosting if usage is steady
- Spaces are excellent for demos, but often not ideal as a primary production inference service (resource limits, scaling behavior)

---
 
**Best fit for**: Research/MVP, internal demo, low-volume production, or if your team wants minimal infra work.

## 2. Amazon EC2 

Amazon EC2 is AWS’s virtual machine service. You can choose:
- CPU instances (cheaper, slower for deep learning inference)
- GPU instances (better latency/throughput for vision models)

### How it can be used for alt-text generation
Common production pattern:
- Put your model code in a service (e.g., FastAPI)
- Run it on an EC2 instance (CPU or GPU)

Expose an endpoint like:

``POST /alt-text (image → JSON text)``

**Integration with backend**

- Also really easy: backend calls your own REST API.
- Can control:
    - request/response format
    - auth (JWT, API keys)
    - rate limiting
    - retries/timeouts

### Cost profile
- Costs depend heavily on region and instance type.
- A commonly referenced GPU inference instance such as g4dn.xlarge is cited around $0.584USD/hour (region: AWS Canada (Central)).
- If always on, it can be hundreds per month

### Pros
- Often cheapest at steady usage 
- Full control over privacy/security + prompt/output formatting
- Easy backend integration (it’s your endpoint)
- Flexible scaling options later

### Cons
- responsible for DevOps tasks such as deployment, updates, monitoring, security patching
- may need basic cloud ops experience

--- 
**Best fit for**: “cost vs control” for a backend-integrated product once you move beyond demo stage.

## 3. Hostinger
Hostinger is a web hosting provider offering shared hosting, cloud hosting, and virtual private server (VPS) plans. Their VPS plans are marketed as developer-friendly and lower cost than typical cloud.

Hostinger’s VPS pricing 
- Their site advertises Kernel based VM VPS plans in the range of about CA$6.99–CA$27.89/month (plan tiers vary).

### How it can be used for alt-text generation
- Host a backend API and/or routing service cheaply on a VPS.
- But: Most VPS plans are *CPU-only* and aren’t designed for GPU inference.
- Hostinger markets “LLM hosting” workflows (such as Ollama templates), which is convenient for some local LLM deployments on VPS. However, for vision models (image → text), CPU-only inference may be too slow depending on the model.

**Integration with backend**
- If you host your backend on Hostinger already, integration is straightforward. (But our system is not on Hostinger)
- But if model inference needs GPU, Hostinger VPS may not be the right layer.

### Pros
- Very low monthly cost for hosting routing/backend logic
- Simple for web apps, dashboards, APIs

### Cons
- Not a standard choice for GPU inference
- Scaling is typically via plan upgrades (less flexible than cloud autoscaling).
- You still manage the server (unmanaged VPS characteristics are common).

---

**Best fit for:**  Hosting the backend layer, not AI model/inference

## 4. Google Gemini API 

Google’s Gemini Developer API provides managed access to Gemini models via API calls. You send input (including images for multimodal) and receive text outputs.

### How it can be used for alt-text generation

- Backend uploads image (or passes image bytes/URL depending on SDK)
- Gemini returns a generated description / alt text

**-Integration with backend**
- Extremely easy:
- API key/token
- HTTP request/SDK call
- No infrastructure, no servers to manage

### Cost profile
Google’s official pricing documentation indicates image input pricing as a per-image tokenized cost (example shown as $0.0011 per image for image input in the pricing docs, subject to model/tier).
(Exact cost depends on model selection and pricing tier.)

### Pros
- Fastest time-to-ship
- Good baseline quality
- No infra/DevOps

### Cons
- Costs scale with usage (per call/per image)
- Less control over model internals, consistent formatting, or long-term cost optimization
- Privacy/security considerations (images leave your system)

--- 

*Best fit as a*: fallback model (look into for rev 1), or when you want the easiest possible integration and accept ongoing API costs.

## Recommendation 

### Comparative Summary 

**Ease of backend integration**

1. Gemini API: easiest (call API)
2. Hugging Face (HF) Inference Endpoints: very easy (call endpoint)
3. EC2: easy once deployed (you own endpoint)
4. Hostinger VPS: easy for backend hosting; less ideal for GPU inference

**Cheapest**

1. EC2 self-host often wins when usage is steady (no per-request premium; pay infra)
2. Hostinger is cheapest for backend/routing, but not typically for GPU inference
3. HF Endpoints can be cost-effective for MVP, may cost more at scale
4. Gemini API can get expensive as requests grow (usage-based)

**Control / privacy**

1. Highest control: EC2 / self-host
2. Medium: HF Endpoints (managed infrastructure, but your model)
3. Lowest: Gemini API (external model)

### Final Recommendation: Amazon EC2 
Self-host the model on Amazon EC2 (GPU if needed)

**Reason:**
- Clean backend integration: just need to expose one REST endpoint
- Strong cost control: paying for a known instance cost (e.g., g4dn family often used for inference) and Mac Students potentially (?) have some credits
    - Even without credits, its cheaper than other alternatives as we don't need to run it for long periods of time 
- No per-image fees, unlike managed APIs
- Full control over:
    - structured output formatting
    - prompt templates for academic diagrams
    - privacy/security policies

**Architecture**
```
Frontend / Backend
        |
        |  POST image
        v
EC2 Instance (FastAPI)
        |
        |  model inference
        v
Alt-text JSON response
```

**Sample Simplified Implementation Steps**

1. Launch an EC2 instance (Ubuntu 22.04, GPU if available)
2. Install Python, PyTorch, and Transformers (already done)
3. Upload the trained model artifacts (our saved models)
4. Load the model using Hugging Face Transformers
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

MODEL_PATH = "/home/ubuntu/model" # this is an example
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

```
5. Expose inference via a lightweight API (i.e. Djangom, FastAPI)

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io, torch

from ai.model_loader import model, processor, device

@csrf_exempt
def generate_alt_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    image = Image.open(io.BytesIO(request.body)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)

    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return JsonResponse({"alt_text": text})


```
6. Call the EC2 endpoint from the backend

```python
import requests

resp = requests.post(
    "http://<EC2_PUBLIC_IP>:8000/generate-alt-text",
    files={"file": open("diagram.png", "rb")}
)

print(resp.json())

```

## Reason for research 
- The group needed to figure out how to host our AI model that is easiest and can seamlessly integrate to our backend, and most effective and less cheap option. 
- Fazmin (AI Specialist from McMaster) recommended these platforms