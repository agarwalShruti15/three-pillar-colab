"""Sign an image with all three pillars of provenance using public SDKs.
Requires: pip install c2pa-python trustmark pillow numpy
Run: python three_pillars_demo.py"""

# # Signing an image with all three pillars of provenance
# 
# Public SDKs only: [`c2pa-python`](https://github.com/contentauth/c2pa-python) (signed manifest) and [`trustmark`](https://github.com/adobe/trustmark) (watermark), plus a small perceptual hash (fingerprint).
# 
# | Pillar | What we do | Why |
# |---|---|---|
# | Metadata | Sign a C2PA manifest with `c2pa-python` that records the watermark ID as a soft binding | Rich, signed history |
# | Watermark | Embed a random ID into the pixels with TrustMark | Survives metadata stripping, exact lookup |
# | Fingerprint | Store a 64-bit perceptual hash in the lookup service | Detects a watermark copied onto another image |
# 
# The lookup service is a local `registry.json` file here. In production it is a cloud service keyed by the watermark ID.
# 
# **Development only:** the signing certificate is the SDK's test certificate, so validators will report the manifest as untrusted. Use your own certificate and a KMS/HSM in production.

import urllib.request, os

# Test certificate and key from the c2pa-python repo (development only) and a sample image from the TrustMark repo.
FIXTURES = "https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/"
for f in ("es256_certs.pem", "es256_private.key"):
    if not os.path.exists(f):
        urllib.request.urlretrieve(FIXTURES + f, f)
if not os.path.exists("input.jpg"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/adobe/trustmark/main/images/ufo_240.jpg", "input.jpg")

import json, random, datetime
import numpy as np
from PIL import Image
from trustmark import TrustMark
from c2pa import Builder, Reader, Signer, Context, C2paSigningAlg, C2paSignerInfo

CERT, KEY = "es256_certs.pem", "es256_private.key"
REGISTRY = "registry.json"        # stands in for a cloud lookup service
TM_ALG = "com.adobe.trustmark.Q"  # soft-binding algorithm name used in TrustMark's C2PA example
SCHEMA = 2                        # TrustMark BCH_4 (see the data schema table in the TrustMark README)
FP_THRESHOLD = 10                 # max differing bits out of 64 for a fingerprint match

tm = TrustMark(verbose=False, model_type="Q", encoding_type=TrustMark.Encoding.BCH_4)

# ## Pillar 2: fingerprint (64-bit perceptual hash)

def _dct_matrix(n=32):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] = np.sqrt(1.0 / n)
    return m

_C = _dct_matrix()

def phash(img):
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = (_C @ g @ _C.T)[:8, :8].flatten()
    return "".join("1" if v else "0" for v in (low > np.median(low[1:])))

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def random_bits(n):
    return "".join(random.choice("01") for _ in range(n))

# ## Sign: watermark, fingerprint, then signed manifest

def load_registry():
    return json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {}

def save_registry(r):
    json.dump(r, open(REGISTRY, "w"), indent=2)

def load_signer():
    info = C2paSignerInfo(
        alg=C2paSigningAlg.ES256,
        sign_cert=open(CERT, "rb").read(),
        private_key=open(KEY, "rb").read(),
        ta_url=b"http://timestamp.digicert.com",
    )
    return Signer.from_info(info)

def manifest_for(wm_id, creator):
    return {
        "claim_generator_info": [{"name": "three_pillars_demo", "version": "0.1"}],
        "format": "image/jpeg",
        "title": "Three pillars demo",
        "assertions": [
            {"label": "c2pa.actions", "data": {"actions": [
                {"action": "c2pa.created",
                 "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCreation"},
                {"action": "c2pa.watermarked"},
            ]}},
            # The watermark ID is the soft binding that lets us find this manifest after metadata is stripped.
            {"label": "c2pa.soft-binding", "data": {
                "alg": TM_ALG,
                "blocks": [{"scope": {}, "value": f"{SCHEMA}*{wm_id}"}],
            }},
            {"label": "stds.schema-org.CreativeWork", "data": {
                "@context": "https://schema.org", "@type": "CreativeWork",
                "author": [{"@type": "Person", "name": creator}],
            }},
        ],
    }

def sign_image(src, out, creator="Demo Creator"):
    cover = Image.open(src).convert("RGB")

    # Pillar 3: watermark. Must happen BEFORE signing because the manifest hard-binds the file bytes.
    wm_id = random_bits(tm.schemaCapacity())
    wm_path = out.replace(".jpg", "_wm.jpg")
    tm.encode(cover, wm_id, MODE="binary").save(wm_path, quality=95)

    # Pillar 2: fingerprint of the pixels we actually saved.
    fp = phash(Image.open(wm_path).convert("RGB"))

    # Pillar 1: signed C2PA manifest with the watermark ID as a soft binding.
    with Context() as ctx, load_signer() as signer, Builder(manifest_for(wm_id, creator), ctx) as b:
        b.sign_file(wm_path, out, signer)

    # Store the manifest and fingerprint under the watermark ID so it can be recovered later.
    with Context() as ctx, Reader(out, context=ctx) as r:
        store = json.loads(r.json())
    reg = load_registry()
    reg[wm_id] = {"manifest_store": store, "fingerprint": fp,
                  "created": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_registry(reg)
    print("Signed:", out)
    print("Watermark ID:", wm_id)
    print("Fingerprint :", fp)
    return wm_id

# ## Verify: use every pillar that survived

def verify(path, label):
    print(f"\n=== {label}: {path}")
    img = Image.open(path).convert("RGB")

    # Pillar 1: is the embedded manifest still there?
    try:
        with Context() as ctx, Reader(path, context=ctx) as r:
            state = json.loads(r.json()).get("validation_state", "unknown")
        print(f"  metadata   : manifest found (validation_state={state})")
    except Exception:
        print("  metadata   : NO manifest (stripped)")

    # Pillar 3: watermark gives an exact lookup key.
    wm_id, present, _ = tm.decode(img, "binary")
    if not present:
        print("  watermark  : none detected")
        return "unknown"
    rec = load_registry().get(wm_id)
    if rec is None:
        print("  watermark  : ID decoded but not in the registry")
        return "unknown"
    print("  watermark  : ID decoded, manifest recovered from the registry")

    # Pillar 2: fingerprint confirms the recovered manifest really describes THIS image.
    d = hamming(phash(img), rec["fingerprint"])
    ok = d <= FP_THRESHOLD
    print(f"  fingerprint: {d}/64 bits differ -> {'MATCH' if ok else 'MISMATCH (possible spoof)'}")
    result = "VERIFIED" if ok else "REJECTED"
    print("  result     :", result)
    return result

# ## Run it

wm_id = sign_image("input.jpg", "signed.jpg", creator="Shruti Agarwal")
verify("signed.jpg", "1. Signed original")

# **Attack 1: a platform strips the metadata and recompresses** (what X.com did to our image).

Image.open("signed.jpg").convert("RGB").save("stripped.jpg", quality=60)
verify("stripped.jpg", "2. Metadata stripped + JPEG q60")

# **Attack 2: resize and re-upload.**

im = Image.open("signed.jpg").convert("RGB")
w, h = im.size
im.resize((int(w * 0.75), int(h * 0.75))).save("social.jpg", quality=70)
verify("social.jpg", "3. Resized to 75% + JPEG q70")

# **Attack 3: spoofing.** Copy the watermark onto an unrelated image. The watermark still decodes, but the fingerprint no longer matches, so the recovered credentials are rejected.

other = Image.effect_mandelbrot((240, 240), (-2.2, -1.5, 0.8, 1.5), 100).convert("RGB")
tm.encode(other, wm_id, MODE="binary").save("spoofed.jpg", quality=95)
verify("spoofed.jpg", "4. Watermark copied onto a different image")
