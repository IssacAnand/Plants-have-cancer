import json
import re
import os
import random
from pathlib import Path
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR  = Path("./data")
OUTPUT_DIR = Path("./data")
DATASETS   = ["plantdoc", "plantvillage", "plantwild"]

SIM_THRESHOLD     = 0.78   # drop captions more similar than this
MAX_PER_CLASS     = 24     # final cap per class
RANDOM_SEED       = 42

random.seed(RANDOM_SEED)

# ── Synonym tables (hardcoded — no WordNet) ───────────────────────────────────

SYNONYMS: dict[str, list[str]] = {
    # appearance verbs
    "appears as":       ["presents as", "manifests as", "is characterized by", "shows", "exhibits", "displays"],
    "characterized by": ["marked by", "distinguished by", "identified by", "defined by"],
    "shows":            ["displays", "exhibits", "presents", "reveals", "features"],
    # colour
    "dark":             ["deep", "dusky", "darkened", "blackish"],
    "brown":            ["tan", "russet", "tawny", "ochre", "brownish"],
    "yellow":           ["pale yellow", "golden-yellow", "yellowish", "chlorotic"],
    "olive-green":      ["olive green", "dull green", "greyish-green"],
    "black":            ["jet-black", "blackened", "sooty", "charcoal"],
    "white":            ["whitish", "chalky", "pale", "ivory"],
    "gray":             ["grey", "ash-grey", "silvery-grey"],
    "orange":           ["orange-red", "rust-coloured", "ochre"],
    # texture
    "velvety":          ["fuzzy", "powdery", "downy", "soft-textured"],
    "scaly":            ["corky", "crusty", "flaky", "rough-textured"],
    "rough":            ["coarse", "uneven", "irregular"],
    "powdery":          ["dusty", "chalky", "flour-like", "mealy"],
    "sunken":           ["depressed", "pitted", "concave", "collapsed"],
    "raised":           ["elevated", "protruding", "swollen", "pustular"],
    "lesion":           ["spot", "patch", "blemish", "mark", "blotch"],
    "lesions":          ["spots", "patches", "blemishes", "blotches", "marks"],
    "spots":            ["lesions", "patches", "blotches", "blemishes"],
    # shape
    "circular":         ["round", "disc-shaped", "oval", "rounded"],
    "irregular":        ["angular", "asymmetric", "variable", "ragged-edged"],
    "small":            ["tiny", "minute", "pinpoint", "minor"],
    "large":            ["extensive", "broad", "expansive", "wide"],
    # effects
    "yellowing":        ["chlorosis", "chlorotic discolouration", "leaf yellowing"],
    "defoliation":      ["premature leaf drop", "leaf shed", "leaf loss"],
    "wilting":          ["drooping", "flagging", "collapse of tissue"],
    "distortion":       ["leaf curl", "puckering", "malformation", "warping"],
    "necrosis":         ["tissue death", "cell death", "necrotic breakdown"],
    "blight":           ["blighting", "rapid tissue death", "scorching"],
    # location
    "upper surface":    ["adaxial surface", "top surface", "leaf face"],
    "lower surface":    ["underside", "abaxial surface", "leaf underside"],
    "leaf margin":      ["leaf edge", "leaf border", "margin"],
    "leaf tip":         ["apical region", "leaf apex", "tip of the leaf"],
}

def apply_synonyms(text: str, n_swaps: int = 2) -> str:
    """Randomly replace up to n_swaps phrases with synonyms."""
    words = list(SYNONYMS.keys())
    random.shuffle(words)
    swaps = 0
    for phrase in words:
        if swaps >= n_swaps:
            break
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(text):
            replacement = random.choice(SYNONYMS[phrase])
            text = pattern.sub(replacement, text, count=1)
            swaps += 1
    return text


# ── Descriptor extraction ─────────────────────────────────────────────────────

COLOR_PAT    = r'\b(olive[- ]green|olive green|dark green|dark|brown|yellow|black|orange|red|white|gray|grey|pale|tan|russet|ochre|chlorotic|purple|violet)\b'
TEXTURE_PAT  = r'\b(velvety|scaly|rough|powdery|watery|corky|greasy|necrotic|sunken|raised|pustular|crusty|fuzzy|downy|slimy)\b'
SHAPE_PAT    = r'\b(circular|round|oval|irregular|angular|elongated|small|large|tiny|minute|concentric|ring-like|water-soaked)\b'
LOCATION_PAT = r'\b(upper surface|lower surface|leaf margin|leaf tip|midrib|vein|edge|underside|upper|lower|both surfaces|stem|fruit|petiole)\b'
EFFECT_PAT   = r'\b(distortion|defoliation|wilting|yellowing|chlorosis|necrosis|blight|rot|drop|curl|shrivel|stunting|leaf loss|dieback)\b'
SEVERITY_PAT = r'\b(mild|severe|light|heavy|moderate|extensive|widespread|localised|isolated|scattered|dense|profuse)\b'
STAGE_PAT    = r'\b(early|initial|later|advanced|mature|young|developing|progressing|late-stage|early-stage)\b'

def extract_descriptors(captions: list[str]) -> dict:
    text = " ".join(captions).lower()
    def unique(pat): return list(dict.fromkeys(re.findall(pat, text)))
    return {
        "colors":    unique(COLOR_PAT),
        "textures":  unique(TEXTURE_PAT),
        "shapes":    unique(SHAPE_PAT),
        "locations": unique(LOCATION_PAT),
        "effects":   unique(EFFECT_PAT),
        "severity":  unique(SEVERITY_PAT),
        "stages":    unique(STAGE_PAT),
    }

def pick(lst: list, fallback: str = "") -> str:
    return lst[0] if lst else fallback

def pick2(lst: list, fallback1="", fallback2="") -> tuple:
    if len(lst) >= 2: return lst[0], lst[1]
    if len(lst) == 1: return lst[0], fallback2
    return fallback1, fallback2


# ── Template banks ────────────────────────────────────────────────────────────
# Each returns a list[str] of new captions.
# Templates are deliberately varied so synonym-expansion produces further diversity.

def gen_progression(label: str, d: dict) -> list[str]:
    colour  = pick(d["colors"], "dark")
    texture = pick(d["textures"], "irregular")
    effect  = pick(d["effects"], "yellowing")
    effect2 = d["effects"][1] if len(d["effects"]) > 1 else "defoliation"
    shape   = pick(d["shapes"], "circular")
    return [
        f"In the early stage of {label}, {shape} water-soaked spots first appear on the leaf surface before becoming {colour} and {texture}.",
        f"As {label} progresses, the initial pale lesions expand and darken, eventually leading to {effect} and severe tissue damage.",
        f"Early {label} infection shows faint, discoloured flecks that are easy to miss, while mature lesions become prominently {colour} with a {texture} surface.",
        f"At the advanced stage of {label}, lesions coalesce across large portions of the leaf, causing {effect} and {effect2}.",
        f"Young {label} lesions are soft and water-soaked; as they age they develop a {colour}, {texture} appearance and well-defined margins.",
    ]

def gen_severity(label: str, d: dict) -> list[str]:
    colour  = pick(d["colors"], "dark")
    texture = pick(d["textures"], "lesion")
    effect  = pick(d["effects"], "defoliation")
    loc     = pick(d["locations"], "leaf surface")
    return [
        f"A mild {label} infection appears as a few isolated {colour} lesions on the {loc} with no significant tissue damage.",
        f"In severe {label} cases, lesions merge and cover most of the {loc}, causing extensive {effect} and plant stress.",
        f"Light {label} infections are easily overlooked due to their small size and limited spread across the {loc}.",
        f"Heavy {label} infections result in a dense covering of {colour}, {texture} lesions that significantly reduce photosynthetic area.",
        f"Moderate {label} shows scattered lesions of varying sizes on the {loc}, with some surrounding chlorosis but limited spread.",
    ]

def gen_differential(label: str, d: dict) -> list[str]:
    colour  = pick(d["colors"], "dark")
    texture = pick(d["textures"], "irregular")
    loc     = pick(d["locations"], "leaf surface")
    loc2    = d["locations"][1] if len(d["locations"]) > 1 else "underside"
    effect  = pick(d["effects"], "necrosis")
    return [
        f"Unlike fungal mildew which produces white fluffy growth, {label} produces {colour}, {texture} lesions firmly embedded in leaf tissue.",
        f"{label} can be distinguished from bacterial leaf spots by its {texture} surface texture and gradual {colour} discolouration rather than water-soaked margins.",
        f"A key identifying feature of {label} is the {colour} colouration on the {loc}, which differs from similar diseases that primarily affect the {loc2}.",
        f"While other leaf diseases cause uniform discolouration, {label} produces distinctly {texture} lesions with clearly defined margins.",
        f"{label} is identifiable by its characteristic {colour}, {texture} appearance, distinguishing it from nutrient deficiencies which cause more uniform yellowing.",
    ]

def gen_plant_context(label: str, d: dict) -> list[str]:
    loc   = pick(d["locations"], "leaf surface")
    loc2  = d["locations"][1] if len(d["locations"]) > 1 else "stem"
    effect = pick(d["effects"], "defoliation")
    colour = pick(d["colors"], "dark")
    return [
        f"In a {label}-infected plant, the lower canopy leaves show the most severe symptoms while upper younger leaves may still appear healthy.",
        f"{label} affects the {loc} primarily, though heavily infected plants also show secondary symptoms on the {loc2} and neighbouring tissue.",
        f"Surrounding healthy tissue next to {label} lesions remains firm and green, creating a sharp visual contrast with the {colour} infected areas.",
        f"A plant with advanced {label} shows uneven symptom distribution, with dense lesion clusters near the {loc} gradually thinning toward the leaf edge.",
        f"The overall canopy of a {label}-infected plant looks sparse and patchy due to progressive {effect} from the base upward.",
    ]

def gen_environmental(label: str, d: dict) -> list[str]:
    colour  = pick(d["colors"], "dark")
    texture = pick(d["textures"], "lesion")
    effect  = pick(d["effects"], "defoliation")
    return [
        f"Under prolonged wet and humid conditions, {label} lesions proliferate rapidly and the {texture} surface appears more prominent.",
        f"In dry conditions, {label} lesions appear desiccated and cracked compared to the softer, more {colour} appearance seen in moist weather.",
        f"After periods of rain and warm temperatures, {label} spreads quickly across the canopy, producing dense patches of {colour} discolouration.",
        f"Cool, humid nights followed by warm days create ideal conditions for {label}, resulting in rapid expansion of lesions and increased {effect}.",
        f"During drought stress, {label} lesions appear more sunken and darkened, whereas in moist conditions the margins remain softer and water-soaked.",
    ]

def gen_healthy_contrast(label: str, d: dict) -> list[str]:
    colour  = pick(d["colors"], "dark")
    texture = pick(d["textures"], "rough")
    effect  = pick(d["effects"], "yellowing")
    loc     = pick(d["locations"], "leaf surface")
    return [
        f"Healthy leaves show a uniform bright green colour and smooth surface, while {label}-infected tissue shows irregular {colour} patches and {texture} texture.",
        f"Next to a {label} lesion, unaffected tissue is firm, glossy, and deep green, making the infected area's {colour} and {texture} appearance easy to identify.",
        f"A healthy leaf has consistent venation and colour throughout, whereas {label} disrupts this uniformity with {colour} spots and {effect} around lesion edges.",
        f"The boundary between healthy and {label}-infected tissue on the {loc} is often marked by a distinct yellow or chlorotic halo.",
        f"Healthy tissue near {label} lesions shows no structural changes, while the infected area exhibits {texture} surface changes and {colour} discolouration.",
    ]


PERSPECTIVE_GENERATORS = [
    ("progression",  gen_progression),
    ("severity",     gen_severity),
    ("differential", gen_differential),
    ("plant_context",gen_plant_context),
    ("environmental",gen_environmental),
    ("healthy",      gen_healthy_contrast),
]

# ── Perspective-matched CLIP prefixes ─────────────────────────────────────────

CLIP_PREFIXES = {
    "visual":       ["a photo of {label}: {cap}",
                     "a close-up photo of {label}: {cap}",
                     "an image showing {label}: {cap}",
                     "a plant disease photograph of {label}: {cap}",
                     "a macro photo of {label}: {cap}"],
    "progression":  ["a photo showing the progression of {label}: {cap}",
                     "an image of {label} at an early stage: {cap}",
                     "a photo of advanced {label}: {cap}"],
    "severity":     ["a photo of mild {label}: {cap}",
                     "a photo of severe {label}: {cap}",
                     "an image showing {label} severity: {cap}"],
    "differential": ["a plant disease identification photo of {label}: {cap}",
                     "a diagnostic photo of {label}: {cap}"],
    "plant_context":["a wide-view photo of {label} on a plant: {cap}",
                     "a whole-plant photo showing {label}: {cap}"],
    "environmental":["a photo of {label} after wet weather: {cap}",
                     "an image of {label} under humid conditions: {cap}"],
    "healthy":      ["a comparison photo of healthy vs {label}-infected tissue: {cap}",
                     "a side-by-side image of {label} and healthy leaf: {cap}"],
}

def wrap_prefix(perspective: str, label: str, caption: str, index: int = 0) -> str:
    templates = CLIP_PREFIXES.get(perspective, CLIP_PREFIXES["visual"])
    tmpl = templates[index % len(templates)]
    cap = caption[0].lower() + caption[1:] if caption else caption
    return tmpl.format(label=label, cap=cap)


# ── Fragment recombination ────────────────────────────────────────────────────

def extract_fragments(captions: list[str]) -> list[str]:
    """
    Pull noun-phrase-like fragments (descriptor + noun combos) from captions.
    These are recombined to form hybrid captions without repeating full sentences.
    """
    fragments = []
    patterns = [
        r'(?:dark|olive[- ]green|brown|yellow|black|orange|white|gray|pale)[^,.;]{5,40}(?:lesions?|spots?|patches?|areas?|tissue)',
        r'(?:velvety|scaly|rough|powdery|sunken|raised|necrotic)[^,.;]{5,35}(?:surface|texture|appearance|growth)',
        r'(?:surrounded by|accompanied by|leading to|causing)[^,.;]{5,45}',
        r'(?:on the|along the|near the)\s+(?:upper|lower|leaf)?[^,.;]{5,35}(?:surface|margin|tip|edge|midrib|vein)',
    ]
    for cap in captions:
        for pat in patterns:
            matches = re.findall(pat, cap, re.IGNORECASE)
            fragments.extend(m.strip() for m in matches if len(m.split()) >= 3)
    return list(dict.fromkeys(fragments))  # deduplicate


def recombine_fragments(label: str, fragments: list[str], n: int = 4) -> list[str]:
    if len(fragments) < 2:
        return []
    results = []
    random.shuffle(fragments)
    for i in range(min(n, len(fragments) - 1)):
        f1, f2 = fragments[i], fragments[i + 1]
        # Don't combine fragments that are too similar in wording
        shared = set(f1.lower().split()) & set(f2.lower().split())
        if len(shared) > 4:
            continue
        cap = f"{label} shows {f1.lower()}, {f2.lower()}."
        results.append(cap)
    return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(captions: list[str], threshold: float = SIM_THRESHOLD) -> list[str]:
    if len(captions) <= 1:
        return captions
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    try:
        tfidf = vec.fit_transform(captions)
    except ValueError:
        return captions
    kept = [0]
    for i in range(1, len(captions)):
        sims = cosine_similarity(tfidf[i], tfidf[kept]).flatten()
        if sims.max() < threshold:
            kept.append(i)
    return [captions[i] for i in kept]


# ── Label utilities ───────────────────────────────────────────────────────────

def normalise_label(raw: str) -> str:
    cleaned = raw.replace("___", " ").replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).strip()

def strip_clip_prefix(caption: str) -> str:
    """Remove the 'a photo of X: ' prefix to get the raw description."""
    m = re.search(r":\s+(.+)$", caption)
    return m.group(1) if m else caption


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_class(raw_label: str, existing_captions: list[str]) -> list[str]:
    label = normalise_label(raw_label)

    # 1. Strip CLIP prefixes to get raw descriptions
    raw_desc = [strip_clip_prefix(c) for c in existing_captions]

    # 2. Extract descriptors
    d = extract_descriptors(raw_desc)

    # 3. Keep existing visual-appearance captions (already CLIP-prefixed)
    visual_caps = list(existing_captions)

    # 4. Generate perspective captions
    new_caps: list[str] = []
    for perspective, generator in PERSPECTIVE_GENERATORS:
        templates = generator(label, d)
        for idx, cap in enumerate(templates):
            # Synonym-expand odd-indexed captions for variety
            if idx % 2 == 1:
                cap = apply_synonyms(cap, n_swaps=2)
            wrapped = wrap_prefix(perspective, label, cap, index=idx)
            new_caps.append(wrapped)

    # 5. Fragment recombination (hybrid captions from within-class phrases)
    fragments = extract_fragments(raw_desc)
    hybrid_caps = recombine_fragments(label, fragments, n=4)
    for idx, cap in enumerate(hybrid_caps):
        wrapped = wrap_prefix("visual", label, cap, index=idx)
        new_caps.append(wrapped)

    # 6. Combine and deduplicate
    all_caps = visual_caps + new_caps
    deduped  = deduplicate(all_caps, SIM_THRESHOLD)

    # 7. Cap at MAX_PER_CLASS
    return deduped[:MAX_PER_CLASS]


def process_dataset(name: str):
    src_clip = INPUT_DIR / f"{name}_clip.json"
    src_raw  = INPUT_DIR / f"{name}.json"
    src      = src_clip if src_clip.exists() else src_raw
    dest     = OUTPUT_DIR / f"{name}_clean.json"

    with open(src) as f:
        data: dict[str, list[str]] = json.load(f)

    output      = {}
    total_before = 0
    total_after  = 0

    for raw_label, captions in data.items():
        total_before += len(captions)
        result = process_class(raw_label, captions)
        output[normalise_label(raw_label)] = result
        total_after += len(result)

    with open(dest, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[{name}]  {total_before} → {total_after} captions  ({total_after // len(data)}/class avg)  → {dest}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ds in DATASETS:
        if not ((INPUT_DIR / f"{ds}_clip.json").exists() or (INPUT_DIR / f"{ds}.json").exists()):
            print(f"[SKIP] {ds} — no source JSON found")
            continue
        process_dataset(ds)
    print("\nDone.")