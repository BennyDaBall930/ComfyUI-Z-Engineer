"""Generate the example Z-Image Turbo + Z-Engineer workflow JSON."""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "zengineer"))
import prompt_utils  # noqa: E402

SEED_PROMPT = "a red panda barista pouring latte art in a cozy alpine cafe at golden hour"


def node(nid, ntype, pos, size, inputs, outputs, widgets, order=0):
    return {
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }


def out(name, otype, links):
    return {"name": name, "type": otype, "links": links}


def inp(name, itype, link, widget=None):
    d = {"name": name, "type": itype, "link": link}
    if widget:
        d["widget"] = {"name": widget}
    return d


nodes = [
    node(1, "ZEngineerCLIPLoaderGGUF", [-620, 160], [420, 110], [],
         [out("clip", "CLIP", [1, 2])],
         ["Z-Image-Engineer-V6-Q4_K_M.gguf", "default"], order=0),
    node(2, "ZEngineerEnhance", [-140, 160], [460, 600],
         [inp("clip", "CLIP", 1)],
         [out("prompt", "STRING", [3])],
         [SEED_PROMPT, prompt_utils.V6_SYSTEM_PROMPT, 6606, "fixed", 0.20, 0.9, 40,
          0.03, 1.05, 320, True, True, True, False, "\\n---\\n"], order=4),
    node(3, "CLIPTextEncode", [380, 160], [300, 140],
         [inp("clip", "CLIP", 2), inp("text", "STRING", 3, widget="text")],
         [out("CONDITIONING", "CONDITIONING", [4, 5])],
         [""], order=5),
    node(4, "ConditioningZeroOut", [380, 360], [300, 60],
         [inp("conditioning", "CONDITIONING", 5)],
         [out("CONDITIONING", "CONDITIONING", [6])],
         [], order=6),
    node(5, "UNETLoader", [-620, -40], [420, 110], [],
         [out("MODEL", "MODEL", [7])],
         ["z_image_turbo_bf16.safetensors", "default"], order=1),
    node(6, "ModelSamplingAuraFlow", [-140, -40], [300, 80],
         [inp("model", "MODEL", 7)],
         [out("MODEL", "MODEL", [8])],
         [3], order=3),
    node(7, "EmptySD3LatentImage", [380, 480], [300, 120], [],
         [out("LATENT", "LATENT", [9])],
         [1024, 1024, 1], order=2),
    node(8, "KSampler", [740, 160], [300, 280],
         [inp("model", "MODEL", 8), inp("positive", "CONDITIONING", 4),
          inp("negative", "CONDITIONING", 6), inp("latent_image", "LATENT", 9)],
         [out("LATENT", "LATENT", [10])],
         [0, "randomize", 8, 1, "res_multistep", "simple", 1], order=7),
    node(9, "VAELoader", [740, 500], [300, 60], [],
         [out("VAE", "VAE", [11])],
         ["ae.safetensors"], order=8),
    node(10, "VAEDecode", [1090, 160], [200, 80],
         [inp("samples", "LATENT", 10), inp("vae", "VAE", 11)],
         [out("IMAGE", "IMAGE", [12])],
         [], order=9),
    node(11, "SaveImage", [1340, 160], [400, 450],
         [inp("images", "IMAGE", 12)],
         [],
         ["z-image-zengineer"], order=10),
]

links = [
    [1, 1, 0, 2, 0, "CLIP"],
    [2, 1, 0, 3, 0, "CLIP"],
    [3, 2, 0, 3, 1, "STRING"],
    [4, 3, 0, 8, 1, "CONDITIONING"],
    [5, 3, 0, 4, 0, "CONDITIONING"],
    [6, 4, 0, 8, 2, "CONDITIONING"],
    [7, 5, 0, 6, 0, "MODEL"],
    [8, 6, 0, 8, 0, "MODEL"],
    [9, 7, 0, 8, 3, "LATENT"],
    [10, 8, 0, 10, 0, "LATENT"],
    [11, 9, 0, 10, 1, "VAE"],
    [12, 10, 0, 11, 0, "IMAGE"],
]

workflow = {
    "last_node_id": 11,
    "last_link_id": 12,
    "nodes": nodes,
    "links": links,
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}

targets = sys.argv[1:] or [os.path.join(REPO, "example_workflows", "z_image_turbo_z_engineer.json")]
for target in targets:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=2)
    print("wrote", target)
