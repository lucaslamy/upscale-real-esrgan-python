#!/usr/bin/env python3

"""
Upscale_CPU.py : version 100% CPU, sans Vulkan ni GPU, pour environnements sans accès root (ex: o2switch).
Utilise Real-ESRGAN via PyTorch (CPU), compatible images + vidéos, supporte multi-passes et CLI.

Dépendances :
  - torch (CPU)
  - realesrgan
  - opencv-python
  - imageio[ffmpeg]

Installation :
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ pip install --upgrade pip
  $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  $ pip install realesrgan opencv-python imageio[ffmpeg]

Utilisation :
  $ python upscale_cpu.py input.jpg output.png --passes 2 --model RealESRGAN_x4plus
  $ python upscale_cpu.py input.mp4 output.mp4 --passes 4 --model RealESRGAN_x4plus
"""

import warnings
warnings.filterwarnings("ignore", message=".*functional_tensor module is deprecated.*")

import os
import sys
import argparse
import warnings
from pathlib import Path
from time import time

import cv2
import imageio
from realesrgan import RealESRGANer
import torch
from tqdm import tqdm
import numpy as np

# ▶ Supprimer le warning torchvision obsolète
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

# ▶ Choix possibles
PASSES_ALLOWED = [1, 2, 4, 8, 16]
MODELS = ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"]

# ▶ Chemin des modèles (manuels)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATHS = {
    "RealESRGAN_x4plus": MODEL_DIR / "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": MODEL_DIR / "RealESRGAN_x4plus_anime_6B.pth"
}

# ▶ Initialisation du bon modèle (RRDB ou VGG)
def get_model_instance(model_name):
    if model_name == "RealESRGAN_x4plus":
        from basicsr.archs.rrdbnet_arch import RRDBNet
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                               num_feat=64, num_conv=6, upscale=4, act_type='prelu')
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

# ▶ Vérifie si c'est une vidéo
def is_video(path):
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}

# ▶ Upscale image (multi-passes)
def upscale_frame(modeler, frame, passes):
    img = frame
    for pass_num in range(1, passes + 1):
        img, percent = modeler.enhance(img)
        try:
            percent = float(percent)
            tqdm.write(f"   ↪ Pass {pass_num}/{passes} : {percent*100:.1f}%")
        except (ValueError, TypeError):
            tqdm.write(f"   ↪ Pass {pass_num}/{passes}")
    return img

# ▶ Image
def upscale_image(input_path, output_path, model_name, passes):
    print("🖼  Chargement de l'image...")
    device = torch.device("cpu")
    model_path = str(MODEL_PATHS[model_name])
    model_instance = get_model_instance(model_name)

    modeler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model_instance,
        dni_weight=None,
        device=device
    )

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"❌ Erreur : impossible de lire l'image '{input_path}'")
        sys.exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"🔁 Upscaling ({passes} passes)...")
    upscaled = upscale_frame(modeler, image, passes)

    if isinstance(upscaled, np.ndarray):
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), upscaled)
        print(f"✅ Image upscalée sauvegardée : {output_path}")
    else:
        print("❌ Erreur : sortie de l'enhancement invalide.")
        sys.exit(1)

# ▶ Vidéo
def upscale_video(input_path, output_path, model_name, passes):
    print("🎬 Traitement de la vidéo...")
    device = torch.device("cpu")
    model_path = str(MODEL_PATHS[model_name])
    model_instance = get_model_instance(model_name)

    modeler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model_instance,
        dni_weight=None,
        device=device
    )

    reader = imageio.get_reader(str(input_path))
    fps = reader.get_meta_data().get("fps", 25)
    writer = imageio.get_writer(str(output_path), fps=fps)

    total = reader.count_frames()
    print(f"🎞  {total} frames à traiter...")

    for i, frame in enumerate(tqdm(reader, total=total, desc="🔁 Upscaling vidéo")):
        upscaled = upscale_frame(modeler, frame, passes)
        writer.append_data(upscaled)

    writer.close()
    print(f"✅ Vidéo upscalée sauvegardée : {output_path}")

# ▶ Main CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Fichier image ou vidéo à upscale")
    parser.add_argument("output", help="Chemin de sortie")
    parser.add_argument("--passes", type=int, default=1, choices=PASSES_ALLOWED, help="Nombre de passes (1/2/4/8/16)")
    parser.add_argument("--model", default="RealESRGAN_x4plus", choices=MODELS, help="Modèle Real-ESRGAN")
    args = parser.parse_args()

    if not MODEL_PATHS[args.model].exists():
        print(f"❌ Le fichier modèle {MODEL_PATHS[args.model]} est introuvable.")
        print("👉 Télécharge manuellement le modèle ici :")
        print("   https://github.com/xinntao/Real-ESRGAN/blob/master/docs/pretrained_models.md")
        sys.exit(1)

    start = time()
    print("🚀 Lancement de l'upscaling...")

    if is_video(args.input):
        upscale_video(args.input, args.output, args.model, args.passes)
    else:
        upscale_image(args.input, args.output, args.model, args.passes)

    print(f"🕓 Terminé en {round(time() - start, 2)} secondes")

if __name__ == "__main__":
    main()
