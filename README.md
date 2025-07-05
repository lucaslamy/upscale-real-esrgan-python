# 🔍 Real-ESRGAN CPU Upscaler (PyTorch-only)

> 100% CPU-based Real-ESRGAN image & video upscaler — no GPU, no Vulkan, no root access required.

Ideal for constrained environments (e.g. shared hosting, o2switch, headless Linux servers).

---

## 🚀 Features

- Fully CPU (PyTorch backend, no NCNN or Vulkan)
- Upscales **images** and **videos** (multi-pass)
- Compatible with models:
  - `RealESRGAN_x4plus`
  - `RealESRGAN_x4plus_anime_6B`
- No root or `sudo` needed
- Shows progress with each frame

---

## 🧪 Tested on

- Python 3.11
- Torch 2.x (CPU-only)
- Ubuntu / Debian / Alpine (headless)

---

## 🛠 Installation (no root)

```bash
# 1. Clone repo (or copy upscale.py)
git clone https://github.com/yourname/upscale-cpu && cd upscale-cpu

# 2. Create virtualenv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ Dependencies are pinned to CPU versions only.

---

## 📦 requirements.txt

```txt
torch==2.3.0+cpu
torchvision==0.18.0+cpu
torchaudio==2.3.0+cpu
realesrgan
opencv-python
imageio[ffmpeg]
tqdm
```

---

## 📁 Download Real-ESRGAN model files

If you want, you can change the models used.

Create a `weights/` folder, then download these files:

| Model | Description | Link |
|-------|-------------|------|
| `RealESRGAN_x4plus.pth` | General model (photos/videos) | [🔗 Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth) |
| `RealESRGAN_x4plus_anime_6B.pth` | Best for anime/illustration | [🔗 Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus_anime_6B.pth) |

Or use:

```bash
mkdir -p weights
wget -O weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth
wget -O weights/RealESRGAN_x4plus_anime_6B.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus_anime_6B.pth
```

---

## 📸 Example usage

### Image upscale

```bash
python upscale.py input.jpg output.png --passes 2 --model RealESRGAN_x4plus
```

### Video upscale

```bash
python upscale.py input.mp4 output.mp4 --passes 4 --model RealESRGAN_x4plus
```

---

## 🧠 How it works

- Loads image/video using OpenCV or `imageio`
- Applies RealESRGAN upscale model for the desired number of passes
- Each pass resizes the image progressively (e.g., 4x → 16x after 2 passes)
- Video is processed frame-by-frame with a progress bar (`tqdm`)
- Works entirely on CPU

---

## 🛠️ Alias Shell - Lancement en tâche de fond

Ces deux alias permettent de :

- `upscale-run`: **lancer l’upscaling en tâche de fond** avec `screen` et journaliser la sortie.
- `upscale-env`: **activer rapidement l’environnement virtuel Python** pour Real-ESRGAN.

---

### ✅ Prérequis

- Dossier du projet : `/root/upscale`
- Environnement virtuel activable : `/root/upscale/venv/bin/activate`
- Script Python : `/root/upscale/upscale.py`
- `screen` est installé (`apt install screen` si nécessaire)

---

### 🧩 Configuration des alias

Ajoutez ceci à votre `~/.bashrc`, `~/.zshrc`, ou `/root/.bashrc` :

```bash
upscale-run() {
  LOGFILE=/root/upscale/upscale.log
  screen -S upscale -dm bash -c "source /root/upscale/venv/bin/activate && python /root/upscale/upscale.py $1 $2 --passes $3 --model RealESRGAN_x4plus >> \$LOGFILE 2>&1"
}

upscale-env() {
  cd /root/upscale
  source /root/upscale/venv/bin/activate 
}
```

Ensuite, rechargez le shell :

```bash
source ~/.bashrc
# ou
source ~/.zshrc
```

---

### 🚀 Utilisation de `upscale-run`

```bash
upscale-run <input_path> <output_path> <passes>
```

#### Exemple :

```bash
upscale-run input/photo.jpg output/photo_upscaled.jpg 2
```

Cela va :

- lancer un processus détaché dans un screen nommé `upscale`
- activer l’environnement virtuel
- exécuter le script avec les paramètres donnés
- écrire les logs dans `/root/upscale/upscale.log`

#### 📖 Lire les logs

```bash
tail -f /root/upscale/upscale.log
```

#### 🖥 Rejoindre la session `screen`

```bash
screen -r upscale
```

#### ⏎ Détacher du screen

Appuyez sur : `Ctrl + A`, puis `D`

---

### 🧪 Utilisation de `upscale-env`

Permet de naviguer dans le dossier projet et d’activer l’environnement virtuel :

```bash
upscale-env
```

---

### 🧹 Gestion des sessions `screen`

```bash
screen -ls                   # Liste les sessions actives
screen -S upscale -X quit    # Ferme la session "upscale"
```


---

## ❓ FAQ

### ❌ My image doesn't get larger?

Make sure you're using the **correct model** and check the `--passes` value. Each pass multiplies resolution by ~4x. Use:

```bash
--passes 2  # for 16x upscale
```

---

### 💬 Warning: `torchvision.transforms.functional_tensor is deprecated`

This warning is automatically filtered in `upscale.py`. You can ignore it.

---

## ✅ Credits

- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
