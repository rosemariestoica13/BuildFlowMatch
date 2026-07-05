# BuildFlowMatch (waft-a2, backbone dav2)

Repo minimal, extras din [WAFT](.) doar cu ce e nevoie ca modelul `waft-a2`
(dav2 backbone, `BuildFlowMatch`) sa functioneze: incarcare checkpoint,
un pas (sau mai multi) de fine-tuning pe un set de date propriu (poze + gt),
si inferenta/vizualizare inainte-dupa.

## Structura

```
config/a2/dav2/sintel-gm.json   - configul modelului (hyperparametri)
model/                          - arhitectura waft-a2 + backbone dav2
thirdparty/DepthAnythingV2/     - encoder-ul dav2 (vendored, fara submodule)
dataloader/custom.py            - dataset generic pentru poze+gt proprii
criterion/loss.py               - loss-ul folosit la antrenare
finetune_demo.py                - scriptul de fine-tuning + preview
colab_finetune_demo.ipynb       - notebook Colab, cu upload de dataset propriu
colab_finetune_demo_minimal.ipynb - notebook Colab minimal: doar checkpoint,
                                   restul (date exemplu, 1 pas, inferenta) e inclus
sample_data/                    - pereche de imagini exemplu + flux optic cunoscut,
                                   folosita de notebook-ul minimal
checkpoints/                    - pui aici checkpoint-ul de pornire (.pth)
depth-anything-ckpts/           - pui aici greutatile encoder-ului dav2
```

## 1. Checkpoint de pornire

Modelul are nevoie de doua seturi de greutati:

1. **Checkpoint BuildFlowMatch** (waft-a2, dav2) — descarca-l din Google Drive-ul
   proiectului original si pune-l in `checkpoints/sintel-gm-final.pth`
   (sau alt nume, il dai ca `--ckpt` la script).
2. **Encoder dav2 (Depth Anything V2, vits)** — fisierul
   `depth_anything_v2_vits.pth`, pus in `depth-anything-ckpts/`. E acelasi
   fisier folosit de repo-ul principal; poate fi descarcat public de la
   Depth-Anything-V2 (vezi notebook, celula de download) sau incarcat manual.

## 2. Setul tau de date (poze + ground truth)

Creeaza un folder (ex. `data/custom/`) cu structura:

```
data/custom/
  image1/000000.png   000001.png   ...
  image2/000000.png   000001.png   ...
  flow/000000.flo      000001.flo   ...
```

- `image1[i]` + `image2[i]` = o pereche de cadre consecutive.
- `flow[i]` = fluxul optic ground-truth de la `image1[i]` la `image2[i]`,
  in format Middlebury `.flo` (vezi `utils/frame_utils.py`,
  functiile `readFlow` / `writeFlow` daca vrei sa il generezi din Python
  cu `writeFlow(path, flow_array)` unde `flow_array` are shape `(H, W, 2)`).
- Fisierele se potrivesc dupa ordinea alfabetica din fiecare folder, deci
  pastreaza acelasi numar de fisiere si o ordine consistenta in cele trei
  foldere.

## 3. Rulare

Doua variante de notebook Colab:

- `colab_finetune_demo_minimal.ipynb` — cel mai simplu: incarci doar checkpoint-ul,
  restul (o pereche de imagini exemplu din `sample_data/`, 1 pas de fine-tuning,
  inferenta inainte/dupa) e deja inclus in repo.
- `colab_finetune_demo.ipynb` — varianta completa, cu upload al propriului
  set de date (vezi sectiunea 2 mai sus) si numar configurabil de pasi.

Local, echivalentul rularii minimale e:

```bash
pip install -r requirements.txt
python finetune_demo.py \
    --cfg config/a2/dav2/sintel-gm.json \
    --ckpt checkpoints/sintel-gm-final.pth \
    --data_dir sample_data \
    --steps 1 \
    --out_ckpt checkpoints/finetuned.pth
```

(pentru setul tau de date, inlocuieste `--data_dir sample_data` cu `--data_dir data/custom`
si `--steps 1` cu numarul de pasi dorit).

Scriptul salveaza `demo_out/flow_before.jpg` si `demo_out/flow_after.jpg`
(vizualizarea fluxului optic pe prima pereche din setul tau, inainte si
dupa fine-tuning) si checkpoint-ul fine-tuned in `--out_ckpt`.
