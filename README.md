# Facial Emotion Recognition in Children with Autism Spectrum Disorder

**Segmentation-Based CNN and Valence–Arousal Mapping**  
Companion code for the manuscript **“Facial Emotion Recognition in Children with Autism Spectrum Disorder Using Segmentation-Based CNN and Valence-Arousal Mapping”**, Junidar et al., submitted to *Jurnal RESTI*.

## Contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit inference app: upload, face detection/cropping, MediaPipe face-region segmentation, ResNet18 prediction, and Valence–Arousal visualization |
| `resnet18_best.pth` | Supplied ResNet18 model checkpoint; retain its attribution and verify permission to distribute |
| `01_Create_Validation_Set.ipynb` | Research notebook containing dataset preparation and multiple stages including model training (the filename does not capture its complete scope) |
| `3_statistical_test.ipynb` | Notebook for paired comparison of MobileNetV2 and ResNet18 predictions; requires additional model/test data to run |
| `requirements.txt` | Main application dependencies |

## Run the Streamlit application

Use Python 3.10 in a clean environment (versions were documented in the study; installation compatibility may vary by OS).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the command from the repository directory so the app finds `resnet18_best.pth`. The application requires a working installation of the packages in `requirements.txt`; end-to-end execution has **not** been independently verified in this packaging step. PyTorch and TorchVision must be installed as mutually compatible builds for your platform.

## Model and processing

The application uses MediaPipe Face Mesh, crops the face and forms convex-hull masks around the eyes, eyebrows, nose and mouth, followed by dilation and background masking. The segmented image is resized to 224 × 224 and normalized using ImageNet statistics. The loaded ResNet18 predicts **joy** or **sadness**; its predicted class is mapped to **fixed**, not regression-estimated, coordinates:

| Class | Valence | Arousal |
|---|---:|---:|
| Joy | +0.8 | +0.6 |
| Sadness | -0.7 | -0.4 |

The app's plot axes are labeled Valence (horizontal) and Arousal (vertical). This label correction does not modify the model or predictions.

## Dataset and evaluation

The study uses the [Autistic Children Emotions Dataset (Kaggle)](https://www.kaggle.com/datasets/fatmamtalaat/autistic-children-emotions-dr-fatma-m-talaat), selecting 606 images (392 joy, 214 sadness):

| Split | Joy | Sadness | Total |
|---|---:|---:|---:|
| Train | 280 | 160 | 440 |
| Validation | 70 | 40 | 110 |
| Test | 42 | 14 | 56 |

Reported test performance on 56 images: MobileNetV2 accuracy **92.86%**, F1 **92.67%**; ResNet18 accuracy **94.64%**, F1 **94.41%**. Results were reported using an **image-level** split; subject-wise separation was not documented. The reported paired t-test yielded **p = 0.322**. These are results reported by the study, not newly reproduced by this release.

### Reproducing the notebooks

The notebooks were exported with saved execution outputs removed to avoid inadvertently publishing children's photographs or potentially sensitive output artifacts. Their source cells are otherwise retained. **Original local Windows filesystem paths remain in the notebook source and must be adapted** to your own dataset location. Some notebook steps depend on intermediary files or the trained MobileNetV2 checkpoint not included here. This release does not guarantee a turnkey reproduction of every experiment.

## Responsible use and limitations

- No child facial photographs are redistributed here. Obtain images from the original dataset provider and confirm their permissions and consent conditions.
- The model differentiates *emotion categories* within a dataset; it **does not diagnose autism**.
- The image-level test does not establish performance for previously unseen participants or clinical populations.
- Do not expose uploaded children's images to an open public service without appropriate permissions, privacy safeguards and ethical review.

## Citation

Junidar et al., “Facial Emotion Recognition in Children with Autism Spectrum Disorder Using Segmentation-Based CNN and Valence-Arousal Mapping,” *Jurnal RESTI*, manuscript under editorial evaluation. Add the final DOI only after publication.

## License

No open-source license has been selected by the authors. A public repository does not by itself grant permission to reuse or redistribute the code or model weights. Authors should select a license if they intend open reuse.
