# Preserving Archival Heritage: Spanish Manuscript to English Pipeline

CSEC.303 Midterm Project, Phase 1 prototype (Ateneo de Naga University).
Authors: Francine Angela Elopre, Dominique Valencia.

The prototype takes a scanned Philippine-American War page and runs three stages:

1. **Preprocessing** (OpenCV): grayscale, denoise, lighting normalization, CLAHE, deskew, thresholding, cleanup.
2. **Text recognition**: line segmentation, then Tesseract (OCR) and TrOCR (HTR), scored with CER/WER.
3. **Translation**: Marian `Helsinki-NLP/opus-mt-es-en`, with and without spelling modernization, scored with BLEU/chrF.

Each stage is one Jupyter notebook, and each reads the files the previous one saved.

## Folder layout

```
Project Midterms/
├── data/
│   ├── 735771.png                  # input scan
│   └── ground_truth/sample.txt     # manual transcription, one line per manuscript line
├── notebook/
│   ├── preprocessing.ipynb         # Stage 1
│   ├── text_recognitionv3.ipynb    # Stage 2 (the version that produced the reported results)
│   └── translation.ipynb           # Stage 3
└── outputs/                        # created automatically
    ├── processed_images/
    ├── recognition/
    └── translation/
```

`text_recognition.ipynb` and `text_recognitionv2.ipynb` are earlier versions; use **v3**.
Run the notebooks from either the project root or `notebook/`; they find `data/` and `outputs/` on their own.

## Requirements

### Programs
| Program | Notes |
|---|---|
| Python 3.10+ | With Jupyter (`pip install notebook`). |
| **Tesseract OCR 5.x** with the **Spanish** data (`spa`) | A separate program, not a pip package. Results here used 5.3.4. |
| GPU (optional) | TrOCR ran on CUDA in the reported results. It also works on CPU, only slower. |

Install Tesseract with Spanish data:
- **Windows:** UB Mannheim installer; tick *Spanish* under additional language data. If Python cannot find it, uncomment the `tesseract_cmd` line in Step 1 of the recognition notebook.
- **macOS:** `brew install tesseract tesseract-lang`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr tesseract-ocr-spa`

Check it with `tesseract --list-langs`; `spa` must be listed.

### Python libraries
| Library | Used for | Stage |
|---|---|---|
| `opencv-python` | all image processing (results used OpenCV 5.0.0) | 1, 2 |
| `numpy` | image arrays | 1, 2, 3 |
| `matplotlib` | figures | 1, 2, 3 |
| `pytesseract` | Python wrapper for Tesseract | 1 (optional check), 2 |
| `pillow` | line cropping | 2 |
| `pandas` | result tables and CSVs | 2, 3 |
| `jiwer` | CER and WER | 2 |
| `torch`, `transformers` | TrOCR and Marian models | 2, 3 |
| `sentencepiece`, `sacremoses` | Marian tokenizer | 3 |
| `sacrebleu` | BLEU and chrF | 3 |

One-line install:

```bash
pip install opencv-python numpy matplotlib pillow pandas pytesseract jiwer \
            torch transformers sentencepiece sacremoses sacrebleu notebook
```

The notebooks also contain `%pip install` cells that do this for you.

### Internet and disk
Internet is needed **once**, to download the models: TrOCR `microsoft/trocr-base-handwritten` (about 1.3 GB) and Marian `Helsinki-NLP/opus-mt-es-en` (about 300 MB).

## Step-by-step

### Step 1: `preprocessing.ipynb`
1. Make sure `data/735771.png` exists.
2. Run all cells in order.
3. It saves 9 images to `outputs/processed_images/`:
   `01_gray`, `02_denoised`, `03_lighting_normalized`, `04_clahe`, `05_deskewed`, `06a_threshold_global`, `06b_threshold_otsu`, `06c_threshold_adaptive`, `07_final`.
4. The next stage needs `05_deskewed.png` (grayscale version) and `07_final.png` (binarized version).

The optional OCR check near the end is skipped automatically if Tesseract is not installed.

### Step 2: `text_recognitionv3.ipynb`
1. Confirm Step 1 outputs exist, and that `data/ground_truth/sample.txt` exists (22 lines, one per manuscript line). Without it the notebook still produces text but cannot score it.
2. Run all cells in order. It will:
   - segment the page into text lines (Step 3 of the notebook) and print how many lines it found;
   - run Tesseract on the full page and per line (`--psm 6` and `--psm 7`) on both image versions;
   - run TrOCR per line on both versions;
   - score everything against `sample.txt` (strict and relaxed CER/WER).
3. Outputs in `outputs/recognition/`:
   - `results.csv`, `per_line_binarized.csv`
   - `tesseract_*__*.txt`, `trocr_per_line__*.txt` (one file per engine and image version)
   - `lines/line_NN.png` (cropped line images)
   - `line_segmentation.png`, `line_examples.png`, `comparison.png`
4. **Check the line count.** On the test page it detects **28 lines for 22 real ones** (punch holes and paper damage are read as lines). When the counts differ, the notebook skips the per-line comparison but still gives page-level scores. Tuning options are in the notebook's Step 1 settings (`MIN_INK_RATIO`, `SMEAR_WIDTH_FACTOR`, `MANUAL_BOXES`).
5. `ground_truth_TEMPLATE.txt` is only written when no ground truth is found, and it is **not** a transcription: it is Tesseract's guess for you to correct.

### Step 3: `translation.ipynb`
1. Confirm the `.txt` files from Step 2 exist in `outputs/recognition/`.
2. Run all cells in order. Each source (the perfect transcription plus the four recognized texts) is translated twice: modernization off and on.
3. Outputs in `outputs/translation/`:
   - one `.txt` per source and setting (`..._modernized_off.txt`, `..._modernized_on.txt`)
   - `results.csv` (BLEU, chrF, time), `translations.csv`, `comparison.png`

## Known issues to check before trusting results

- **Translation reference.** In Step 1 of `translation.ipynb`, `REFERENCE_PATH` points to `data/ground_truth/sample.txt`, which is the *Spanish* transcription. BLEU and chrF are therefore computed against Spanish and are not valid translation scores. To fix this, write a human English translation (for example `data/ground_truth/sample_en.txt`) and point `REFERENCE_PATH` at it.
- **Modernization table.** The `MODERNIZE` dictionary is only a starter list, and it changed nothing on the test page. Extend it with the period spellings found in your own pages.
- **Long passages.** Translating one long passage makes Marian drop lines. Splitting into sentences is recommended (see the report).
- **Typewritten test page.** The sample is typewritten, so TrOCR (trained on English handwriting) is at a disadvantage. Test on handwritten pages before generalizing.
- **Different images.** To use another scan, change `IMAGE_PATH` in `preprocessing.ipynb` and replace `data/ground_truth/sample.txt` with a transcription of that page.

## Troubleshooting

| Problem | Fix |
|---|---|
| `tesseract is not installed or it's not in your PATH` | Install Tesseract and add it to PATH, or set `pytesseract.pytesseract.tesseract_cmd`. |
| Notebook warns about language `'spa'` and falls back to `eng` | Install the Spanish data. Do **not** report results from the English fallback. |
| `Could not read ... 07_final.png` | Run Stage 1 first. |
| `NO GROUND TRUTH LOADED` | Create `data/ground_truth/sample.txt`, one line per manuscript line, with accents. |
| TrOCR or Marian fails to load | Check internet and that `torch`, `transformers`, `sentencepiece` are installed. |
| TrOCR is slow | Use a GPU, or set `HTR_MODEL = "microsoft/trocr-small-handwritten"`. |

## Citing the LaTeX report

`main.tex` is the project report and `references.bib` holds the same eight references. `main.tex` currently uses an inline `thebibliography`. To use the `.bib` file instead, replace that block with:

```latex
\bibliographystyle{IEEEtran}
\bibliography{references}
```

then run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
