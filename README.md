# 🐾 Pet Image Classifier — CNN with PyTorch (ResNet, AlexNet, VGG)

A Python command-line application that uses **pretrained Convolutional Neural Networks** to classify pet images, determine whether each image contains a dog, and benchmark three CNN architectures — **ResNet**, **AlexNet**, and **VGG16** — to identify the best performer.

Built as part of the **Udacity AI Programming with Python Nanodegree** — Pre-training project using transfer learning with ImageNet models.

---

## 📌 Project Overview

The program extracts the true pet label from each image filename, runs it through a chosen CNN classifier, and compares the classifier's prediction against the true label. It answers two core questions:

1. **Is this image a dog?** (regardless of breed)
2. **If it's a dog, is the breed correctly identified?**

All three models are benchmarked in batch to determine which provides the best overall classification accuracy.

---

## 📁 Project Structure

```
├── check_images.py                      # Main entry point — orchestrates the full pipeline
├── get_input_args.py                    # Parses CLI arguments (--dir, --arch, --dogfile)
├── get_pet_labels.py                    # Extracts pet labels from image filenames
├── classify_images.py                   # Runs CNN classifier & compares to pet labels
├── adjust_results4_isadog.py            # Flags whether pet/classifier labels are dogs
├── calculates_results_stats.py          # Computes counts & accuracy percentages
├── print_results.py                     # Prints summary + misclassification details
├── classifier.py                        # CNN wrapper (ResNet18, AlexNet, VGG16)
├── print_functions_for_lab_checks.py    # Validation helpers for each pipeline step
├── test_classifier.py                   # Standalone test for the classifier function
├── dognames.txt                         # Reference list of valid dog breed names
├── imagenet1000_clsid_to_human.txt      # ImageNet class ID → human-readable label map
├── run_models_batch.sh                  # Shell script to run all 3 models in sequence
├── run_models_batch_uploaded.sh         # Same, for uploaded custom images
├── pet_images/                          # 40 test images (dogs, cats, non-animals)
├── alexnet_pet-images.txt               # AlexNet classification output log
├── resnet_pet-images.txt                # ResNet classification output log
├── vgg_pet-images.txt                   # VGG classification output log
└── check_images.txt                     # Q&A answers for uploaded image classification
```

---

## ⚙️ Pipeline

The program runs through **6 sequential steps**, each implemented in its own module:

```
1. get_input_args()          → Parse --dir, --arch, --dogfile from CLI
2. get_pet_labels()          → Extract true labels from image filenames
3. classify_images()         → Run CNN classifier, compare to pet labels
4. adjust_results4_isadog()  → Flag is-a-dog / is-NOT-a-dog for both labels
5. calculates_results_stats() → Compute accuracy counts and percentages
6. print_results()           → Print summary + optional misclassification detail
```

### Results Dictionary Structure

Each image is stored as a key→list entry throughout the pipeline:

| Index | Content | Type |
|---|---|---|
| 0 | Pet image label (from filename) | string |
| 1 | Classifier label (from CNN) | string |
| 2 | Label match (1 = match, 0 = no match) | int |
| 3 | Pet label is-a-dog (1 = yes, 0 = no) | int |
| 4 | Classifier label is-a-dog (1 = yes, 0 = no) | int |

---

## 🧠 CNN Models

All three models are pretrained on **ImageNet** (1,000 classes) and loaded via `torchvision.models`:

| Model | Architecture | Notes |
|---|---|---|
| **VGG16** | Deep, sequential conv layers | Highest breed accuracy |
| **ResNet18** | Residual connections | Best balance on uploaded images |
| **AlexNet** | Simpler, faster | Lowest accuracy but quickest runtime |

The `classifier.py` wrapper handles image preprocessing (resize → center crop → normalize to ImageNet stats) and returns the human-readable ImageNet class label.

---

## 📊 Metrics Computed

| Metric | Description |
|---|---|
| `n_images` | Total number of images |
| `n_dogs_img` | Number of dog images |
| `n_notdogs_img` | Number of non-dog images |
| `n_match` | Exact label matches |
| `n_correct_dogs` | Dogs correctly classified as dogs |
| `n_correct_notdogs` | Non-dogs correctly classified as non-dogs |
| `n_correct_breed` | Dogs with correct breed identified |
| `pct_match` | % exact label matches |
| `pct_correct_dogs` | % dogs correctly identified |
| `pct_correct_breed` | % dog breeds correctly identified |
| `pct_correct_notdogs` | % non-dogs correctly identified |

---

## 🚀 How to Run

### Single model
```bash
python3 check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

### All three models in batch
```bash
sh run_models_batch.sh
```

### CLI Arguments

| Argument | Default | Options | Description |
|---|---|---|---|
| `--dir` | `pet_images/` | any folder path | Directory containing images to classify |
| `--arch` | `vgg` | `vgg`, `resnet`, `alexnet` | CNN model architecture to use |
| `--dogfile` | `dognames.txt` | any .txt file | Text file with valid dog breed names |

### Example output
```
*** Results Summary for CNN Model Architecture VGG ***
N Images: 40
N Dog Images: 30
N Not-Dog Images: 10

pct_match: 87.5
pct_correct_dogs: 100.0
pct_correct_breed: 93.33
pct_correct_notdogs: 100.0

** Total Elapsed Runtime: 0:0:42
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `PyTorch` / `torchvision` | Pretrained CNN models (ResNet18, AlexNet, VGG16) |
| `PIL / Pillow` | Image loading and preprocessing |
| `argparse` | CLI argument parsing |
| `os` / `listdir` | Filename-based label extraction |
| `time` | Runtime measurement |

---

## 🔍 Key Design Notes

- **Label extraction from filenames** — true pet labels are parsed from the image filename by splitting on `_`, keeping only alphabetic words, and lowercasing. e.g. `Boston_terrier_02259.jpg` → `boston terrier`
- **Dog detection** — `dognames.txt` is loaded into a dictionary for O(1) lookup; both the pet label and the classifier label are checked independently
- **Classifier label matching** — because ImageNet can return comma-separated synonyms (e.g. `dalmatian, coach dog, carriage dog`), the pet label is checked for substring membership in the full classifier string
- **Misclassification reporting** — `print_results()` can optionally print all cases where a dog was misidentified as non-dog (or vice versa) and all cases where the breed was incorrectly named

---

## ✅ Results Summary (from output logs)

| Model | % Correct Dogs | % Correct Breed | % Correct Non-Dogs |
|---|---|---|---|
| VGG16 | 100% | 93.3% | 100% |
| ResNet | 100% | 90.0% | 90.0% |
| AlexNet | 100% | 80.0% | 100% |

**Best overall: VGG16** — highest breed accuracy with perfect dog/non-dog detection.  
**Best on uploaded images: ResNet** — more robust on real-world images outside the standard pet set.

---

## 📝 Notes & Limitations

- All three CNN models are pretrained on ImageNet — no fine-tuning is performed in this project
- Label matching is substring-based, which can produce false positives if a non-dog class name contains a dog breed name as a substring
- The `dognames.txt` file must cover both standard pet labels and all ImageNet dog class synonyms for accurate dog detection
- Runtime varies significantly by model: VGG16 is slowest, AlexNet is fastest

---

## 👤 Author

**Mohga** — Built as a pre-training submission for the **Udacity AI Programming with Python Nanodegree** — Introduction to CNNs & Transfer Learning module.
