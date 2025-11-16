# The Detection Game 

An interactive study comparing human vs AI performance in detecting AI-generated artwork.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run src/Home.py
```

## Project Structure

```
├── src/                    # Main application code
│   ├── backend/           # appliation backend core logic and utilties files
│   │   ├── config.py      # Configuration settings
│   │   ├── detector.py    # AI detection model
│   │   ├── feedback.py    # Progress tracking & analytics
│   │   ├── prepare_images.py  # Data preprocessing
│   │   └── utils.py       # Database & utility functions
│   ├── pages/             # Streamlit pages
│   └── Home.py            # Main application python file
├── data/                  # Dataset and metadata
│   ├── art_testset/       # Processed image dataset (1000 images)
│   └── results.db         # Research data storage
├── scripts/               # Analysis scripts
├── models/                # Pre-trained AI detection model
└── requirements.txt       # Python dependencies
```

##  Research Features

- **Human vs AI Testing**: Interactive image classification
- **Comprehensive Survey**: User background and perception data
- **Real-time Analytics**: Research dashboard with insights
- **Secure Dataset**: Balanced AI generators with no filename leakage
- **Generator Analysis**: Performance by AI model (DALL-E, Midjourney, etc.)


## The testset

- **1,000 Images**: 500 human-made art + 500 AI-generated art
- **6 AI Generators**: Stable Diffusion, Latent Diffution, Midjourney, DreamStudio, StarryAI, DALL-E
- **Art Styles**: Art Nouveau, Baroque, Expressionism, Impressionism, Post Impressionism, Realism, Renaissance, Romanticism, Surrealism, Ukiyo-e, Pixiv (anime and illustrations) and general art.

## Acknowledgments

This project uses DINOv3 for feature extraction in our AI detection model. 

## Citations

If you use this work, please also cite DINOv3:

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```