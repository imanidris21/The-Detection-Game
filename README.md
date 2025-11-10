# The Detection Game 🎯

A platform for studying human vs AI performance in detecting AI-generated artwork.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run src/Home.py
```

## 📁 Project Structure

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

##  Security Features

- **Filename Anonymization**: No data leakage from image names
- **Fallback Mode Removed**: Eliminates filename-based cheating
- **Balanced Dataset**: Equal representation across 6 AI generators

## The testset

- **1,000 Images**: 500 human-made art + 500 AI-generated art
- **6 AI Generators**: Stable Diffusion, Latent Diffution, Midjourney, DreamStudio, StarryAI, DALL-E
- **Art Styles**: Art Nouveau, Baroque, Expressionism, Impressionism, Post Impressionism, Realism, Renaissance, Romanticism, Surrealism, Ukiyo-e, Pixiv (anime and illustrations) and general art. 



