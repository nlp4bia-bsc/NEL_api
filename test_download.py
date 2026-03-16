import os
import sys
import json
from app.initializer import main as download_models

def main():
    lang = "es"
    entities = ["disease", "symptoms"]
    download_models(lang, entities, negation=True)

if __name__ == "__main__":
    main()