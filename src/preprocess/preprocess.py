import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from typing import Generator

class PreTraitement:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def supprimer_valeurs_manquantes(self) -> None:
        self.dataset.dropna(inplace=True)

    def nettoyer_texte(self, text: str) -> str:
        text = text.lower()
        text = text.strip()
        text = re.sub(r'<.*?>', '', text)   # HTML
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return text

def generer_donnees(dataset: pd.DataFrame, taille_lot: int = 1000) -> Generator:
    for debut in range(0, len(dataset), taille_lot):
        yield dataset.iloc[debut:debut + taille_lot]
def generer_et_traiter_donnees(filename: str) -> pd.DataFrame:
    #dataset = load_dataset("imdb", split=split)
    df = pd.read_csv("../../data/IMDB Dataset.csv",engine='python',on_bad_lines='skip')
    le = LabelEncoder()
    for lot in generer_donnees(df):
        lot["sentiment"]=le.fit_transform(lot["sentiment"])
        traitement = PreTraitement(lot)
        traitement.supprimer_valeurs_manquantes()
        lot["review"] = lot["review"].apply(traitement.nettoyer_texte)
        lot.to_csv(filename, mode="a", header=False, index=False)
    return lot