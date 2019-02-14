from cltk.stem.sanskrit.code import configuration
from cltk.stem.sanskrit.code.apply import *
from cltk.stem.sanskrit.code.iast_dev import *
from cltk.stem.sanskrit.code.dev_iast import *
from git import Repo
import os

config = configuration.config
class SandhiSplitter():
    def __init__(self, text, isIAST=False):
        self.text = text
        self.isiast = isIAST
        if self.isiast == False:
            self.text = dviast(self.text)
            
    def getSandhi(self):
        if os.path.exists("../data/input/additional-data-0-128.json") == False:
            print("Trained model not found, Downloading....")
            Repo.clone_from("https://github.com/sainimohit23/SanskritModel", "../data")
        preds = split_sandhi(self.text, config)
        textSplitted = self.text.split()
        predsSplitted = preds.split()
        d = {}
        for i in range(len(predsSplitted)):
            li = []
            for word in predsSplitted[i].split('-'):
                if len(word)>0:
                    li.append(iastdv(word))
            d[iastdv(textSplitted[i])] = li
            
        return d