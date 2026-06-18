import sys
import os

sys.path.append(os.path.abspath("."))

from models.gcn import MolecularGCN

model = MolecularGCN()

print(model)