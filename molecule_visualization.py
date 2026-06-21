from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Ethanol molecule
mol = Chem.MolFromSmiles("CCO")

mol = Chem.AddHs(mol)

AllChem.EmbedMolecule(mol)

img = Draw.MolToImage(
    mol,
    size=(600, 600)
)

img.save("results/ethanol.png")

print("Image saved!")