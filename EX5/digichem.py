import os, sys

import pandas as pd
import numpy as np

import py3Dmol
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

# unit 01

def viewmol(mol, blocktype='xyz', width=500, height=500):

    viewer = py3Dmol.view(width=width, height=height)
    if blocktype == 'xyz':
        viewer.addModel(rdkit.Chem.MolToXYZBlock(mol), 'xyz')
    elif blocktype == 'mol':
        viewer.addModel(rdkit.Chem.MolToMolBlock(mol), 'mol')
    else:
        print('Only blocktype mol or xyz are possible.')
        
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    viewer.show()

################################################################
# unit 02
#################################################################

def get_distance_matrix(mol, redundant=True):
    """
    Generate a distance matrix representing pairwise distances between atoms in a molecule.

    Parameters:
    - mol: RDKit molecule object
        The input molecule for which the distance matrix is calculated.
    - redundant: bool, optional (default=True)
        If True, remove hydrogen atoms from the molecule before calculation.

    Returns:
    - dist_matrix: numpy.ndarray
        The calculated distance matrix, where element dist_matrix[i, j] represents
        the Euclidean distance between atom i and atom j in the molecule.

    Notes:
    - The distance matrix is symmetric with zeros along the diagonal.
    - The distances are calculated based on the Cartesian coordinates of the atoms
      obtained from the conformer of the molecule.

    If the molecule does not have any conformers, hydrogen atoms will be added and
    the molecule will be embedded using a force field (ETKDG) to generate geometry.
    If 'redundant' is set to True, hydrogen atoms will be removed from the molecule
    before calculating distances.
    """

    if not mol.GetNumConformers():
        # Geometry will be generated from force field
        mol = rdkit.Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    if redundant:
        # Remove H atoms if redundant
        mol = rdkit.Chem.RemoveHs(mol)

    Z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    num_atoms = len(Z)

    # Get coordinates from mol object
    coords = [[mol.GetConformer().GetAtomPosition(i).x,
               mol.GetConformer().GetAtomPosition(i).y,
               mol.GetConformer().GetAtomPosition(i).z] for i in range(num_atoms)]

    # Calculate distances
    dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix

def get_coulomb_matrix(mol, sortedCM=True, redundant=True):
    """
    Generate a Coulomb matrix structural representation from an RDKit molecule object.

    Parameters:
    - mol: RDKit molecule object
        The input molecule for which the Coulomb matrix is calculated.
    - sortedCM: bool, optional (default=True)
        Whether to return a sorted Coulomb matrix based on atomic numbers.
    - redundant: bool, optional (default=True)
        If True, remove hydrogen atoms from the molecule before calculation.
        If False, add hydrogen atoms to the molecule if none are present.

    Returns:
    - Z: list or array-like
        Atomic numbers of the atoms in the molecule.
    - coulomb_mat: numpy.ndarray
        The calculated Coulomb matrix representing the molecular structure.
        If sortedCM is True, the matrix is sorted based on atomic numbers.

    Raises:
    - AssertionError: If the dimensions of the distance matrix do not match
      the number of atoms in the molecule.
    """

    # Generate 3D structure from force field if not existent
    if not mol.GetNumConformers():
        mol = rdkit.Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Remove hydrogens if redundant flag is set
    if redundant:
        mol = rdkit.Chem.RemoveHs(mol)
                   
    # Calculate distances
    dist_matrix = get_distance_matrix(mol=mol, redundant=redundant)

    # shape from current Mol object
    Z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    num_atoms = len(Z)

    # Check matrix shape
    assert dist_matrix.shape[0] == dist_matrix.shape[1] == num_atoms
    
    # Calculate Coulomb matrix
    coulomb_mat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                cm = Z[i] * Z[j] / dist_matrix[i, j]
                coulomb_mat[i,j] = cm
                coulomb_mat[j,i] = cm
            else:
                cm = 0.5*Z[i]**(2.4)
                coulomb_mat[i,i] = cm

    # Sort Coulomb matrix if sortedCM is True
    if sortedCM:
        sorted_Z = np.sort(Z)[::-1]
        idx_sorted = np.argsort(Z)[::-1]
        sorted_coulomb_mat = coulomb_mat[:, idx_sorted][idx_sorted, :]
        
        return sorted_Z, sorted_coulomb_mat
        
    else:
        return Z, coulomb_mat

def __xyz2mol__(xyz_file, charge):

    raw_mol = rdkit.Chem.MolFromXYZFile(xyz_file)
    mol = rdkit.Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    return mol

def __xyz2dict__(xyz_path, extension='xyz', charges={'K13': -1, 'K17': -1, 'K18': 0}):

    xyz_files = sorted([n for n in os.listdir(xyz_path) if n.endswith(extension)])

    d_struc = {}
    for name in xyz_files:
        tmp = {}
        xyz_file = os.path.join(xyz_path, name)
        molid = name[:-4].split('_')[-1]
        mol_charge = charges.get(molid, +1)
        mol = __xyz2mol__(xyz_file=xyz_file, charge=mol_charge)
        
        tmp = {
            'mol':mol,
            'atNum': mol.GetNumAtoms(),
            'bondNum': mol.GetNumBonds(),
            'smiles': rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(mol)),
            'inchi': rdkit.Chem.MolToInchi(rdkit.Chem.RemoveHs(mol)),
            'adj_mat': rdkit.Chem.GetAdjacencyMatrix(mol),
            'dist_mat_top': rdkit.Chem.GetDistanceMatrix(mol),
            'dist_mat_geo_red': get_distance_matrix(mol=mol, redundant=True),
            'coulomb_mat_red': get_coulomb_matrix(mol=mol, sortedCM=True, redundant=True)
        }
       
        d_struc[molid] = tmp

    return d_struc
    
#######################################################
## unit 03
#######################################################

def tanimoto_similarity(mol1, mol2):

    fp1 = rdkit.Chem.RDKFingerprint(rdkit.Chem.RemoveHs(mol1))
    fp2 = rdkit.Chem.RDKFingerprint(rdkit.Chem.RemoveHs(mol2))

    tan_sim = round(rdkit.DataStructs.TanimotoSimilarity(fp1,fp2),3)

    return tan_sim

from rdkit.Chem import rdFMCS

def mcs_similarity(mol1,mol2):
    # get maximum common substructure for a pair of molecules
    mcs = rdFMCS.FindMCS([mol1,mol2], timeout=10)
    # get number of commonb bonds
    mcs_bonds = mcs.numBonds

    #get number of bonds for each molecule (only heavy atoms)
    mol1_bonds = mol1.GetNumBonds()
    mol2_bonds = mol2.GetNumBonds()

    # compute MCS-based Tanimoto similarity
    tan_mcs = mcs_bonds / (mol1_bonds + mol2_bonds - mcs_bonds)

    return(tan_mcs)

# unit 07

class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df

    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names
    def __str__(self):
        return "%d bit FP" % len(self.fp)
    def __len__(self):
        return len(self.fp)

class Spectrum:
    """
    class, useful to pack features in pandas df

    Parameters
    ----------
    spec : np.array
        Features stored in numpy array
    """
    def __init__(self, spec):
        self.spec = spec
        self.spec_min = min(self.spec[:,0])
        self.spec_max = max(self.spec[:,0])
        self.step_size = (self.spec[1,0]-self.spec[0,0])

    def __str__(self):
        return str(self.spec_min)+" to "+str(self.spec_max)+" nm, steps: "+str(self.step_size)+" nm"

    def __len__(self):
        return len(self.spec[:,0])

# -------------------------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def plot2select_k(X_test, y_test, X_train, y_train, max_k=10, p=2, metric='minkowski'):

    error_rate_train = []
    acc_train = []

    error_rate_test = []
    acc_test = []

    # Calculate error rates and accuracies
    for i in range(1, max_k+1):
        knn = Pipeline(
                steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=i, p=p, metric=metric))]
                )

        knn.fit(X_train, y_train)

        pred_train = knn.predict(X_train)
        pred_test = knn.predict(X_test)

        error_rate_test.append(np.mean(pred_test != y_test))
        error_rate_train.append(np.mean(pred_train != y_train))

        acc_test.append(metrics.accuracy_score(y_test, pred_test))
        acc_train.append(metrics.accuracy_score(y_train, pred_train))

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,3.5))

    # Plot Error Rate
    axes[0].plot(range(1, max_k+1), error_rate_test, linestyle='dashed',
                 color='orange', marker='o', markerfacecolor='orange', markersize=8, label='test')
    axes[0].plot(range(1, max_k+1), error_rate_train, linestyle='dashed',
                 color='blue', marker='o', markerfacecolor='blue', markersize=8, label='train')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Error Rate')

    # Plot Accuracy
    axes[1].plot(range(1, max_k+1), acc_test, linestyle='dashed', color='orange',
                 marker='o', markerfacecolor='orange', markersize=8, label='test')
    axes[1].plot(range(1, max_k+1), acc_train, linestyle='dashed', color='blue',
                 marker='o', markerfacecolor='blue', markersize=8, label='train')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Accuracy')

    # Adjust
    plt.legend()
    plt.tight_layout()
    plt.show()

