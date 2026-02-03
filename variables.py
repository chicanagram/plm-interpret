address_dict = {
    'examples': '../PIPS/PIPS-tools/examples/',
    'PIPS': '../PIPS/PIPS-GalOx-data/',
    'PIPS2': '../PIPS/PIPS2-UPOs-data/',
    'influenza-resistance': '../PIPS/influenza-resistance/',
    'ProtSolM': '../solubility-data/ProtSolM/',
    'SoluProtMut': '../solubility-data/SoluProtMut/',
    'SoluProtMut_PKS': '../solubility-data/SoluProtMut/',
    'SoluProtMut_LGK': '../solubility-data/SoluProtMut/',
    'SoluProtMut_bLac': '../solubility-data/SoluProtMut/',
    'SoluProtMut_PKS-bLac': '../solubility-data/SoluProtMut/',
    'SoluProtMut_PKS-bLac-LGK': '../solubility-data/SoluProtMut/',
    'PON-Sol2': '../solubility-data/PON-Sol2/',
    'ECOHARVEST': '../ECOHARVEST/',
    'pips-insilico': '../pips-insilico/data/',
    'databases': '../seq-db/',
    'plm-interpret-data': '../plm-interpret-data/',
    'plm-interpret-data-ssd': 'E:/projects/plm-interpret-data/'
}

subfolders = {
    'sequences': 'sequences/',
    'msa': 'msa/',
    'blast': 'blast/',
    'hmm': 'hmm/',
    'conservation_analysis': 'conservation_analysis/',
    'aggregation': 'aggregation/',
    'stability': 'stability/',
    'ml_prediction': 'ml_prediction/',
    'yasara': 'yasara/',
    'pdb': 'pdb/',
    'sce': 'sce/',
    'patents': 'patents/',
    'seqsearch': 'seqsearch/',
    'protein_embeddings': 'protein_embeddings/',
    'sae_latents': 'sae_latents/',
    'expdata': 'expdata/',
    'mutagenesis_proposal': 'mutagenesis_proposal/',
    'solubility': 'solubility/',
    'camsol': 'camsol/',
    'deepsolue': 'deepsolue/',
    'ponsol2': 'ponsol2/',
    'netsolp': 'netsolp/',
    'ohe': 'ohe/',
}

mapping = {
    'A': 'Ala',
    'H': 'His',
    'Y': 'Tyr',
    'R': 'Arg',
    'T': 'Thr',
    'K': 'Lys',
    'M': 'Met',
    'D': 'Asp',
    'N': 'Asn',
    'C': 'Cys',
    'Q': 'Gln',
    'E': 'Glu',
    'G': 'Gly',
    'I': 'Ile',
    'L': 'Leu',
    'F': 'Phe',
    'P': 'Pro',
    'S': 'Ser',
    'W': 'Trp',
    'V': 'Val'
    }

mapping_rev = {
    'ALA': 'A',
    'HIS': 'H',
    'TYR': 'Y',
    'ARG': 'R',
    'THR': 'T',
    'LYS': 'K',
    'MET': 'M',
    'ASP': 'D',
    'ASN': 'N',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'ILE': 'I',
    'LEU': 'L',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'TRP': 'W',
    'VAL': 'V'
}

aaList = list("ACDEFGHIKLMNPQRSTVWY")
aaList_with_X = list("ACDEFGHIKLMNPQRSTVWYX")
# aaList = ['A','H','Y','R','T','K','M','D','N','C','Q','E','G','I','L','F','P','S','W','V']
aa2idx = {aa: i for i, aa in enumerate(aaList)}

element_mapping = {
    1: "H",  # Hydrogen
    6: "C",  # Carbon
    7: "N",  # Nitrogen
    8: "O",  # Oxygen
    15: "P", # Phosphorus
    16: "S", # Sulfur
    11: "Na",# Sodium
    12: "Mg",# Magnesium
    19: "K", # Potassium
    20: "Ca",# Calcium
    17: "Cl",# Chlorine
    9: "F",  # Fluorine
    35: "Br",# Bromine
    53: "I"  # Iodine
}

amino_acid_groups = {
    "np": ["F", "L", "I", "V", "M", "A", "W", "G", "P"],
    "p~": ["Y", "C", "T", "S", "H", "Q", "N"],
    "p-": ["E", "D"],  # Acidic
    "p+": ["K", "R"]   # Basic
}

aa_polarity_mapping = {
    "A": "np",
    "V": "np",
    "I": "np",
    "L": "np",
    "M": "np",
    "F": "np",
    "W": "np",
    "P": "np",
    "G": "np",

    "S": "p~",
    "T": "p~",
    "N": "p~",
    "Q": "p~",
    "Y": "p~",
    "C": "p~",

    "D": "p-",
    "E": "p-",

    "K": "p+",
    "R": "p+",
    "H": "p+",  # weak base, pKa ~6
}

# Kyte–Doolittle Hydropathy Index
# (positive = more hydrophobic)
kyte_doolittle_hydrophobicity_index = {
    "A": 1.8,
    "V": 4.2,
    "I": 4.5,
    "L": 3.8,
    "M": 1.9,
    "F": 2.8,
    "W": -0.9,
    "P": -1.6,
    "G": -0.4,
    "C": 2.5,
    "S": -0.8,
    "T": -0.7,
    "Y": -1.3,
    "N": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "E": -3.5,
    "K": -3.9,
    "R": -4.5,
    "H": -3.2,
}

# Hopp–Woods Polarity Index
# (positive = more polar / surface-exposed)
hopp_woods_polarity_index = {
    "R": 3.0,
    "K": 3.0,
    "D": 3.0,
    "E": 3.0,
    "Q": 0.2,
    "N": 0.2,
    "H": -0.5,
    "S": 0.3,
    "T": -0.4,
    "Y": -2.3,
    "C": -1.0,
    "G": 0.0,
    "P": 0.0,
    "A": -0.5,
    "V": -1.5,
    "I": -1.8,
    "L": -1.8,
    "M": -1.3,
    "F": -2.5,
    "W": -3.4,
}

aa_to_cmap_color_mapping = {
    'Clustal': {
        '-': 0,  # Gaps
        'A': 1, 'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1,  # Hydrophobic (Green)
        'K': 2, 'R': 2,  # Positive charge (Blue)
        'D': 3, 'E': 3,  # Negative charge (Red)
        'S': 4, 'T': 4, 'N': 4, 'Q': 4,  # Polar (Cyan)
        'Y': 5, 'H': 5,  # Aromatic (Magenta)
        'G': 6,  # Glycine (Orange)
        'P': 7,  # Proline (Yellow)
        'X': 8
    },
    'Taylor': {
        '-': 0,  # Gaps (Light Gray)
        'A': 1, 'V': 1, 'L': 1, 'I': 1, 'M': 1,  # Hydrophobic (Green)
        'F': 2, 'Y': 2, 'W': 2,  # Aromatic (Blue)
        'K': 3, 'R': 3, 'H': 3,  # Positive Charge (Red)
        'D': 4, 'E': 4,  # Negative Charge (Magenta)
        'S': 5, 'T': 5, 'N': 5, 'Q': 5,  # Polar Uncharged (Cyan)
        'C': 6,  # Cysteine (Yellow)
        'G': 7,  # Glycine (Orange)
        'P': 8,  # Proline (Brown)
        'X': 9  # Ambiguous/Unknown (Black)
    }
}
palettes = {
    'Clustal': [
        "#d3d3d3",  # Gaps (Light Gray)
        "#32CD32",  # Hydrophobic (Green)
        "#0000FF",  # Positive (Blue)
        "#FF0000",  # Negative (Red)
        "#00FFFF",  # Polar (Cyan)
        "#FF00FF",  # Aromatic (Magenta)
        "#FFA500",  # Glycine (Orange)
        "#FFFF00",  # Proline (Yellow)
        "#000000"  # Ambiguous (Black)
    ],
    'Taylor': [
        "#D3D3D3",  # Gaps (Light Gray)
        "#33FF00",  # Hydrophobic (Green)
        "#0099FF",  # Aromatic (Blue)
        "#FF0000",  # Positive Charge (Red)
        "#CC00FF",  # Negative Charge (Magenta)
        "#00FFFF",  # Polar Uncharged (Cyan)
        "#FFFF00",  # Cysteine (Yellow)
        "#FF9900",  # Glycine (Orange)
        "#996633",  # Proline (Brown)
        "#000000"  # Ambiguous (Black)
    ],

    'Taylor_yasara': [
        "grey",  # Gaps (Light Gray)
        "green",  # Hydrophobic (Green)
        "blue",  # Aromatic (Blue)
        "red",  # Positive Charge (Red)
        "magenta",  # Negative Charge (Magenta)
        "cyan",  # Polar Uncharged (Cyan)
        "yellow",  # Cysteine (Yellow)
        "orange",  # Glycine (Orange)
        "brown",  # Proline (Brown)
        "black"  # Ambiguous (Black)
    ]
}

aa_taylor_colorcode_yasara = {
    '-': 'grey',
    'A': 'green',
    'V': 'green',
    'L': 'green',
    'I': 'green',
    'M': 'green',
    'F': 'blue',
    'Y': 'blue',
    'W': 'blue',
    'K': 'red',
    'R': 'red',
    'H': 'red',
    'D': 'magenta',
    'E': 'magenta',
    'S': 'cyan',
    'T': 'cyan',
    'N': 'cyan',
    'Q': 'cyan',
    'C': 'yellow',
    'G': 'orange',
    'P': 'brown',
    'X': 'black'
}

amino_acids_by_size = [
    'G',  # Glycine
    'A',  # Alanine
    'S',  # Serine
    'C',  # Cysteine
    'D',  # Aspartic Acid
    'N',  # Asparagine
    'T',  # Threonine
    'E',  # Glutamic Acid
    'Q',  # Glutamine
    'P',  # Proline
    'V',  # Valine
    'H',  # Histidine
    'M',  # Methionine
    'I',  # Isoleucine
    'L',  # Leucine
    'F',  # Phenylalanine
    'Y',  # Tyrosine
    'W',  # Tryptophan
    'K',  # Lysine
    'R'   # Arginine
]

aa_sidechain_volume = {
    "G": 48.0,
    "A": 67.0,
    "S": 73.0,
    "C": 86.0,
    "D": 91.0,
    "T": 93.0,
    "N": 96.0,
    "P": 90.0,
    "V": 105.0,
    "E": 109.0,
    "Q": 114.0,
    "H": 118.0,
    "M": 124.0,
    "I": 124.0,
    "L": 124.0,
    "K": 135.0,
    "F": 135.0,
    "Y": 141.0,
    "R": 148.0,
    "W": 163.0,
}

hetatm_non_metal_ion = ['HEM', 'NAG']
hetatm_metal_ion = [' MG', '  K', ' NA', ' MN', ' CA']
hetatm_anion = [' CL']
