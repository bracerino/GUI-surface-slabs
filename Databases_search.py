import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from matminer.featurizers.structure import PartialRadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from collections import defaultdict
from itertools import combinations
import streamlit.components.v1 as components
import py3Dmol
from io import StringIO
import pandas as pd
import plotly.graph_objs as go
from pymatgen.core import Structure as PmgStructure
import matplotlib.colors as mcolors
import streamlit as st
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from math import cos, radians, sqrt
import io
import re
import spglib
from pymatgen.core import Structure
from aflow import search, K
from aflow import search  # ensure your file is not named aflow.py!
import aflow.keywords as AFLOW_K
import requests




def get_full_conventional_structure_diffra(structure, symprec=1e-3):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords

    species_list = [site.species for site in structure]
    species_to_type = {}
    type_to_species = {}
    type_index = 1

    types = []
    for sp in species_list:
        sp_tuple = tuple(sorted(sp.items()))  # make it hashable
        if sp_tuple not in species_to_type:
            species_to_type[sp_tuple] = type_index
            type_to_species[type_index] = sp
            type_index += 1
        types.append(species_to_type[sp_tuple])

    cell = (lattice, positions, types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    std_lattice = dataset.std_lattice
    std_positions = dataset.std_positions
    std_types = dataset.std_types

    new_species_list = [type_to_species[t] for t in std_types]

    conv_structure = Structure(
        lattice=std_lattice,
        species=new_species_list,
        coords=std_positions,
        coords_are_cartesian=False
    )

    return conv_structure



def get_formula_type(formula):
    elements = []
    counts = []

    import re
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    for element, count in matches:
        elements.append(element)
        counts.append(int(count) if count else 1)

    if len(elements) == 1:
        return "A"

    elif len(elements) == 2:
        # Binary compounds
        if counts[0] == 1 and counts[1] == 1:
            return "AB"
        elif counts[0] == 1 and counts[1] == 2:
            return "AB2"
        elif counts[0] == 2 and counts[1] == 1:
            return "A2B"
        elif counts[0] == 1 and counts[1] == 3:
            return "AB3"
        elif counts[0] == 3 and counts[1] == 1:
            return "A3B"
        elif counts[0] == 1 and counts[1] == 4:
            return "AB4"
        elif counts[0] == 4 and counts[1] == 1:
            return "A4B"
        elif counts[0] == 1 and counts[1] == 5:
            return "AB5"
        elif counts[0] == 5 and counts[1] == 1:
            return "A5B"
        elif counts[0] == 1 and counts[1] == 6:
            return "AB6"
        elif counts[0] == 6 and counts[1] == 1:
            return "A6B"
        elif counts[0] == 2 and counts[1] == 3:
            return "A2B3"
        elif counts[0] == 3 and counts[1] == 2:
            return "A3B2"
        elif counts[0] == 2 and counts[1] == 5:
            return "A2B5"
        elif counts[0] == 5 and counts[1] == 2:
            return "A5B2"
        elif counts[0] == 1 and counts[1] == 12:
            return "AB12"
        elif counts[0] == 12 and counts[1] == 1:
            return "A12B"
        elif counts[0] == 2 and counts[1] == 17:
            return "A2B17"
        elif counts[0] == 17 and counts[1] == 2:
            return "A17B2"
        elif counts[0] == 3 and counts[1] == 4:
            return "A3B4"
        else:
            return f"A{counts[0]}B{counts[1]}"

    elif len(elements) == 3:
        # Ternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1:
            return "ABC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3:
            return "ABC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1:
            return "AB3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1:
            return "A3BC"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4:
            return "AB2C4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4:
            return "A2BC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 2:
            return "AB4C2"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1:
            return "A2B4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 2:
            return "A4BC2"
        elif counts[0] == 4 and counts[1] == 2 and counts[2] == 1:
            return "A4B2C"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1:
            return "AB2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1:
            return "A2BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2:
            return "ABC2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4:
            return "ABC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1:
            return "AB4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1:
            return "A4BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 5:
            return "ABC5"
        elif counts[0] == 1 and counts[1] == 5 and counts[2] == 1:
            return "AB5C"
        elif counts[0] == 5 and counts[1] == 1 and counts[2] == 1:
            return "A5BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 6:
            return "ABC6"
        elif counts[0] == 1 and counts[1] == 6 and counts[2] == 1:
            return "AB6C"
        elif counts[0] == 6 and counts[1] == 1 and counts[2] == 1:
            return "A6BC"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 1:
            return "A2B2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 2:
            return "A2BC2"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 2:
            return "AB2C2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1:
            return "A3B2C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2:
            return "A3BC2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2:
            return "AB3C2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1:
            return "A2B3C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3:
            return "A2BC3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3:
            return "AB2C3"
        elif counts[0] == 3 and counts[1] == 3 and counts[2] == 1:
            return "A3B3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 3:
            return "A3BC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 3:
            return "AB3C3"
        elif counts[0] == 4 and counts[1] == 3 and counts[2] == 1:
            return "A4B3C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 3:
            return "A4BC3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 3:
            return "AB4C3"
        elif counts[0] == 3 and counts[1] == 4 and counts[2] == 1:
            return "A3B4C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 4:
            return "A3BC4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "AB3C4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "ABC6"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 7:
            return "A2B2C7"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}"

    elif len(elements) == 4:
        # Quaternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "ABCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "ABCD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "ABC3D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "AB3CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A3BCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "ABCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "ABC4D"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "AB4CD"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A4BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 4:
            return "AB2CD4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "A2BCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 4:
            return "ABC2D4"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4 and counts[3] == 1:
            return "AB2C4D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "A2BC4D"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "A2B4CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A2BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "AB2CD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "ABC2D"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "ABCD2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "A3B2CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "A3BC2D"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "A3BCD2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2 and counts[3] == 1:
            return "AB3C2D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 2:
            return "AB3CD2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 2:
            return "ABC3D2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "A2B3CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "A2BC3D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "A2BCD3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3 and counts[3] == 1:
            return "AB2C3D"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 3:
            return "AB2CD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 3:
            return "ABC2D3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 6:
            return "A1B4C1D6"
        elif counts[0] == 5 and counts[1] == 3 and counts[2] == 1 and counts[3] == 13:
            return "A5B3C1D13"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 4 and counts[3] == 9:
            return "A2B2C4D9"

        elif counts == [3, 2, 1, 4]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C1D4"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}"

    elif len(elements) == 5:
        # Five-element compounds (complex minerals like apatite)
        if counts == [1, 1, 1, 1, 1]:
            return "ABCDE"
        elif counts == [10, 6, 2, 31, 1]:  # Apatite-like: Ca10(PO4)6(OH)2
            return "A10B6C2D31E"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13DE"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13"
        elif counts == [3, 2, 3, 12, 1]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C3D12E"

        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}E{counts[4]}"

    elif len(elements) == 6:
        # Six-element compounds (very complex minerals)
        if counts == [1, 1, 1, 1, 1, 1]:
            return "ABCDEF"
        elif counts == [1, 1, 2, 6, 1, 1]:  # Complex silicate-like
            return "ABC2D6EF"
        else:
            # For 6+ elements, use a more compact notation
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, ...
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)

    else:
        if len(elements) <= 10:
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, G, H, I, J
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)
        else:
            return "Complex"

SPACE_GROUP_SYMBOLS = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P21", 5: "C2", 6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc", 10: "P2/m",
    11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c", 15: "C2/c", 16: "P222", 17: "P2221", 18: "P21212", 19: "P212121", 20: "C2221",
    21: "C222", 22: "F222", 23: "I222", 24: "I212121", 25: "Pmm2", 26: "Pmc21", 27: "Pcc2", 28: "Pma2", 29: "Pca21", 30: "Pnc2",
    31: "Pmn21", 32: "Pba2", 33: "Pna21", 34: "Pnn2", 35: "Cmm2", 36: "Cmc21", 37: "Ccc2", 38: "Amm2", 39: "Aem2", 40: "Ama2",
    41: "Aea2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2", 45: "Iba2", 46: "Ima2", 47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban",
    51: "Pmma", 52: "Pnna", 53: "Pmna", 54: "Pcca", 55: "Pbam", 56: "Pccn", 57: "Pbcm", 58: "Pnnm", 59: "Pmmn", 60: "Pbcn",
    61: "Pbca", 62: "Pnma", 63: "Cmcm", 64: "Cmca", 65: "Cmmm", 66: "Cccm", 67: "Cmma", 68: "Ccca", 69: "Fmmm", 70: "Fddd",
    71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma", 75: "P4", 76: "P41", 77: "P42", 78: "P43", 79: "I4", 80: "I41",
    81: "P-4", 82: "I-4", 83: "P4/m", 84: "P42/m", 85: "P4/n", 86: "P42/n", 87: "I4/m", 88: "I41/a", 89: "P422", 90: "P4212",
    91: "P4122", 92: "P41212", 93: "P4222", 94: "P42212", 95: "P4322", 96: "P43212", 97: "I422", 98: "I4122", 99: "P4mm", 100: "P4bm",
    101: "P42cm", 102: "P42nm", 103: "P4cc", 104: "P4nc", 105: "P42mc", 106: "P42bc", 107: "P42mm", 108: "P42cm", 109: "I4mm", 110: "I4cm",
    111: "I41md", 112: "I41cd", 113: "P-42m", 114: "P-42c", 115: "P-421m", 116: "P-421c", 117: "P-4m2", 118: "P-4c2", 119: "P-4b2", 120: "P-4n2",
    121: "I-4m2", 122: "I-4c2", 123: "I-42m", 124: "I-42d", 125: "P4/mmm", 126: "P4/mcc", 127: "P4/nbm", 128: "P4/nnc", 129: "P4/mbm", 130: "P4/mnc",
    131: "P4/nmm", 132: "P4/ncc", 133: "P42/mmc", 134: "P42/mcm", 135: "P42/nbc", 136: "P42/mnm", 137: "P42/mbc", 138: "P42/mnm", 139: "I4/mmm", 140: "I4/mcm",
    141: "I41/amd", 142: "I41/acd", 143: "P3", 144: "P31", 145: "P32", 146: "R3", 147: "P-3", 148: "R-3", 149: "P312", 150: "P321",
    151: "P3112", 152: "P3121", 153: "P3212", 154: "P3221", 155: "R32", 156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c", 160: "R3m",
    161: "R3c", 162: "P-31m", 163: "P-31c", 164: "P-3m1", 165: "P-3c1", 166: "R-3m", 167: "R-3c", 168: "P6", 169: "P61", 170: "P65",
    171: "P62", 172: "P64", 173: "P63", 174: "P-6", 175: "P6/m", 176: "P63/m", 177: "P622", 178: "P6122", 179: "P6522", 180: "P6222",
    181: "P6422", 182: "P6322", 183: "P6mm", 184: "P6cc", 185: "P63cm", 186: "P63mc", 187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
    191: "P6/mmm", 192: "P6/mcc", 193: "P63/mcm", 194: "P63/mmc", 195: "P23", 196: "F23", 197: "I23", 198: "P213", 199: "I213", 200: "Pm-3",
    201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3", 205: "Pa-3", 206: "Ia-3", 207: "P432", 208: "P4232", 209: "F432", 210: "F4132",
    211: "I432", 212: "P4332", 213: "P4132", 214: "I4132", 215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c", 220: "I-43d",
    221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m", 225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c", 229: "Im-3m", 230: "Ia-3d"
}


ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"


MINERALS = {
    # Cubic structures
    225: {  # Fm-3m
        "Rock Salt (NaCl)": "Na Cl",
        "Fluorite (CaF2)": "Ca F2",
        "Anti-Fluorite (Li2O)": "Li2 O",
    },
    229: {  # Im-3m
        "BCC Iron": "Fe",
    },
    221: {  # Pm-3m
        "Perovskite (SrTiO3)": "Sr Ti O3",
        "ReO3 type": "Re O3",
        "Inverse-perovskite (Ca3TiN)": "Ca3 Ti N",
        "Cesium chloride (CsCl)": "Cs Cl"
    },
    227: {  # Fd-3m
        "Diamond": "C",

        "Normal spinel (MgAl2O4)": "Mg Al2 O4",
        "Inverse spinel (Fe3O4)": "Fe3 O4",
        "Pyrochlore (Ca2NbO7)": "Ca2 Nb2 O7",
        "β-Cristobalite (SiO2)": "Si O2"

    },
    216: {  # F-43m
        "Zinc Blende (ZnS)": "Zn S",
        "Half-anti-fluorite (Li4Ti)": "Li4 Ti"
    },
    215: {  # P-43m


    },
    230: {  # Ia-3d
        "Garnet (Ca3Al2Si3O12)": "Ca3 Al2 Si3 O12",
    },
    205: {  # Pa-3
        "Pyrite (FeS2)": "Fe S2",
    },
    224:{
        "Cuprite (Cu2O)": "Cu2 O",
    },
    # Hexagonal structures
    194: {  # P6_3/mmc
        "HCP Magnesium": "Mg",
        "Ni3Sn type": "Ni3 Sn",
        "Graphite": "C",
        "MoS2 type": "Mo S2",
        "Nickeline (NiAs)": "Ni As",
    },
    186: {  # P6_3mc
        "Wurtzite (ZnS)": "Zn S"
    },
    191: {  # P6/mmm


        "AlB2 type": "Al B2",
        "CaCu5 type": "Ca Cu5"
    },
    #187: {  # P-6m2
#
 #   },
    156: {
        "CdI2 type": "Cd I2",
    },
    164: {
    "CdI2 type": "Cd I2",
    },
    166: {  # R-3m
    "Delafossite (CuAlO2)": "Cu Al O2"
    },
    # Tetragonal structures
    139: {  # I4/mmm
        "β-Tin (Sn)": "Sn",
        "MoSi2 type": "Mo Si2"
    },
    136: {  # P4_2/mnm
        "Rutile (TiO2)": "Ti O2"
    },
    123: {  # P4/mmm
        "CuAu (L10)": "Cu Au"
    },
    141: {  # I41/amd
        "Anatase (TiO2)": "Ti O2",
        "Zircon (ZrSiO4)": "Zr Si O4"
    },
    122: {  # P-4m2
        "Chalcopyrite (CuFeS2)": "Cu Fe S2"
    },
    129: {  # P4/nmm
        "PbO structure": "Pb O"
    },

    # Orthorhombic structures
    62: {  # Pnma
        "Aragonite (CaCO3)": "Ca C O3",
        "Cotunnite (PbCl2)": "Pb Cl2",
        "Olivine (Mg2SiO4)": "Mg2 Si O4",
        "Barite (BaSO4)": "Ba S O4",
        "Perovskite (GdFeO3)": "Gd Fe O3"
    },
    63: {  # Cmcm
        "α-Uranium": "U",
        "CrB structure": "Cr B",
        "TlI structure": "Tl I",
    },
   # 74: {  # Imma
   #
   # },
    64: {  # Cmca
        "α-Gallium": "Ga"
    },

    # Monoclinic structures
    14: {  # P21/c
        "Baddeleyite (ZrO2)": "Zr O2",
        "Monazite (CePO4)": "Ce P O4"
    },
    206: {  # C2/m
        "Bixbyite (Mn2O3)": "Mn2 O3"
    },
    15: {  # C2/c
        "Gypsum (CaSO4·2H2O)": "Ca S H4 O6",
        "Scheelite (CaWO4)": "Ca W O4"
    },

    1: {
        "Kaolinite": "Al2 Si2 O9 H4"

    },
    # Triclinic structures
    2: {  # P-1
        "Wollastonite (CaSiO3)": "Ca Si O3",
        #"Kaolinite": "Al2 Si2 O5"
    },

    # Other important structures
    167: {  # R-3c
        "Calcite (CaCO3)": "Ca C O3",
        "Corundum (Al2O3)": "Al2 O3"
    },
    176: {  # P6_3/m
        "Apatite (Ca5(PO4)3OH)": "Ca5 P3 O13 H"
    },
    58: {  # Pnnm
        "Marcasite (FeS2)": "Fe S2"
    },
    198: {  # P213
        "FeSi structure": "Fe Si"
    },
    88: {  # I41/a
        "Scheelite (CaWO4)": "Ca W O4"
    },
    33: {  # Pna21
        "FeAs structure": "Fe As"
    },
    96: {  # P4/ncc
        "α-Cristobalite (SiO2)": "Si O2"
    },
    92: {
        "α-Cristobalite (SiO2)": "Si O2"
    },
    152: {  # P3121
        "Quartz (SiO2)": "Si O2"
    },
    148: {  # R-3
        "Ilmenite (FeTiO3)": "Fe Ti O3",
        "Dolomite (CaMgC2O6)": "Ca Mg C2 O6",
    },
    180: {  # P4_3 32
        "β-quartz (SiO2)": "Si O2"
    }
}


def identify_structure_type(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()

        formula = structure.composition.reduced_formula
        formula_type = get_formula_type(formula)
       # print("------")
        print(formula)
       # print(formula_type)
        #print(spg_number)
        if spg_number in STRUCTURE_TYPES and spg_number == 62 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Aragonite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==167 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
          #  print("YES")
          # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Calcite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==227 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "SiO2":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**β - Cristobalite (SiO2)**"
        elif formula == "C" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Graphite**"
        elif formula == "MoS2" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**MoS2 Type**"
        elif formula == "NiAs" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Nickeline (NiAs)**"
        elif formula == "ReO3" and spg_number in STRUCTURE_TYPES and spg_number ==221 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**ReO3 type**"
        elif formula == "TlI" and spg_number in STRUCTURE_TYPES and spg_number ==63 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**TlI structure**"
        elif spg_number in STRUCTURE_TYPES and formula_type in STRUCTURE_TYPES[
            spg_number]:
           # print("YES")
            structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**{structure_type}**"

        pearson = f"{crystal_system[0]}{structure.num_sites}"
        return f"**{crystal_system.capitalize()}** (Formula: {formula_type}, Pearson: {pearson})"

    except Exception as e:
        return f"Error identifying structure: {str(e)}"
STRUCTURE_TYPES = {
    # Cubic Structures
    225: {  # Fm-3m
        "A": "FCC (Face-centered cubic)",
        "AB": "Rock Salt (NaCl)",
        "AB2": "Fluorite (CaF2)",
        "A2B": "Anti-Fluorite",
        "AB3": "Cu3Au (L1₂)",
        "A3B": "AuCu3 type",
        "ABC": "Half-Heusler (C1b)",
        "AB6": "K2PtCl6 (cubic antifluorite)",
    },
    92: {
        "AB2": "α-Cristobalite (SiO2)"
    },
    229: {  # Im-3m
        "A": "BCC (Body-centered cubic)",
        "AB12": "NaZn13 type",
        "AB": "Tungsten carbide (WC)"
    },
    221: {  # Pm-3m
        "A": "Simple cubic (SC)",
        "AB": "Cesium Chloride (CsCl)",
        "ABC3": "Perovskite (Cubic, ABO3)",
        "AB3": "Cu3Au type",
        "A3B": "Cr3Si (A15)",
        #"AB6": "ReO3 type"
    },
    227: {  # Fd-3m
        "A": "Diamond cubic",

        "AB2": "Fluorite-like",
        "AB2C4": "Normal spinel",
        "A3B4": "Inverse spinel",
        "AB2C4": "Spinel",
        "A8B": "Gamma-brass",
        "AB2": "β - Cristobalite (SiO2)",
        "A2B2C7": "Pyrochlore"
    },
    55: {  # Pbca
        "AB2": "Brookite (TiO₂ polymorph)"
    },
    216: {  # F-43m
        "AB": "Zinc Blende (Sphalerite)",
        "A2B": "Antifluorite"
    },
    215: {  # P-43m
        "ABC3": "Inverse-perovskite",
        "AB4": "Half-anti-fluorite"
    },
    223: {  # Pm-3n
        "AB": "α-Mn structure",
        "A3B": "Cr3Si-type"
    },
    230: {  # Ia-3d
        "A3B2C1D4": "Garnet structure ((Ca,Mg,Fe)3(Al,Fe)2(SiO4)3)",
        "AB2": "Pyrochlore"
    },
    217: {  # I-43m
        "A12B": "α-Mn structure"
    },
    219: {  # F-43c
        "AB": "Sodium thallide"
    },
    205: {  # Pa-3
        "A2B": "Cuprite (Cu2O)",
        "AB6": "ReO3 structure",
        "AB2": "Pyrite (FeS2)",
    },
    156: {
        "AB2": "CdI2 type",
    },
    # Hexagonal Structures
    194: {  # P6_3/mmc
        "AB": "Wurtzite (high-T)",
        "AB2": "AlB2 type (hexagonal)",
        "A3B": "Ni3Sn type",
        "A3B": "DO19 structure (Ni3Sn-type)",
        "A": "Graphite (hexagonal)",
        "A": "HCP (Hexagonal close-packed)",
        #"AB2": "MoS2 type",
    },
    186: {  # P6_3mc
        "AB": "Wurtzite (ZnS)",
    },
    191: {  # P6/mmm


        "AB2": "AlB2 type",
        "AB5": "CaCu5 type",
        "A2B17": "Th2Ni17 type"
    },
    193: {  # P6_3/mcm
        "A3B": "Na3As structure",
        "ABC": "ZrBeSi structure"
    },
   # 187: {  # P-6m2
#
 #   },
    164: {  # P-3m1
        "AB2": "CdI2 type",
        "A": "Graphene layers"
    },
    166: {  # R-3m
        "A": "Rhombohedral",
        "A2B3": "α-Al2O3 type",
        "ABC2": "Delafossite (CuAlO2)"
    },
    160: {  # R3m
        "A2B3": "Binary tetradymite",
        "AB2": "Delafossite"
    },

    # Tetragonal Structures
    139: {  # I4/mmm
        "A": "Body-centered tetragonal",
        "AB": "β-Tin",
        "A2B": "MoSi2 type",
        "A3B": "Ni3Ti structure"
    },
    136: {  # P4_2/mnm
        "AB2": "Rutile (TiO2)"
    },
    123: {  # P4/mmm
        "AB": "γ-CuTi",
        "AB": "CuAu (L10)"
    },
    140: {  # I4/mcm
        "AB2": "Anatase (TiO2)",
        "A": "β-W structure"
    },
    141: {  # I41/amd
        "AB2": "Anatase (TiO₂)",
        "A": "α-Sn structure",
        "ABC4": "Zircon (ZrSiO₄)"
    },
    122: {  # P-4m2
        "ABC2": "Chalcopyrite (CuFeS2)"
    },
    129: {  # P4/nmm
        "AB": "PbO structure"
    },

    # Orthorhombic Structures
    62: {  # Pnma
        "ABC3": "Aragonite (CaCO₃)",
        "AB2": "Cotunnite (PbCl2)",
        "ABC3": "Perovskite (orthorhombic)",
        "A2B": "Fe2P type",
        "ABC3": "GdFeO3-type distorted perovskite",
        "A2BC4": "Olivine ((Mg,Fe)2SiO4)",
        "ABC4": "Barite (BaSO₄)"
    },
    63: {  # Cmcm
        "A": "α-U structure",
        "AB": "CrB structure",
        "AB2": "HgBr2 type"
    },
    74: {  # Imma
        "AB": "TlI structure",
    },
    64: {  # Cmca
        "A": "α-Ga structure"
    },
    65: {  # Cmmm
        "A2B": "η-Fe2C structure"
    },
    70: {  # Fddd
        "A": "Orthorhombic unit cell"
    },

    # Monoclinic Structures
    14: {  # P21/c
        "AB": "Monoclinic structure",
        "AB2": "Baddeleyite (ZrO2)",
        "ABC4": "Monazite (CePO4)"
    },
    12: {  # C2/m
        "A2B2C7": "Thortveitite (Sc2Si2O7)"
    },
    15: {  # C2/c
        "A1B4C1D6": "Gypsum (CaH4O6S)",
        "ABC6": "Gypsum (CaH4O6S)",
        "ABC4": "Scheelite (CaWO₄)",
        "ABC5": "Sphene (CaTiSiO₅)"
    },
    1: {
        "A2B2C4D9": "Kaolinite"
    },
    # Triclinic Structures
    2: {  # P-1
        "AB": "Triclinic structure",
        "ABC3": "Wollastonite (CaSiO3)",
    },

    # Other important structures
    99: {  # P4mm
        "ABCD3": "Tetragonal perovskite"
    },
    167: {  # R-3c
        "ABC3": "Calcite (CaCO3)",
        "A2B3": "Corundum (Al2O3)"
    },
    176: {  # P6_3/m
        "A10B6C2D31E": "Apatite (Ca10(PO4)6(OH)2)",
        "A5B3C1D13": "Apatite (Ca5(PO4)3OH",
        "A5B3C13": "Apatite (Ca5(PO4)3OH"
    },
    58: {  # Pnnm
        "AB2": "Marcasite (FeS2)"
    },
    11: {  # P21/m
        "A2B": "ThSi2 type"
    },
    72: {  # Ibam
        "AB2": "MoSi2 type"
    },
    198: {  # P213
        "AB": "FeSi structure",
        "A12": "β-Mn structure"
    },
    88: {  # I41/a
        "ABC4": "Scheelite (CaWO4)"
    },
    33: {  # Pna21
        "AB": "FeAs structure"
    },
    130: {  # P4/ncc
        "AB2": "Cristobalite (SiO2)"
    },
    152: {  # P3121
        "AB2": "Quartz (SiO2)"
    },
    200: {  # Pm-3
        "A3B3C": "Fe3W3C"
    },
    224: {  # Pn-3m
        "AB": "Pyrochlore-related",
        "A2B": "Cuprite (Cu2O)"
    },
    127: {  # P4/mbm
        "AB": "σ-phase structure",
        "AB5": "CaCu5 type"
    },
    148: {  # R-3
        "ABC3": "Calcite (CaCO₃)",
        "ABC3": "Ilmenite (FeTiO₃)",
        "ABCD3": "Dolomite",
    },
    69: {  # Fmmm
        "A": "β-W structure"
    },
    128: {  # P4/mnc
        "A3B": "Cr3Si (A15)"
    },
    206: {  # Ia-3
        "AB2": "Pyrite derivative",
        "AB2": "Pyrochlore (defective)",
        "A2B3": "Bixbyite"
    },
    212: {  # P4_3 32

        "A4B3": "Mn4Si3 type"
    },
    180: {
        "AB2": "β-quartz (SiO2)",
    },
    226: {  # Fm-3c
        "AB2": "BiF3 type"
    },
    196: {  # F23
        "AB2": "FeS2 type"
    },
    96: {
        "AB2": "α-Cristobalite (SiO2)"
    }

}
def get_cod_entries(params):
    try:
        response = requests.get('https://www.crystallography.net/cod/result', params=params)
        if response.status_code == 200:
            results = response.json()
            return results  # Returns a list of entries
        else:
            st.error(f"COD search error: {response.status_code}")
            return []
    except Exception as e:
        st.write(
            "Error during connection to COD database. Probably reason is that the COD database server is currently down.")

def sort_formula_alphabetically(formula_input):
    formula_parts = formula_input.strip().split()
    return " ".join(sorted(formula_parts))

def get_cif_from_cod(entry):
    file_url = entry.get('file')
    if file_url:
        response = requests.get(f"https://www.crystallography.net/cod/{file_url}.cif")
        if response.status_code == 200:
            return response.text
    return None


def get_structure_from_mp(mp_id):
    with MPRester(MP_API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
        return structure


from pymatgen.io.cif import CifParser


def get_structure_from_cif_url(cif_url):
    response = requests.get(f"https://www.crystallography.net/cod/{cif_url}.cif")
    if response.status_code == 200:
        #  writer = CifWriter(response.text, symprec=0.01)
        #  parser = CifParser.from_string(writer)
        #  structure = parser.get_structures(primitive=False)[0]
        return response.text
    else:
        raise ValueError(f"Failed to fetch CIF from URL: {cif_url}")


def get_cod_str(cif_content):
    parser = CifParser.from_str(cif_content)
    structure = parser.get_structures(primitive=False)[0]
    return structure


import concurrent.futures
import requests
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure

def check_structure_size_and_warn(structure, structure_name="structure"):
    n_atoms = len(structure)

    if n_atoms > 75:
        st.info(f"ℹ️ **Structure Notice**: {structure_name} contains a large number of **{n_atoms} atoms**. "
                f"Calculations may take longer depending on selected parameters. Please be careful to "
                f"not consume much memory, we are hosted on a free server. 😊")
        return "moderate"
    else:
        return "small"

def fetch_and_parse_cod_cif(entry):
    file_id = entry.get('file')
    if not file_id:
        return None, None, None, "Missing file ID in entry"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        cif_url = f"https://www.crystallography.net/cod/{file_id}.cif"
        response = requests.get(cif_url, timeout=15, headers=headers)
        response.raise_for_status()
        cif_content = response.text
        parser = CifParser.from_str(cif_content)

        structure = parser.get_structures(primitive=False)[0]
        cod_id = f"cod_{file_id}"
        return cod_id, structure, entry, None

    except Exception as e:
        return None, None, None, str(e)

def databases(show_database_search = False):


    def get_space_group_info(number):
        symbol = SPACE_GROUP_SYMBOLS.get(number, f"SG#{number}")
        return symbol

    if show_database_search:
        with st.expander("Search for Structures Online in Databases", icon="🔍", expanded=True):
            cols, cols2, cols3 = st.columns([1.5, 1.5, 3.5])
            with cols:
                db_choices = st.multiselect(
                    "Select Database(s)",
                    options=["Materials Project", "AFLOW", "COD"],
                    default=["Materials Project", "AFLOW", "COD"],
                    help="Choose which databases to search for structures. You can select multiple databases."
                )

                if not db_choices:
                    st.warning("Please select at least one database to search.")

                st.markdown(
                    "**Maximum number of structures to be found in each database (for improving performance):**")
                col_limits = st.columns(3)

                search_limits = {}
                if "Materials Project" in db_choices:
                    with col_limits[0]:
                        search_limits["Materials Project"] = st.number_input(
                            "MP Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from Materials Project"
                        )
                if "AFLOW" in db_choices:
                    with col_limits[1]:
                        search_limits["AFLOW"] = st.number_input(
                            "AFLOW Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from AFLOW"
                        )
                if "COD" in db_choices:
                    with col_limits[2]:
                        search_limits["COD"] = st.number_input(
                            "COD Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from COD"
                        )

            with cols2:
                search_mode = st.radio(
                    "Search by:",
                    options=["Elements", "Structure ID", "Space Group + Elements", "Formula", "Search Mineral"],
                    help="Choose your search strategy"
                )

                if search_mode == "Elements":
                    selected_elements = st.multiselect(
                        "Select elements for search:",
                        options=ELEMENTS,
                        default=["Sr", "Ti", "O"],
                        help="Choose one or more chemical elements"
                    )
                    search_query = " ".join(selected_elements) if selected_elements else ""

                elif search_mode == "Structure ID":
                    structure_ids = st.text_area(
                        "Enter Structure IDs (one per line):",
                        value="mp-5229\ncod_1512124\naflow:010158cb2b41a1a5",
                        help="Enter structure IDs. Examples:\n- Materials Project: mp-5229\n- COD: cod_1512124 (with cod_ prefix)\n- AFLOW: aflow:010158cb2b41a1a5 (AUID format)"
                    )

                elif search_mode == "Space Group + Elements":
                    col_sg1, col_sg2 = st.columns(2)
                    with col_sg1:
                        all_space_groups_help = "Enter space group number (1-230)\n\nAll space groups:\n\n"
                        for num in sorted(SPACE_GROUP_SYMBOLS.keys()):
                            all_space_groups_help += f"• {num}: {SPACE_GROUP_SYMBOLS[num]}\n\n"

                        space_group_number = st.number_input(
                            "Space Group Number:",
                            min_value=1,
                            max_value=230,
                            value=221,
                            help=all_space_groups_help
                        )
                        sg_symbol = get_space_group_info(space_group_number)
                        st.info(f"#:**{sg_symbol}**")

                    selected_elements = st.multiselect(
                        "Select elements for search:",
                        options=ELEMENTS,
                        default=["Sr", "Ti", "O"],
                        help="Choose one or more chemical elements"
                    )

                elif search_mode == "Formula":
                    formula_input = st.text_input(
                        "Enter Chemical Formula:",
                        value="Sr Ti O3",
                        help="Enter chemical formula with spaces between elements. Examples:\n- Sr Ti O3 (strontium titanate)\n- Ca C O3 (calcium carbonate)\n- Al2 O3 (alumina)"
                    )

                elif search_mode == "Search Mineral":
                    mineral_options = []
                    mineral_mapping = {}

                    for space_group, minerals in MINERALS.items():
                        for mineral_name, formula in minerals.items():
                            option_text = f"{mineral_name} - SG #{space_group}"
                            mineral_options.append(option_text)
                            mineral_mapping[option_text] = {
                                'space_group': space_group,
                                'formula': formula,
                                'mineral_name': mineral_name
                            }

                    # Sort mineral options alphabetically
                    mineral_options.sort()

                    selected_mineral = st.selectbox(
                        "Select Mineral Structure:",
                        options=mineral_options,
                        help="Choose a mineral structure type. The exact formula and space group will be automatically set.",
                        index=2
                    )

                    if selected_mineral:
                        mineral_info = mineral_mapping[selected_mineral]

                        # col_mineral1, col_mineral2 = st.columns(2)
                        # with col_mineral1:
                        sg_symbol = get_space_group_info(mineral_info['space_group'])
                        st.info(
                            f"**Structure:** {mineral_info['mineral_name']}, **Space Group:** {mineral_info['space_group']} ({sg_symbol}), "
                            f"**Formula:** {mineral_info['formula']}")

                        space_group_number = mineral_info['space_group']
                        formula_input = mineral_info['formula']

                        st.success(
                            f"**Search will use:** Formula = {formula_input}, Space Group = {space_group_number}")

                show_element_info = st.checkbox("ℹ️ Show information about element groups")
                if show_element_info:
                    st.markdown("""
                    **Element groups note:**
                    **Common Elements (14):** H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca  
                    **Transition Metals (10):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn  
                    **Alkali Metals (6):** Li, Na, K, Rb, Cs, Fr  
                    **Alkaline Earth (6):** Be, Mg, Ca, Sr, Ba, Ra  
                    **Noble Gases (6):** He, Ne, Ar, Kr, Xe, Rn  
                    **Halogens (5):** F, Cl, Br, I, At  
                    **Lanthanides (15):** La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu  
                    **Actinides (15):** Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr  
                    **Other Elements (51):** All remaining elements
                    """)

            if st.button("Search Selected Databases"):
                if not db_choices:
                    st.error("Please select at least one database to search.")
                else:
                    for db_choice in db_choices:
                        if db_choice == "Materials Project":
                            mp_limit = search_limits.get("Materials Project", 50)
                            with st.spinner(f"Searching **the MP database** (limit: {mp_limit}), please wait. 😊"):
                                try:
                                    with MPRester(MP_API_KEY) as mpr:
                                        docs = None

                                        if search_mode == "Elements":
                                            elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                            if not elements_list:
                                                st.error("Please enter at least one element for the search.")
                                                continue
                                            elements_list_sorted = sorted(set(elements_list))
                                            docs = mpr.materials.summary.search(
                                                elements=elements_list_sorted,
                                                num_elements=len(elements_list_sorted),
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Structure ID":
                                            mp_ids = [id.strip() for id in structure_ids.split('\n')
                                                      if id.strip() and id.strip().startswith('mp-')]
                                            if not mp_ids:
                                                st.warning(
                                                    "No valid Materials Project IDs found (should start with 'mp-')")
                                                continue
                                            docs = mpr.materials.summary.search(
                                                material_ids=mp_ids,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Space Group + Elements":
                                            elements_list = sorted(set(selected_elements))
                                            if not elements_list:
                                                st.warning(
                                                    "Please select elements for Materials Project space group search.")
                                                continue

                                            search_params = {
                                                "elements": elements_list,
                                                "num_elements": len(elements_list),
                                                "fields": ["material_id", "formula_pretty", "symmetry", "nsites",
                                                           "volume"],
                                                "spacegroup_number": space_group_number
                                            }

                                            docs = mpr.materials.summary.search(**search_params)

                                        elif search_mode == "Formula":
                                            if not formula_input.strip():
                                                st.warning(
                                                    "Please enter a chemical formula for Materials Project search.")
                                                continue

                                            # Convert space-separated format to compact format (Sr Ti O3 -> SrTiO3)
                                            clean_formula = formula_input.strip()
                                            if ' ' in clean_formula:
                                                parts = clean_formula.split()
                                                compact_formula = ''.join(parts)
                                            else:
                                                compact_formula = clean_formula

                                            docs = mpr.materials.summary.search(
                                                formula=compact_formula,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Search Mineral":
                                            if not selected_mineral:
                                                st.warning(
                                                    "Please select a mineral structure for Materials Project search.")
                                                continue
                                            clean_formula = formula_input.strip()
                                            if ' ' in clean_formula:
                                                parts = clean_formula.split()
                                                compact_formula = ''.join(parts)
                                            else:
                                                compact_formula = clean_formula

                                            # Search by formula and space group
                                            docs = mpr.materials.summary.search(
                                                formula=compact_formula,
                                                spacegroup_number=space_group_number,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        if docs:
                                            status_placeholder = st.empty()
                                            st.session_state.mp_options = []
                                            st.session_state.full_structures_see = {}
                                            limited_docs = docs[:mp_limit]

                                            for doc in limited_docs:
                                                full_structure = mpr.get_structure_by_material_id(doc.material_id,
                                                                                                  conventional_unit_cell=True)
                                                st.session_state.full_structures_see[doc.material_id] = full_structure
                                                lattice = full_structure.lattice
                                                leng = len(full_structure)
                                                lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                                               f"{lattice.alpha:.1f}, {lattice.beta:.1f}, {lattice.gamma:.1f} °")
                                                st.session_state.mp_options.append(
                                                    f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol} #{doc.symmetry.number}) [{lattice_str}], {float(doc.volume):.1f} Å³, {leng} atoms"
                                                )
                                                status_placeholder.markdown(
                                                    f"- **Structure loaded:** `{full_structure.composition.reduced_formula}` ({doc.material_id})"
                                                )
                                            if len(limited_docs) < len(docs):
                                                st.info(
                                                    f"Showing first {mp_limit} of {len(docs)} total Materials Project results. Increase limit to see more.")
                                            st.success(
                                                f"Found {len(st.session_state.mp_options)} structures in Materials Project.")
                                        else:
                                            st.session_state.mp_options = []
                                            st.warning("No matching structures found in Materials Project.")
                                except Exception as e:
                                    st.error(f"An error occurred with Materials Project: {e}")

                        elif db_choice == "AFLOW":
                            aflow_limit = search_limits.get("AFLOW", 50)
                            with st.spinner(f"Searching **the AFLOW database** (limit: {aflow_limit}), please wait. 😊"):
                                try:
                                    results = []

                                    if search_mode == "Elements":
                                        elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                        if not elements_list:
                                            st.warning("Please enter elements for AFLOW search.")
                                            continue
                                        ordered_elements = sorted(elements_list)
                                        ordered_str = ",".join(ordered_elements)
                                        aflow_nspecies = len(ordered_elements)

                                        results = list(
                                            search(catalog="icsd")
                                            .filter(
                                                (AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                                            .select(
                                                AFLOW_K.auid,
                                                AFLOW_K.compound,
                                                AFLOW_K.geometry,
                                                AFLOW_K.spacegroup_relax,
                                                AFLOW_K.aurl,
                                                AFLOW_K.files,
                                            )
                                        )

                                    elif search_mode == "Structure ID":
                                        aflow_auids = []
                                        for id_line in structure_ids.split('\n'):
                                            id_line = id_line.strip()
                                            if id_line.startswith('aflow:'):
                                                auid = id_line.replace('aflow:', '').strip()
                                                aflow_auids.append(auid)

                                        if not aflow_auids:
                                            st.warning("No valid AFLOW AUIDs found (should start with 'aflow:')")
                                            continue

                                        results = []
                                        for auid in aflow_auids:
                                            try:
                                                result = list(search(catalog="icsd")
                                                              .filter(AFLOW_K.auid == f"aflow:{auid}")
                                                              .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                      AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                      AFLOW_K.files))
                                                results.extend(result)
                                            except Exception as e:
                                                st.warning(f"AFLOW search failed for AUID '{auid}': {e}")
                                                continue

                                    elif search_mode == "Space Group + Elements":
                                        if not selected_elements:
                                            st.warning("Please select elements for AFLOW space group search.")
                                            continue
                                        ordered_elements = sorted(selected_elements)
                                        ordered_str = ",".join(ordered_elements)
                                        aflow_nspecies = len(ordered_elements)

                                        try:
                                            results = list(search(catalog="icsd")
                                                           .filter((AFLOW_K.species % ordered_str) &
                                                                   (AFLOW_K.nspecies == aflow_nspecies) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))
                                        except Exception as e:
                                            st.warning(f"AFLOW space group search failed: {e}")
                                            results = []


                                    elif search_mode == "Formula":

                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for AFLOW search.")

                                            continue

                                        def convert_to_aflow_formula(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                                if match:
                                                    element = match.group(1)

                                                    count = match.group(2) if match.group(
                                                        2) else "1"  # Add "1" if no number

                                                    elements_dict[element] = count

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        # Generate 2x multiplied formula
                                        def multiply_formula_by_2(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                                if match:
                                                    element = match.group(1)

                                                    count = int(match.group(2)) if match.group(2) else 1

                                                    elements_dict[element] = str(count * 2)  # Multiply by 2

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        aflow_formula = convert_to_aflow_formula(formula_input)

                                        aflow_formula_2x = multiply_formula_by_2(formula_input)

                                        if aflow_formula_2x != aflow_formula:

                                            results = list(search(catalog="icsd")

                                                           .filter((AFLOW_K.compound == aflow_formula) |

                                                                   (AFLOW_K.compound == aflow_formula_2x))

                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,

                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching for both {aflow_formula} and {aflow_formula_2x} formulas simultaneously")

                                        else:
                                            results = list(search(catalog="icsd")
                                                           .filter(AFLOW_K.compound == aflow_formula)
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(f"Searching for formula {aflow_formula}")


                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning("Please select a mineral structure for AFLOW search.")
                                            continue

                                        def convert_to_aflow_formula_mineral(formula_input):
                                            import re
                                            formula_parts = formula_input.strip().split()
                                            elements_dict = {}
                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                                if match:
                                                    element = match.group(1)

                                                    count = match.group(2) if match.group(
                                                        2) else "1"  # Always add "1" for single atoms

                                                    elements_dict[element] = count

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        def multiply_mineral_formula_by_2(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:
                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                                if match:
                                                    element = match.group(1)
                                                    count = int(match.group(2)) if match.group(2) else 1
                                                    elements_dict[element] = str(count * 2)  # Multiply by 2
                                            aflow_parts = []
                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")
                                            return "".join(aflow_parts)

                                        aflow_formula = convert_to_aflow_formula_mineral(formula_input)

                                        aflow_formula_2x = multiply_mineral_formula_by_2(formula_input)

                                        # Search for both formulas with space group constraint in a single query

                                        if aflow_formula_2x != aflow_formula:
                                            results = list(search(catalog="icsd")
                                                           .filter(((AFLOW_K.compound == aflow_formula) |
                                                                    (AFLOW_K.compound == aflow_formula_2x)) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching {mineral_info['mineral_name']} for both {aflow_formula} and {aflow_formula_2x} with space group {space_group_number}")

                                        else:
                                            results = list(search(catalog="icsd")
                                                           .filter((AFLOW_K.compound == aflow_formula) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching {mineral_info['mineral_name']} for formula {aflow_formula} with space group {space_group_number}")

                                    if results:
                                        status_placeholder = st.empty()
                                        st.session_state.aflow_options = []
                                        st.session_state.entrys = {}

                                        limited_results = results[:aflow_limit]

                                        for entry in limited_results:
                                            st.session_state.entrys[entry.auid] = entry
                                            st.session_state.aflow_options.append(
                                                f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax}) {entry.geometry}"
                                            )
                                            status_placeholder.markdown(
                                                f"- **Structure loaded:** `{entry.compound}` (aflow_{entry.auid})"
                                            )
                                        if len(limited_results) < len(results):
                                            st.info(
                                                f"Showing first {aflow_limit} of {len(results)} total AFLOW results. Increase limit to see more.")
                                        st.success(f"Found {len(st.session_state.aflow_options)} structures in AFLOW.")
                                    else:
                                        st.session_state.aflow_options = []
                                        st.warning("No matching structures found in AFLOW.")
                                except Exception as e:
                                    st.warning(f"No matching structures found in AFLOW.")
                                    st.session_state.aflow_options = []

                        elif db_choice == "COD":
                            cod_limit = search_limits.get("COD", 50)
                            with st.spinner(f"Searching **the COD database** (limit: {cod_limit}), please wait. 😊"):
                                try:
                                    cod_entries = []

                                    if search_mode == "Elements":
                                        elements = [el.strip() for el in search_query.split() if el.strip()]
                                        if elements:
                                            params = {'format': 'json', 'detail': '1'}
                                            for i, el in enumerate(elements, start=1):
                                                params[f'el{i}'] = el
                                            params['strictmin'] = str(len(elements))
                                            params['strictmax'] = str(len(elements))
                                            cod_entries = get_cod_entries(params)
                                        else:
                                            st.warning("Please enter elements for COD search.")
                                            continue

                                    elif search_mode == "Structure ID":
                                        cod_ids = []
                                        for id_line in structure_ids.split('\n'):
                                            id_line = id_line.strip()
                                            if id_line.startswith('cod_'):
                                                # Extract numeric ID from cod_XXXXX format
                                                numeric_id = id_line.replace('cod_', '').strip()
                                                if numeric_id.isdigit():
                                                    cod_ids.append(numeric_id)

                                        if not cod_ids:
                                            st.warning(
                                                "No valid COD IDs found (should start with 'cod_' followed by numbers)")
                                            continue

                                        cod_entries = []
                                        for cod_id in cod_ids:
                                            try:
                                                params = {'format': 'json', 'detail': '1', 'id': cod_id}
                                                entry = get_cod_entries(params)
                                                if entry:
                                                    if isinstance(entry, list):
                                                        cod_entries.extend(entry)
                                                    else:
                                                        cod_entries.append(entry)
                                            except Exception as e:
                                                st.warning(f"COD search failed for ID {cod_id}: {e}")
                                                continue

                                    elif search_mode == "Space Group + Elements":
                                        elements = selected_elements
                                        if elements:
                                            params = {'format': 'json', 'detail': '1'}
                                            for i, el in enumerate(elements, start=1):
                                                params[f'el{i}'] = el
                                            params['strictmin'] = str(len(elements))
                                            params['strictmax'] = str(len(elements))
                                            params['space_group_number'] = str(space_group_number)

                                            cod_entries = get_cod_entries(params)
                                        else:
                                            st.warning("Please select elements for COD space group search.")
                                            continue

                                    elif search_mode == "Formula":
                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for COD search.")
                                            continue

                                        # alphabet sorting
                                        alphabet_form = sort_formula_alphabetically(formula_input)
                                        print(alphabet_form)
                                        params = {'format': 'json', 'detail': '1', 'formula': alphabet_form}
                                        cod_entries = get_cod_entries(params)

                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning("Please select a mineral structure for COD search.")
                                            continue

                                        # Use both formula and space group for COD search
                                        alphabet_form = sort_formula_alphabetically(formula_input)
                                        params = {
                                            'format': 'json',
                                            'detail': '1',
                                            'formula': alphabet_form,
                                            'space_group_number': str(space_group_number)
                                        }
                                        cod_entries = get_cod_entries(params)

                                    if cod_entries and isinstance(cod_entries, list):
                                        st.session_state.cod_options = []
                                        st.session_state.full_structures_see_cod = {}
                                        status_placeholder = st.empty()
                                        limited_entries = cod_entries[:cod_limit]
                                        errors = []

                                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                            future_to_entry = {executor.submit(fetch_and_parse_cod_cif, entry): entry
                                                               for
                                                               entry in limited_entries}

                                            processed_count = 0
                                            for future in concurrent.futures.as_completed(future_to_entry):
                                                processed_count += 1
                                                status_placeholder.markdown(
                                                    f"- **Processing:** {processed_count}/{len(limited_entries)} entries...")
                                                try:
                                                    cod_id, structure, entry_data, error = future.result()
                                                    if error:
                                                        original_entry = future_to_entry[future]
                                                        errors.append(
                                                            f"Entry `{original_entry.get('file', 'N/A')}` failed: {error}")
                                                        continue  # Skip to the next completed future
                                                    if cod_id and structure and entry_data:
                                                        st.session_state.full_structures_see_cod[cod_id] = structure

                                                        spcs = entry_data.get("sg", "Unknown")
                                                        spcs_number = entry_data.get("sgNumber", "Unknown")
                                                        cell_volume = structure.lattice.volume
                                                        option_str = (
                                                            f"{cod_id}: {structure.composition.reduced_formula} ({spcs} #{spcs_number}) [{structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} Å, {structure.lattice.alpha:.2f}, "
                                                            f"{structure.lattice.beta:.2f}, {structure.lattice.gamma:.2f}°], {cell_volume:.1f} Å³, {len(structure)} atoms"
                                                        )
                                                        st.session_state.cod_options.append(option_str)

                                                except Exception as e:
                                                    errors.append(
                                                        f"A critical error occurred while processing a result: {e}")
                                        status_placeholder.empty()
                                        if st.session_state.cod_options:
                                            if len(limited_entries) < len(cod_entries):
                                                st.info(
                                                    f"Showing first {cod_limit} of {len(cod_entries)} total COD results. Increase limit to see more.")
                                            st.success(
                                                f"Found and processed {len(st.session_state.cod_options)} structures from COD.")
                                        else:
                                            st.warning("COD: No matching structures could be successfully processed.")
                                        if errors:
                                            st.error(f"Encountered {len(errors)} error(s) during the search.")
                                            with st.container(border=True):
                                                for e in errors:
                                                    st.warning(e)
                                    else:
                                        st.session_state.cod_options = []
                                        st.warning("COD: No matching structures found.")
                                except Exception as e:
                                    st.warning(f"COD search error: {e}")
                                    st.session_state.cod_options = []

            # with cols2:
            #     image = Image.open("images/Rabbit2.png")
            #     st.image(image, use_container_width=True)

            with cols3:
                if any(x in st.session_state for x in ['mp_options', 'aflow_options', 'cod_options']):
                    tabs = []
                    if 'mp_options' in st.session_state and st.session_state.mp_options:
                        tabs.append("Materials Project")
                    if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                        tabs.append("AFLOW")
                    if 'cod_options' in st.session_state and st.session_state.cod_options:
                        tabs.append("COD")

                    if tabs:
                        selected_tab = st.tabs(tabs)

                        tab_index = 0
                        if 'mp_options' in st.session_state and st.session_state.mp_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in Materials Project")
                                selected_structure = st.selectbox("Select a structure from MP:",
                                                                  st.session_state.mp_options)
                                selected_id = selected_structure.split(":")[0].strip()
                                composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
                                file_name = f"{selected_id}_{composition}.cif"
                                file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

                                if selected_id in st.session_state.full_structures_see:
                                    selected_entry = st.session_state.full_structures_see[selected_id]

                                    conv_lattice = selected_entry.lattice
                                    cell_volume = selected_entry.lattice.volume
                                    density = str(selected_entry.density).split()[0]
                                    n_atoms = len(selected_entry)
                                    atomic_den = n_atoms / cell_volume

                                    structure_type = identify_structure_type(selected_entry)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(selected_entry)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                    st.write(
                                        f"**Material ID:** {selected_id}, **Formula:** {composition}, N. of Atoms {n_atoms}")

                                    st.write(
                                        f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    mp_url = f"https://materialsproject.org/materials/{selected_id}"
                                    st.write(f"**Link:** {mp_url}")

                                    col_mpd, col_mpb = st.columns([2, 1])
                                    with col_mpd:
                                        if st.button("Add Selected Structure (MP)", key="add_btn_mp"):
                                            pmg_structure = st.session_state.full_structures_see[selected_id]
                                            check_structure_size_and_warn(pmg_structure, f"MP structure {selected_id}")
                                            st.session_state.full_structures[file_name] = pmg_structure
                                            cif_writer = CifWriter(pmg_structure)
                                            cif_content = cif_writer.__str__()
                                            cif_file = io.BytesIO(cif_content.encode('utf-8'))
                                            cif_file.name = file_name
                                            if 'uploaded_files' not in st.session_state:
                                                st.session_state.uploaded_files = []
                                            if all(f.name != file_name for f in st.session_state.uploaded_files):
                                                st.session_state.uploaded_files.append(cif_file)
                                            st.success("Structure added from Materials Project!")
                                    with col_mpb:
                                        st.download_button(
                                            label="Download MP CIF",
                                            data=str(
                                                CifWriter(st.session_state.full_structures_see[selected_id],
                                                          symprec=0.01)),
                                            file_name=file_name,
                                            type="primary",
                                            mime="chemical/x-cif"
                                        )
                                        st.info(
                                            f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                            tab_index += 1

                        if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in AFLOW")
                                st.warning(
                                    "The AFLOW does not provide atomic occupancies and includes only information about primitive cell in API. For better performance, volume and n. of atoms are purposely omitted from the expander.")
                                selected_structure = st.selectbox("Select a structure from AFLOW:",
                                                                  st.session_state.aflow_options)
                                selected_auid = selected_structure.split(": ")[0].strip()
                                selected_entry = next(
                                    (entry for entry in st.session_state.entrys.values() if
                                     entry.auid == selected_auid),
                                    None)
                                if selected_entry:

                                    cif_files = [f for f in selected_entry.files if
                                                 f.endswith("_sprim.cif") or f.endswith(".cif")]

                                    if cif_files:

                                        cif_filename = cif_files[0]

                                        # Correct the AURL: replace the first ':' with '/'

                                        host_part, path_part = selected_entry.aurl.split(":", 1)

                                        corrected_aurl = f"{host_part}/{path_part}"

                                        file_url = f"http://{corrected_aurl}/{cif_filename}"
                                        response = requests.get(file_url)
                                        cif_content = response.content

                                        structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'),
                                                                                  fmt="cif")
                                        converted_structure = get_full_conventional_structure(structure_from_aflow,
                                                                                              symprec=0.1)

                                        conv_lattice = converted_structure.lattice
                                        cell_volume = converted_structure.lattice.volume
                                        density = str(converted_structure.density).split()[0]
                                        n_atoms = len(converted_structure)
                                        atomic_den = n_atoms / cell_volume

                                        structure_type = identify_structure_type(converted_structure)
                                        st.write(f"**Structure type:** {structure_type}")
                                        analyzer = SpacegroupAnalyzer(structure_from_aflow)
                                        st.write(
                                            f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
                                        st.write(
                                            f"**AUID:** {selected_entry.auid}, **Formula:** {selected_entry.compound}, **N. of Atoms:** {n_atoms}")
                                        st.write(
                                            f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, "
                                            f"γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                        st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                        linnk = f"https://aflowlib.duke.edu/search/ui/material/?id=" + selected_entry.auid
                                        st.write("**Link:**", linnk)

                                        if st.button("Add Selected Structure (AFLOW)", key="add_btn_aflow"):
                                            if 'uploaded_files' not in st.session_state:
                                                st.session_state.uploaded_files = []
                                            cif_file = io.BytesIO(cif_content)
                                            cif_file.name = f"{selected_entry.compound}_{selected_entry.auid}.cif"

                                            st.session_state.full_structures[cif_file.name] = structure_from_aflow

                                            check_structure_size_and_warn(structure_from_aflow, cif_file.name)
                                            if all(f.name != cif_file.name for f in st.session_state.uploaded_files):
                                                st.session_state.uploaded_files.append(cif_file)
                                            st.success("Structure added from AFLOW!")

                                        st.download_button(
                                            label="Download AFLOW CIF",
                                            data=cif_content,
                                            file_name=f"{selected_entry.compound}_{selected_entry.auid}.cif",
                                            type="primary",
                                            mime="chemical/x-cif"
                                        )
                                        st.info(
                                            f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                                    else:
                                        st.warning("No CIF file found for this AFLOW entry.")
                            tab_index += 1

                        # COD tab
                        if 'cod_options' in st.session_state and st.session_state.cod_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in COD")
                                selected_cod_structure = st.selectbox(
                                    "Select a structure from COD:",
                                    st.session_state.cod_options,
                                    key='sidebar_select_cod'
                                )
                                cod_id = selected_cod_structure.split(":")[0].strip()
                                if cod_id in st.session_state.full_structures_see_cod:
                                    selected_entry = st.session_state.full_structures_see_cod[cod_id]
                                    lattice = selected_entry.lattice
                                    cell_volume = selected_entry.lattice.volume
                                    density = str(selected_entry.density).split()[0]
                                    n_atoms = len(selected_entry)
                                    atomic_den = n_atoms / cell_volume

                                    idcodd = cod_id.removeprefix("cod_")

                                    structure_type = identify_structure_type(selected_entry)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(selected_entry)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                    st.write(
                                        f"**COD ID:** {idcodd}, **Formula:** {selected_entry.composition.reduced_formula}, **N. of Atoms:** {n_atoms}")
                                    st.write(
                                        f"**Conventional Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    cod_url = f"https://www.crystallography.net/cod/{cod_id.split('_')[1]}.html"
                                    st.write(f"**Link:** {cod_url}")

                                    file_name = f"{selected_entry.composition.reduced_formula}_COD_{cod_id.split('_')[1]}.cif"

                                    if st.button("Add Selected Structure (COD)", key="sid_add_btn_cod"):
                                        cif_writer = CifWriter(selected_entry, symprec=0.01)
                                        cif_data = str(cif_writer)
                                        st.session_state.full_structures[file_name] = selected_entry
                                        cif_file = io.BytesIO(cif_data.encode('utf-8'))
                                        cif_file.name = file_name
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        if all(f.name != file_name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)

                                        check_structure_size_and_warn(selected_entry, file_name)
                                        st.success("Structure added from COD!")

                                    st.download_button(
                                        label="Download COD CIF",
                                        data=str(CifWriter(selected_entry, symprec=0.01)),
                                        file_name=file_name,
                                        mime="chemical/x-cif", type="primary",
                                    )
                                    st.info(
                                        f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
