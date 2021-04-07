# Modified from https://github.com/wengong-jin/iclr19-graph2graph
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import networkx as nx
import sascorer
import drd2_scorer
from tdc import Oracle

def similarity(a, b, sim_type):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    if sim_type == "binary":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    else:
        fp1 = AllChem.GetMorganFingerprint(amol, 2, useChirality=False)
        fp2 = AllChem.GetMorganFingerprint(bmol, 2, useChirality=False)

    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return sim

def get_prop(s, prop="logp"):
    if prop == "logp":
        return penalized_logp(s)
    elif prop == "drd2":
        return drd2(s)
    elif prop == "qed":
        return qed(s)
    elif prop == "gsk":
        return gsk(s)
    else:
        raise ValueError("unsupported properties")

def drd2(s):
    if s is None: return 0.0
    if Chem.MolFromSmiles(s) is None:
        return 0.0
    return drd2_scorer.get_score(s)

def qed(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return 0.0
    return QED.qed(mol)



from tdc import Oracle
oracle_gsk = None

def gsk(s):
    if oracle_gsk is None: oracle_gsk = Oracle(name = 'GSK3B')

    if s is None: return 0.0
    return oracle_gsk(s)    

# Modified from https://github.com/bowenliu16/rl_graph_generation
def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)
    
    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    
    return normalized_log_p + normalized_SA + normalized_cycle


def analyze_penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)
    
    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    
    prop = normalized_log_p + normalized_SA + normalized_cycle
    return prop, log_p, SA, cycle_score, normalized_log_p, normalized_SA, normalized_cycle


def smiles2D(s):
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol)

if __name__ == "__main__":
    print(round(penalized_logp('N#Cc1ccc2c(c1)OCCOCCOCCOc1ccc(C#N)cc1OCCOCCOCCO2'), 2), 5.30)
