import pdb
import argparse
import math
from properties import similarity, drd2, qed, penalized_logp

parser = argparse.ArgumentParser()
parser.add_argument('--prop', required=True)
parser.add_argument('--data_path', required=True)

args = parser.parse_args()
data_file = open(args.data_path, 'r')
pair_smiles = data_file.readlines()

new_data_path = args.data_path.split(".txt")[0] + "_with_prop.txt"
new_data_file = open(new_data_path, 'w')

sims = []
props = []
for pair in pair_smiles:
    smiles = pair.rstrip().split(" ")
    smile1 = smiles[0]
    smile2 = smiles[1]
    
    sim = similarity(smile1, smile2)
    if args.prop == "logp":
        prop1 = penalized_logp(smile1)
        prop2 = penalized_logp(smile2)
    elif args.prop == "qed":
        prop1 = qed(smile1)
        prop2 = qed(smile2)
    elif args.prop == "drd2":
        prop1 = drd2(smile1)
        prop2 = drd2(smile2)
    else:
        raise ValueError("wrong argument; should be (logp, qed, drd2)")
    sims.append(sim)
    props.append([prop1, prop2])

    new_data_file.write("%s %s %.2f %.4f %.4f\n" % (smile1, smile2, sim, prop1, prop2))

new_data_file.close()

avg_sim = sum(sims) / len(sims)
imps = [p[1]-p[0] for p in props]
avg_imp = sum(imps) / len(props)

std_sim = math.sqrt(sum([(sim-avg_sim) ** 2 for sim in sims]))
std_imp = math.sqrt(sum([(imp-avg_imp) ** 2 for imp in imps]))

print("avg_sim: %.4f(%.4f); avg_imp: %.4f(%.4f)\n" % (avg_sim, std_sim, avg_imp, std_imp))
print("max_sim: %.4f; max_imp: %.4f\n" % (max(sims), max(imp)))
print("min_sim: %.4f; min_imp: %.4f\n" % (min(sims), min(imp)))
