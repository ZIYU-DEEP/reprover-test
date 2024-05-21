from lean_dojo import *
import lean_dojo
import json
from pathlib import Path

DST_DIR = Path("./minif2f_dataset")
train_path = DST_DIR / "random/train.json"
proofs_train = json.load(train_path.open())
print(len(proofs_train))
print(type(proofs_train))
collect_minif2f_valid = []
collect_minif2f_test = []
for proof in proofs_train[::-1]:
    if proof["traced_tactics"] != []:
        if 'MiniF2F/Valid' in proof['file_path']:
            collect_minif2f_valid.append(proof)
        if 'MiniF2F/Test' in proof['file_path']:
            collect_minif2f_test.append(proof)
        if proof['full_name'] == 'mathd_algebra_478':
            print('!!!')
            print(proof)

print(len(collect_minif2f_valid))
print(len(collect_minif2f_test))

DST_DIR = Path("./minif2f_dataset")
train_path = DST_DIR / "random/val.json"
proofs_train = json.load(train_path.open())
for proof in proofs_train[::-1]:
    if proof["traced_tactics"] != []:
        if 'MiniF2F/Valid' in proof['file_path']:
            collect_minif2f_valid.append(proof)
        if 'MiniF2F/Test' in proof['file_path']:
            collect_minif2f_test.append(proof)
    if proof['full_name'] == 'mathd_algebra_478':
        print('!!!')
        print(proof)
print(len(collect_minif2f_valid))
print(len(collect_minif2f_test))

DST_DIR = Path("./minif2f_dataset")
train_path = DST_DIR / "random/test.json"
proofs_train = json.load(train_path.open())
for proof in proofs_train[::-1]:
    if proof["traced_tactics"] != []:
        if 'MiniF2F/Valid' in proof['file_path']:
            collect_minif2f_valid.append(proof)
        if 'MiniF2F/Test' in proof['file_path']:
            collect_minif2f_test.append(proof)
    if proof['full_name'] == 'mathd_algebra_478':
        print('!!!')
        print(proof)

print(len(collect_minif2f_valid))
print(len(collect_minif2f_test))

print(collect_minif2f_valid[0])


with open('minif2f_valid.json', 'w') as f:
    json.dump(collect_minif2f_valid, f)

f.close()

with open('minif2f_test.json', 'w') as f:
    json.dump(collect_minif2f_test, f)

f.close()