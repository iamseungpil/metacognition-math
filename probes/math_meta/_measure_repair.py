import random, re, sys
sys.path.insert(0,'/home/v-seungplee/metacognition-math')
sys.path.insert(0,'/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl')
import countdown as OLD
from src.training import countdown_task as NEW

rng = random.Random(20260818)
insts=[]
while len(insts)<500:
    i = NEW.gen_instance(5, rng)
    if i: insts.append(i)

# --- decoy: old vs new
old_made=old_illegal=old_wrongmultiset=0
new_made=0
r1=random.Random(1); r2=random.Random(1)
for i in insts:
    d_old = OLD.swap_op_decoy(i["witness"], i["nums"], i["target"], r1)
    d_new = NEW.swap_op_decoy(i["witness"], i["nums"], i["target"], r2)
    if d_old is not None:
        old_made+=1
        if NEW.eval_countdown(d_old) is None: old_illegal+=1
        if NEW.expr_numbers(d_old)!=sorted(i["nums"]): old_wrongmultiset+=1
    if d_new is not None: new_made+=1
print(f"decoy  OLD made {old_made}/500  of which Countdown-ILLEGAL {old_illegal} ({100*old_illegal/max(old_made,1):.1f}%)  wrong-multiset {old_wrongmultiset}")
print(f"decoy  NEW made {new_made}/500  illegal 0 by construction")

# --- OF: old vs new
OF = re.compile(r"is (\d+) ([+\-*/]) (\d+) = (-?\d+);")
def check(s, nums):
    g=OF.search(s); 
    if not g: return "unparsed"
    a,op,b,v=int(g.group(1)),g.group(2),int(g.group(3)),int(g.group(4))
    true = {"+":a+b,"-":a-b,"*":a*b,"/":(a/b if b else None)}[op]
    if true is None or true!=v: return "false-equation"
    if NEW.eval_countdown(f"({a}{op}{b})") is None: return "illegal"
    pool=list(nums)
    if a not in pool: return "not-in-nums"
    pool.remove(a)
    if b not in pool: return "not-in-nums"
    return "ok"
from collections import Counter
co,cn=Counter(),Counter()
r1=random.Random(2); r2=random.Random(2)
for i in insts:
    mo=OLD.oracle_metas(i["witness"], i["nums"], r1)
    mn=NEW.oracle_metas(i["witness"], i["nums"], r2, fmt="old")
    co[check(mo["OF"], i["nums"]) if mo["OF"] else "none"]+=1
    cn[check(mn["OF"], i["nums"]) if mn["OF"] else "none"]+=1
print("OF OLD:", dict(co))
print("OF NEW:", dict(cn))

# --- parse_ok difference
junk=["(3+*)","(2+3","3 4","((2+3)","+","1++2"]
print("parse_ok OLD:", [OLD.parse_ok("\\boxed{%s}"%j) for j in junk])
print("parse_ok NEW:", [NEW.parse_ok("\\boxed{%s}"%j) for j in junk])
print("parse_ok NEW on valid:", NEW.parse_ok("\\boxed{(3+7)*8-25}"))

# --- grade float-error case
print("grade OLD on (1/3)*3*7*2 :", OLD.grade("\\boxed{((1/3)*3)*7*2}", [1,3,3,7,2], 14))
print("grade NEW on (1/3)*3*7*2 :", NEW.grade("\\boxed{((1/3)*3)*7*2}", [1,3,3,7,2], 14))
