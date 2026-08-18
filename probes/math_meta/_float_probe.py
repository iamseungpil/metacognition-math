import sys, itertools
sys.path.insert(0,'/home/v-seungplee/metacognition-math')
sys.path.insert(0,'/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl')
import countdown as OLD
from src.training import countdown_task as NEW
from fractions import Fraction
hits=[]
for a in range(1,26):
    for b in range(1,26):
        for c in range(1,26):
            for d in range(1,26):
                # ((a/b)*c)*d  form: exact rational may be integer while float is not
                exact = Fraction(a,b)*c*d
                if exact.denominator!=1: continue
                t=int(exact)
                if t<20 or t>10000: continue
                e=f"(((({a}/{b})*{c})*{d}))"
                if OLD.grade("\\boxed{%s}"%e,[a,b,c,d],t)!=NEW.grade("\\boxed{%s}"%e,[a,b,c,d],t):
                    hits.append((e,t,OLD.grade("\\boxed{%s}"%e,[a,b,c,d],t),NEW.grade("\\boxed{%s}"%e,[a,b,c,d],t)))
                if len(hits)>5: break
            if len(hits)>5: break
        if len(hits)>5: break
    if len(hits)>5: break
print(len(hits), hits[:6])
