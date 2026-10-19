"""Dependency-free continual-learning experiment on the explicitly synthetic stream.

This is a transparent algorithmic proof-of-concept, not evidence about real learners
or Nigerian-language comprehension. Run: python -m src.experiment --out experiments/results
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
from statistics import mean, stdev
from src.simulate import simulate, TOPICS
from src.metrics import backward_transfer, forward_transfer

D = 9  # bias, six topic indicators, difficulty, prior within-topic accuracy

def features(row, hist):
    x = [1.0] + [float(row['topic'] == t) for t in TOPICS]
    x += [{"easy": 0.0, "medium": 0.5, "hard": 1.0}[row["difficulty"]],
          (hist[0] + 1.0) / (hist[1] + 2.0)]
    return x

def sigmoid(z):
    if z >= 0: return 1.0 / (1.0 + math.exp(-min(z, 35)))
    ez = math.exp(max(z, -35)); return ez / (1 + ez)

def predict(w, x): return sigmoid(sum(a*b for a,b in zip(w,x)))

def train_step(w, x, y, lr, penalty=None, anchor=None, fisher=None, lam=0.0):
    p = predict(w,x)
    for j in range(D):
        g = (p-y)*x[j]
        if penalty is not None and anchor is not None:
            g += lam * penalty[j] * (w[j]-anchor[j])
        w[j] -= lr*g
    return p

def clone(w): return list(w)

def histories(rows):
    counts={}; out=[]
    for r in rows:
        k=(r['student_id'],r['topic']); c=counts.get(k,[0,0])
        out.append((r,features(r,c)))
        c[0]+=int(r['correct']); c[1]+=1; counts[k]=c
    return out

def acc(w, data):
    return sum((predict(w,x)>=.5)==bool(int(r['correct'])) for r,x in data)/max(1,len(data))

def fisher_diag(w, data):
    f=[1e-3]*D
    for r,x in data:
        p=predict(w,x)
        for j in range(D): f[j] += (p*(1-p)*x[j])**2
    scale=max(1,len(data))
    return [v/scale for v in f]

def run(seed, n_students=500, qpt=15, epochs=2, lr=.08, buffer_size=300):
    rows=simulate(n_students,qpt,seed)
    # Split by student, never by interaction. Holdout IDs are never optimized on.
    train_ids={f'S{i:04d}' for i in range(n_students) if i%5 != 0}
    train_rows=[r for r in rows if r['student_id'] in train_ids]
    test_rows=[r for r in rows if r['student_id'] not in train_ids]
    train_data=histories(train_rows); test_data=histories(test_rows)
    train_by={t:[p for p in train_data if p[0]['topic']==t] for t in TOPICS}
    test_by={t:[p for p in test_data if p[0]['topic']==t] for t in TOPICS}
    methods=['Fine-tuning','EWC','Replay','Joint training','Task-isolated models']
    matrices={m:[[None]*len(TOPICS) for _ in TOPICS] for m in methods}
    # Zero-init baseline for FWT, measured before any task updates.
    zeros=[0.0]*D
    baselines=[acc(zeros,test_by[t]) for t in TOPICS]
    rng=random.Random(seed+77)
    for method in methods:
        w=clone(zeros); anchor=clone(w); fish=[0.0]*D; buffer=[]; task_models=[]
        for ti,t in enumerate(TOPICS):
            # Save a genuine pre-training evaluation on the next task for FWT.
            if ti > 0:
                if method == 'Task-isolated models':
                    pre_score = acc(zeros, test_by[t])
                else:
                    pre_score = acc(w, test_by[t])
                matrices[method][ti-1][ti] = pre_score
            batch=train_by[t]
            if method=='Joint training': batch=[p for prev in TOPICS[:ti+1] for p in train_by[prev]]
            if method=='Task-isolated models':
                wt=clone(zeros)
                for _ in range(epochs):
                    for r,x in batch: train_step(wt,x,int(r['correct']),lr)
                task_models.append(wt)
            else:
                for _ in range(epochs):
                    seq=list(batch)
                    if method=='Replay' and buffer:
                        seq += [rng.choice(buffer) for _ in range(min(len(buffer),len(batch)))]
                    rng.shuffle(seq)
                    for r,x in seq:
                        penalty=fish if method=='EWC' else None
                        train_step(w,x,int(r['correct']),lr,penalty,anchor,fish,lam=8.0)
                if method=='EWC':
                    newf=fisher_diag(w,batch)
                    fish=[a+b for a,b in zip(fish,newf)]; anchor=clone(w)
                if method=='Replay':
                    buffer.extend(batch)
                    if len(buffer)>buffer_size: buffer=rng.sample(buffer,buffer_size)
            for j in range(ti+1):
                if method=='Task-isolated models': score=acc(task_models[j],test_by[TOPICS[j]])
                elif method=='Joint training':
                    # Single jointly updated model evaluated across observed tasks.
                    score=acc(w,test_by[TOPICS[j]])
                else: score=acc(w,test_by[TOPICS[j]])
                matrices[method][ti][j]=score
    metrics={}
    for m in methods:
        mat=matrices[m]
        bwt=backward_transfer(mat)
        # FWT from the model's initial, pre-task scores (shared initialization).
        fwt=forward_transfer(mat,baselines)
        final=mat[-1]
        metrics[m]={"accuracy":mean(v for v in final if v is not None),"bwt":bwt,"fwt":fwt,
                    "task_accuracy":final,"matrix":mat}
    # Predictive adaptation profile using each student's preceding same-topic history.
    # This is a descriptive pooled curve, not causal evidence or held-out learning gain.
    baseline_correct=[]; by_n={1:[],3:[],5:[],10:[]}
    seen={}
    # Use final EWC learner as shared predictor; evaluate sequentially by test student.
    # Recover model weights by replaying the EWC training deterministically.
    w=clone(zeros); anchor=clone(w); fish=[0.0]*D; buffer=[]; rr=random.Random(seed+77)
    for ti,t in enumerate(TOPICS):
        batch=train_by[t]
        for _ in range(epochs):
            seq=list(batch)
            rr.shuffle(seq)
            for r,x in seq: train_step(w,x,int(r['correct']),lr,fish,anchor,fish,lam=8.0)
        nf=fisher_diag(w,batch); fish=[a+b for a,b in zip(fish,nf)]; anchor=clone(w)
    for r,x in test_data:
        k=(r['student_id'],r['topic']); n=seen.get(k,0)+1; seen[k]=n
        hit=int((predict(w,x)>=.5)==bool(int(r['correct'])))
        if n==1: baseline_correct.append(hit)
        if n in by_n: by_n[n].append(hit)
    adaptation={str(k):mean(v) if v else None for k,v in by_n.items()}
    return {"seed":seed,"students":n_students,"train_students":len(train_ids),"test_students":n_students-len(train_ids),
            "rows":len(rows),"topics":TOPICS,"metrics":metrics,"adaptation_accuracy_by_history_length":adaptation,
            "zero_model_accuracy":mean(baseline_correct)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path('experiments/results'))
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]); p.add_argument('--students',type=int,default=500)
    p.add_argument('--questions-per-topic',type=int,default=15); p.add_argument('--epochs',type=int,default=2)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    runs=[run(s,a.students,a.questions_per_topic,a.epochs) for s in a.seeds]
    raw=a.out/'seed_results.json'; raw.write_text(json.dumps(runs,indent=2)+'\n',encoding='utf-8')
    methods=list(runs[0]['metrics'])
    with (a.out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['method','accuracy_mean','accuracy_sd','bwt_mean','bwt_sd','fwt_mean','fwt_sd'])
        for m in methods:
            vals=[[r['metrics'][m][k] for r in runs] for k in ('accuracy','bwt','fwt')]
            w.writerow([m, round(mean(vals[0]),6), round(stdev(vals[0]),6) if len(vals[0])>1 else 0, round(mean(vals[1]),6), round(stdev(vals[1]),6) if len(vals[1])>1 else 0, round(mean(vals[2]),6), round(stdev(vals[2]),6) if len(vals[2])>1 else 0])
    print('Wrote',raw,'and',a.out/'summary.csv')
    for row in csv.DictReader((a.out/'summary.csv').open()): print(row)
if __name__=='__main__': main()
