"""Resumable scratch training: exact3500 x10, effective batch16, actual native rollout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from mod_benchmark import common as c
from readout_benchmark import runtime as r
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
from full_benchmark.statistics import fit_dataset
from full_benchmark.loss import InitialTeacher,objective
from full_benchmark.evaluation import evaluate

PRESET=os.environ.get('SW_TRAIN_PRESET','')
# cont5000 continues a finished 10,000x10 core for a further 5,000 unique x 10 epochs, and
# unlike cont15000 it is meant to be pointed at a DIFFERENT --dataset: WARM already relaxes the
# binding assertions and resets the epoch schedule, so the weights carry over while the data
# changes. 5,000/16 = 312.5 -> 313 updates per epoch, matching how cont15000 rounds 937.5 to 938.
WARM=PRESET in ('cont15000','cont5000','cont1000')   # resumes a 10,000x10 checkpoint
SCRATCH25=PRESET=='union25000'        # one shuffled run over 25,000, no warm start
if PRESET=='cont1000':
    # The scaling curve. One stage adds 1,000 UNSEEN images x 10 epochs; stages chain, so the
    # absolute total depends on what was resumed and is computed below, as for cont5000.
    PRESET_UNIQUE,PRESET_EPOCHS=1000,10
    PRESET_ADD_UPDATES,PRESET_ADD_SEEN=63*10,10000         # 630 / 10,000 per stage
    PRESET_UPDATES,PRESET_SEEN=None,None
elif PRESET=='cont5000':
    PRESET_UNIQUE,PRESET_EPOCHS=5000,10
    # One cont5000 stage ADDS this much. The absolute total depends on what the resumed
    # checkpoint already carried, so it is computed below rather than fixed here -- otherwise
    # the first continuation passes and every chained one aborts on the final assertion.
    PRESET_ADD_UPDATES,PRESET_ADD_SEEN=313*10,50000        # 3,130 / 50,000 per stage
    PRESET_UPDATES,PRESET_SEEN=None,None
elif WARM:
    PRESET_UNIQUE,PRESET_EPOCHS=15000,10
    PRESET_UPDATES,PRESET_SEEN=6250+938*10,100000+150000    # 15,630 / 250,000 cumulative
elif SCRATCH25:
    PRESET_UNIQUE,PRESET_EPOCHS=25000,10
    PRESET_UPDATES,PRESET_SEEN=1563*10,250000               # 15,630 / 250,000
else:
    PRESET_UNIQUE,PRESET_EPOCHS=10000,10
    PRESET_UPDATES,PRESET_SEEN=6250,100000


def scientific_sources():
    paths=list(Path(__file__).parent.glob('*.py'))
    for directory in ('snn_kuramoto_bidirectional','readout_benchmark'):
        paths+=list((c.CODE/directory).rglob('*.py'))
    import neutral_readout.head
    paths.append(Path(neutral_readout.head.__file__))
    return {str(p):c.sha(p) for p in sorted(set(paths))}


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train(dataset_name,variant,seed,root,*,admission=False):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA required')
    c.immutable();root=Path(root);dataset=Dataset(dataset_name,root)
    namespace='engine_admission' if admission else 'jobs'
    out=root/namespace/f'model{variant}'/dataset_name/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    epochs=1 if admission else PRESET_EPOCHS;batch=2 if admission else 16
    binding=dict(experiment='SW0255',dataset=dataset.binding,variant=variant,seed=seed,
        initialization='random_from_scratch',unique_train=10000,epochs=epochs,effective_batch=batch,
        microbatch=1,loss_reduction='mean of per-image objectives; divide by actual last batch length',
        total_rollout_steps=1024,gradient_tail_steps=64,all_active_groups_trained=True,
        pretrained_checkpoint_used=False,mask_training=variant!=253,
        final_mask_to_body_gradient=variant==254,admission_only=admission,
        zero_bias_policy='shared dendritic bias zero, every seed and model',
        code=scientific_sources(),benchmark=c.immutable())
    # JSON normalization makes comparison stable across restarts.
    binding=json.loads(json.dumps(binding));binding_path=out/'binding.json'
    effective=dict(code=binding['code'],CUDA_algorithm_policy='unchanged; intrinsic original adaptive-pooling backward roundoff retained',
        CPU_partition_policy='exact original energy arithmetic/merge bytecode, blocked independent rows')
    if binding_path.exists():
        previous=json.loads(binding_path.read_bytes())
        if previous!=binding:
            migration=json.loads((out/'runtime_migration.json').read_bytes())
            contract_path=Path(migration['contract_path']);contract=json.loads(contract_path.read_bytes())
            assert c.sha(contract_path)==migration['contract_sha256']
            assert contract['status']=='verified_same_training_with_exact_CPU_partition_acceleration'
            assert {k:v for k,v in previous.items() if k!='code'}=={k:v for k,v in binding.items() if k!='code'}
            assert contract['from_code']==previous['code'] and contract['to_code']==binding['code']
            assert migration['original_binding_sha256']==c.sha(binding_path)
            for proof in contract['proofs']:
                assert c.sha(proof['path'])==proof['sha256']
            effective['migration_sha256']=c.sha(out/'runtime_migration.json')
            binding=previous  # Preserve original dataset/budget binding and checkpoint bytes.
    else:c.write(binding_path,binding)
    binding_sha=c.sha(binding_path)
    effective['original_binding_sha256']=binding_sha
    if (out/'effective_code.json').exists():assert json.loads((out/'effective_code.json').read_bytes())==effective
    else:c.write(out/'effective_code.json',effective)
    effective_sha=c.sha(out/'effective_code.json')
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_bytes());assert complete['binding_sha256']==binding_sha
        return complete
    model=FreshModel(variant,seed).to('cuda:0')
    initial_path=out/'initial.pt'
    if initial_path.exists():
        initial=torch.load(initial_path,map_location='cpu',weights_only=True)
        assert WARM or initial['binding_sha256']==binding_sha
        model.load_state_dict(initial['model']);normalization=initial['normalization']
    else:
        normalization=fit_dataset(model,dataset,progress=lambda n:c.write(out/'progress.json',dict(stage='TRAIN_only_initial_statistics',completed=n,total=10000)))
        initial=dict(model=cpu_state(model),binding_sha256=binding_sha,normalization=normalization)
        r.save(initial_path,initial)
    model.normalization_provenance=normalization['provenance']
    teacher=InitialTeacher(model) if variant!=253 else None
    initial_values={name:initial['model'][name] for name,p in model.named_parameters() if p.requires_grad}
    initial_sha=c.sha(initial_path)
    groups=model.parameter_groups();body=[g for g in groups if g['name']!='mask_head']
    optimizer=torch.optim.Adam(body)
    head_optimizer=torch.optim.Adam([g for g in groups if g['name']=='mask_head']) if model.head is not None else None
    coverage={name:0. for name,p in model.named_parameters() if p.requires_grad}
    updates=0;seen=0;next_epoch=0;next_offset=0
    expect_updates,expect_seen=PRESET_UPDATES,PRESET_SEEN
    resume_path=out/'resume.pt'
    if resume_path.exists():
        resume=torch.load(resume_path,map_location='cuda:0',weights_only=True)
        assert WARM or (resume['binding_sha256']==binding_sha and resume['initial_sha256']==initial_sha)
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        if head_optimizer:head_optimizer.load_state_dict(resume['head_optimizer'])
        coverage=resume['gradient_coverage'];updates=resume['updates'];seen=resume['images_seen']
        if PRESET in ('cont5000','cont1000'):
            # chainable: this stage must end one increment above whatever it resumed
            expect_updates,expect_seen=updates+PRESET_ADD_UPDATES,seen+PRESET_ADD_SEEN
            print(f'{PRESET}: resumed at {updates} updates / {seen} images; '
                  f'this stage targets {expect_updates} / {expect_seen}',flush=True)
        next_epoch=resume['next_epoch'];next_offset=resume['next_offset']
        if WARM and next_epoch*len(dataset.train)+next_offset!=seen:
            # Entering the warm start: the prior run's epoch counter indexes its own
            # 10,000-image schedule and cannot index this one. Restart epochs at 0 while
            # carrying updates and images_seen forward -- the optimizer step count must stay
            # continuous, because the restore assert at the end compares it against updates.
            if PRESET in ('cont5000','cont1000'):
                # the base core, or the base core plus a whole number of this preset's stages
                k_u,k_s=updates-6250,seen-100000
                assert k_u>=0 and k_s>=0 and k_u%PRESET_ADD_UPDATES==0 \
                       and k_s%PRESET_ADD_SEEN==0 and k_u//PRESET_ADD_UPDATES==k_s//PRESET_ADD_SEEN, \
                       (PRESET+' must resume the base 6250/100000 core or a whole number of '
                        'its own stages past it',updates,seen)
                print(f'{PRESET}: entering stage {k_u//PRESET_ADD_UPDATES+1}',flush=True)
            else:
                assert updates==6250 and seen==100000,(updates,seen)
            next_epoch=0;next_offset=0
        else:
            assert next_epoch*len(dataset.train)+next_offset==seen
    def snapshot(epoch,offset):
        r.save(resume_path,dict(model=cpu_state(model),optimizer=optimizer.state_dict(),
            head_optimizer=head_optimizer.state_dict() if head_optimizer else None,
            binding_sha256=binding_sha,initial_sha256=initial_sha,updates=updates,images_seen=seen,
            next_epoch=epoch,next_offset=offset,gradient_coverage=coverage,
            diagnostic_only=admission,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),effective_code_sha256=effective_sha))
        c.write(out/'committed.json',dict(updates=updates,images_seen=seen,next_epoch=epoch,next_offset=offset,
            checkpoint_size=resume_path.stat().st_size,checkpoint_sha256=c.sha(resume_path),binding_sha256=binding_sha))
    started=time.time()
    for epoch in range(next_epoch,epochs):
        order=dataset.train[np.random.RandomState(249000+seed+10000*epoch).permutation(len(dataset.train))]
        offset=next_offset if epoch==next_epoch else 0
        for start in range(offset,len(order),batch):
            if admission and updates==2:break
            selected=order[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            if head_optimizer:head_optimizer.zero_grad(set_to_none=True)
            images=dataset.rgb(selected)
            labels=dataset.labels(selected) if variant!=253 else None
            accumulated_loss=0.;parts_sum={}
            for index,image_id in enumerate(selected):
                c.write(out/'progress.json',dict(stage='training_actual_native_forward_backward',epoch=epoch+1,total_epochs=epochs,
                    optimizer_updates=updates,images_seen=seen,current_image=int(image_id),microbatch=index+1,effective_batch=len(selected)))
                image=torch.from_numpy(images[index:index+1]).to('cuda:0')
                output=model(image,inference_seed=255000+int(image_id)+100000*epoch,live_tail_steps=64)
                total,base,mask,parts=objective(model,output,rgb=image,labels=labels[index:index+1] if labels is not None else None,teacher=teacher)
                (total/len(selected)).backward()
                accumulated_loss+=float(total.detach())/len(selected)
                for key,value in parts.items():parts_sum[key]=parts_sum.get(key,0.)+float(value.detach())/len(selected)
                del total,base,mask,parts,output
            if variant==253:assert not dataset.labels_opened,'unsupervised training opened target masks'
            for name,p in model.named_parameters():
                if not p.requires_grad or p.grad is None:continue
                if not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('nonfinite gradient: '+name)
                coverage[name]+=float(p.grad.detach().abs().sum())
            norm=torch.nn.utils.clip_grad_norm_([p for g in body for p in g['params']],1.,error_if_nonfinite=True)
            if head_optimizer:torch.nn.utils.clip_grad_norm_(list(model.head.parameters()),1.,error_if_nonfinite=True)
            optimizer.step()
            if head_optimizer:head_optimizer.step()
            updates+=1;seen+=len(selected)
            end=start+len(selected);resume_epoch=epoch+int(end==len(order));resume_offset=0 if end==len(order) else end
            if updates==1 or updates%8==0 or end==len(order) or (admission and updates==2):snapshot(resume_epoch,resume_offset)
            with (out/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=updates,epoch=epoch+1,ids=selected.tolist(),loss=accumulated_loss,parts=parts_sum,
                    gradient_norm=float(norm),images_seen=seen,seconds=time.time()-started),allow_nan=False)+'\n')
        if admission:break
    assert seen==(4 if admission else expect_seen) and updates==(2 if admission else expect_updates),(seen,updates)
    inactive=[name for name,value in coverage.items() if value==0]
    assert not inactive,('inactive active parameter tensors',inactive)
    changes={name:float((p.detach().cpu()-initial_values[name]).abs().max()) for name,p in model.named_parameters() if p.requires_grad}
    for group in groups:
        identities={id(p) for p in group['params']}
        assert any(changes[name]>0 for name,p in model.named_parameters() if id(p) in identities),group['name']
    # Test actual checkpoint and optimizer restoration before evaluating.
    digest=state_digest(model);restored=torch.load(resume_path,map_location='cuda:0',weights_only=True)
    model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
    if head_optimizer:head_optimizer.load_state_dict(restored['head_optimizer'])
    assert state_digest(model)==digest and {int(s['step'].item()) for s in optimizer.state.values()}=={updates}
    if head_optimizer:assert {int(s['step'].item()) for s in head_optimizer.state.values()}=={updates}
    c.write(out/'training_report.json',dict(binding_sha256=binding_sha,initial_sha256=initial_sha,updates=updates,images_seen=seen,
        unique_TRAIN=PRESET_UNIQUE,epochs_completed=0 if admission else PRESET_EPOCHS,
        cont5000_stage_added=dict(updates=PRESET_ADD_UPDATES,images_seen=PRESET_ADD_SEEN) if PRESET in ('cont5000','cont1000') else None,
        train_preset=PRESET or 'first10000',
        warm_start='10,000x10 = 100,000 exposures, 6,250 updates' if WARM else None,
        total_exposures=seen if PRESET in ('cont5000','cont1000') else PRESET_SEEN,
        schedule_note='two consecutive halves, not one shuffled run' if WARM else None,
        diagnostic_only=admission,
        active_tensor_gradient_coverage=coverage,parameter_changes=changes,optimizer_resume_verified=True,
        model_sha256=digest,mask_opened_during_training=dataset.labels_opened,seconds=time.time()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),no_parameter_checkpoint_reuse=True,effective_code_sha256=effective_sha))
    # Diagnostic evaluation is ONE reserved image, reported explicitly as such.
    result=evaluate(model,dataset,out/'evaluation',count=1 if admission else None)
    complete=dict(status='engine_diagnostic_passed' if admission else 'full10000_10passes_and_evaluation_complete',
        binding_sha256=binding_sha,training_report_sha256=c.sha(out/'training_report.json'),
        evaluation_sha256=c.sha(out/'evaluation/result.json'),evaluation_count=result['count'],
        variant=variant,seed=seed,dataset=dataset_name,updates=updates,images_seen=seen,diagnostic_only=admission)
    c.write(out/'complete.json',complete);return complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--variant',type=int,choices=(252,253,254,240),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--root',required=True);p.add_argument('--admission',action='store_true')
    a=p.parse_args();print(json.dumps(train(a.dataset,a.variant,a.seed,a.root,admission=a.admission)))
