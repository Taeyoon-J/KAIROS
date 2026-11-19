"""Dataset-specific immutable input contracts and disjoint splits."""
import json
from pathlib import Path
import os
import numpy as np
from mod_benchmark import common as c
from readout_benchmark import runtime as r
from snn_kuramoto_bidirectional.sw240.data import train_ids


class Dataset:
    def __init__(self,dataset,root):
        self.dataset=dataset;self.root=Path(root);self.labels_opened=False
        if dataset=='clevr':
            source,sha=r.canonical_record();self.train=train_ids(35000)
            self.validation=np.arange(1000,1320,dtype=np.int64);self.test=np.arange(1320,1640,dtype=np.int64)
            self.binding=dict(dataset=dataset,canonical_source=source,canonical_manifest_sha256=sha,
                train_ids=self.train.tolist(),validation_ids=self.validation.tolist(),evaluation_ids=self.test.tolist(),
                scoring='native model grid lifted to native128 pixels; plus original fixed64 patch scores',
                split_name='fixed320 reserved evaluation, not an independent original-paper TEST split')
        elif dataset in ('multi_dsprites','tetrominoes','objects_room','clevrtex_original'):
            self.folder=self.root/'inputs'/dataset
            manifest=json.loads((self.folder/'manifest.json').read_bytes())
            assert manifest['status']=='complete_verified_TRAIN35000'
            with np.load(self.folder/'ids.npz',allow_pickle=False) as ids:
                self.train=ids['train'].copy();self.validation=ids['validation'].copy();self.test=ids['test'].copy()
            assert c.sha(self.folder/'ids.npz')==manifest['ids_sha256']
            self.binding=dict(dataset=dataset,manifest_sha256=c.sha(self.folder/'manifest.json'),source=manifest['source'],
                train_ids=self.train.tolist(),validation_ids=self.validation.tolist(),evaluation_ids=self.test.tolist(),
                scoring='original pixels; Tet model64 modal32 then protected lift to35; native32 predictions use protected lift')
            if dataset in ('objects_room','clevrtex_original'):
                with np.load(self.folder/'ids.npz',allow_pickle=False) as ids:self.source_ids=ids['source_ids'].copy()
                self.binding.update(preprocessing=manifest['preprocessing'],evaluation_source_ids=self.source_ids[self.test].tolist(),
                    scoring='standard official cropped/resized128 pixels' if dataset=='clevrtex_original' else 'original64 pixels on explicitly held-out in-distribution scenes',
                    evaluation_contract='fixed1000 official TEST subset' if dataset=='clevrtex_original' else manifest['primary_evaluation'])
        else:raise ValueError('dataset contract not yet implemented; do not silently substitute another dataset')
        assert len(self.train)==len(np.unique(self.train))==35000
        parent=self.train.copy()
        import os as _os
        preset=_os.environ.get('SW_TRAIN_PRESET','')
        first=np.sort(np.random.RandomState(2553500).choice(parent,10000,replace=False))
        unseen=np.sort(np.setdiff1d(parent,first,assume_unique=False))
        second=np.sort(np.random.RandomState(2551500).choice(unseen,15000,replace=False))
        if preset=='cont1000':
            # The scaling curve adds DATA, so each stage takes a fresh 1,000 from `second` --
            # the 15,000 the default draw excluded -- never from `first`, which the base core
            # already saw ten times. Stage index comes from SW_CONT1000_STAGE because the split
            # is built before the checkpoint is read.
            stage=int(os.environ.get('SW_CONT1000_STAGE','1'))
            assert 1<=stage<=15,('SW_CONT1000_STAGE must be 1..15',stage)
            self.train=second[(stage-1)*1000:stage*1000]
            self.binding.update(parent_TRAIN_count=len(parent),subset_seed=2551500,
                                cont1000_stage=stage,
                                warm_start_TRAIN_ids=first.tolist(),
                                train_ids=self.train.tolist(),unique_TRAIN=1000,
                                composition=f'images {(stage-1)*1000}:{stage*1000} of the '
                                            '15,000 RandomState(2551500) draw, which excludes '
                                            'the 10,000 the base core trained on; cumulative '
                                            f'unique images {10000+stage*1000}')
            assert len(self.train)==len(np.unique(self.train))==1000
            assert not np.intersect1d(self.train,first).size
        elif preset=='cont5000':
            # the first 5,000 of the same RandomState(2553500) draw the default preset uses, so
            # a cont5000 run sees a strict subset of the 10,000 and the ids stay reproducible
            self.train=first[:5000]
            self.binding.update(parent_TRAIN_count=len(parent),subset_seed=2553500,
                                train_ids=self.train.tolist(),unique_TRAIN=5000,
                                composition='first 5,000 of the default 10,000 subset; '
                                            'warm start from a 10,000x10 core, possibly on '
                                            'another dataset')
            assert len(self.train)==len(np.unique(self.train))==5000
        elif preset=='cont15000':
            self.train=second
            self.binding.update(parent_TRAIN_count=len(parent),subset_seed=2551500,
                                warm_start_TRAIN_ids=first.tolist(),
                                train_ids=self.train.tolist(),unique_TRAIN=15000)
            assert len(self.train)==len(np.unique(self.train))==15000
            assert not np.intersect1d(self.train,first).size
        elif preset=='union25000':
            self.train=np.sort(np.union1d(first,second))
            self.binding.update(parent_TRAIN_count=len(parent),subset_seed=[2553500,2551500],
                                train_ids=self.train.tolist(),unique_TRAIN=25000,
                                composition='union of the 10,000 and the disjoint 15,000; one shuffled run, not two phases')
            assert len(self.train)==len(np.unique(self.train))==25000
        elif preset:
            raise ValueError('unknown SW_TRAIN_PRESET: '+preset)
        else:
            self.train=first
            self.binding.update(parent_TRAIN_count=len(parent),subset_seed=2553500,
                                train_ids=self.train.tolist(),unique_TRAIN=10000)
            assert len(self.train)==len(np.unique(self.train))==10000
        assert not np.intersect1d(self.train,np.r_[self.validation,self.test]).size
        assert not np.intersect1d(self.validation,self.test).size

    def rgb(self,ids,split='train'):
        ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='clevr':return r.read_rgb(self.root,self.dataset,split,ids)
        return np.asarray(np.load(self.folder/'rgb128.npy',mmap_mode='r')[ids]).copy()

    def labels(self,ids,split='train'):
        self.labels_opened=True;ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='clevr':return r.truth(self.dataset,split,ids)
        return np.asarray(np.load(self.folder/'labels64.npy',mmap_mode='r')[ids],dtype=np.int64).copy()

    def pixels(self,ids):
        self.labels_opened=True;ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='clevr':
            import h5py
            source,_=r.canonical_record()
            with h5py.File(source['path'],'r') as stream:values=np.asarray(stream['mask'][ids,...,0])
            return values.astype(np.int64)
        return np.asarray(np.load(self.folder/'pixels.npy',mmap_mode='r')[ids],dtype=np.int64).copy()
