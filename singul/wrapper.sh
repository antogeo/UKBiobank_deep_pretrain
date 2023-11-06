#!/bin/bash

cp brainage4ad/UKBiobank_deep_pretrain/singul/sub_agepred.py /tmp

mkdir /tmp/subj
cp /data/project/BrainAge4AD/data/ADNI_3T_T1w_BIDS/derivatives/fsl_ukb/sub-ADNI002S1280/ses-M120/sub-ADNI002S1280_ses-M120_T1w_linear_brain.nii.gz /tmp/subj/

singularity shell --fakeroot --nv --bind /tmp brainage4ad/UKBiobank_deep_pretrain/singul/deepl_pred2.simg python3 /tmp/sub_agepred.py --s_path $subj