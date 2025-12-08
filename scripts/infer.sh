# clean logs and cache (optional)
rm -rf ./logs/* DM_P450_model/data/cache/* 
# setting up environment
export PYTHONPATH=$PWD:$PYTHONPATH
# running inference(choose one of the three models you want to use)
python scripts/infer.py -model DM-P450  -inputFA test.fasta -substrate  AGI.sdf | tee logs/infer.log
python scripts/infer.py -model Seq-Only  -inputFA test.fasta -substrate  AGI.sdf | tee logs/infer.log
python scripts/infer.py -model Pocket-Only  -inputFA test.fasta -substrate  AGI.sdf | tee logs/infer.log