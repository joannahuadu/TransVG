export CUDA_VISIBLE_DEVICES=7

# DIOR_RSVG
# python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --batch_size 16 --data_root /mnt/data1/workspace/wmq/YOLO-World/data/refGeo --split_root metainfo --dataset dior_rsvg --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /mnt/data1/workspace/wmq/TransVG/checkpoints/checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --max_query_len 20 --output_dir outputs/dior_rsvg --epochs 90 --lr_drop 60

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --batch_size 16 --data_root /mnt/data1/workspace/wmq/YOLO-World/data/refGeo --split_root metainfo --dataset dior_rsvg --lr_bert 0.0001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /mnt/data1/workspace/wmq/TransVG/checkpoints/checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --max_query_len 20 --output_dir outputs_lr1e-4/dior_rsvg --epochs 90 --lr_drop 60
