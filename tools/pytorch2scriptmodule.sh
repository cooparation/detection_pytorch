python ./tools/pytorch2scriptmodule.py \
    --config-file tmp/ssd300_person.yaml \
    --model_path ./output/person/ssd300_vgg_final.pth \
    --model_out scriptmodel_tmp.pt
    #--config-file configs/ssd300_voc0712.yaml \
