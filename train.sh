objects=('bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')
textures=('carpet' 'grid' 'leather' 'tile' 'wood')


python train.py -lp log/mvtec_train --sub-categories "${textures[@]}" \
-fd ./log/foreground_mvtec_DenseNet_features.denseblock1_320/ \
--steps 500 \
-tps 50 \
--data-dir log/synthetic_mvtec_640_12000_True_jpg

python train.py -lp log/mvtec_train --sub-categories "${objects[@]}" \
-fd ./log/foreground_mvtec_DenseNet_features.denseblock1_320/ \
--steps 40000 \
-tps 2000 \
--data-dir log/synthetic_mvtec_640_12000_True_jpg

