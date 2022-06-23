mkdir -p data
cd data
mkdir -p images
mkdir -p types
mkdir -p masks
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip
unzip fold_1.zip
mv "Fold 1/images/fold1/images.npy" images/fold_1.npy
mv "Fold 1/images/fold1/types.npy" types/fold_1.npy
mv "Fold 1/masks/fold1/masks.npy" masks/fold_1.npy
rm -f fold_1.zip
rm -rf "Fold 1"
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip
unzip fold_2.zip
mv "Fold 2/images/fold2/images.npy" images/fold_2.npy
mv "Fold 2/images/fold2/types.npy" types/fold_2.npy
mv "Fold 2/masks/fold2/masks.npy" masks/fold_2.npy
rm -f fold_2.zip
rm -rf "Fold 2"
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip
unzip fold_3.zip
mv "Fold 3/images/fold3/images.npy" images/fold_3.npy
mv "Fold 3/images/fold3/types.npy" types/fold_3.npy
mv "Fold 3/masks/fold3/masks.npy" masks/fold_3.npy
rm -f fold_3.zip
rm -rf "Fold 3"
