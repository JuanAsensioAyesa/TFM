
prepath="./grids/new/"
for iFile in   400 
do
    echo $iFile
    ./build/bin/CONVERTER $prepath\TummorCells$iFile\.vdb $prepath\Oxygen$iFile\.vdb $iFile
done
