wget -O GaitDatasetB-silh.zip http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip
unzip GaitDatasetB-silh.zip
cd GaitDatasetB-silh
ls | xargs -n 1 tar -xvf
rm *.tar.gz
cd ..
echo "Done"
