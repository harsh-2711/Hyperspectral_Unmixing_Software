sudo apt-get install g++
sudo apt-get install libblas3 libblas-dev liblapack3 liblapack-dev
sudo apt-get install qt4-* qtcreator
qmake XHyperMix.pro -r spec linux-g++-64 CONFIG+=debug CONFIG+=declarative_debug
make
cd src
sh generate.sh
