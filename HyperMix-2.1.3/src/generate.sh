#VD
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/VD_IT.d" -MT"src/VD_IT.d" -o "src/VD_IT.o" "VD_IT.cpp"
g++  -o "../bin/1_VD"  ./src/VD_IT.o   -llapack -lblas
#VD CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/VD.o" "../src/VD.cu"
nvcc --cudart static -link -o  "../bin/1_VD_CUDA"  ./src/VD.o   -llapack -lblas -lcublas
#HYSIME
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/HYSIME_IT.d" -MT"src/HYSIME_IT.d" -o "src/HYSIME_IT.o" "HYSIME_IT.cpp"
g++  -o "../bin/1_HYSIME"  ./src/HYSIME_IT.o   -llapack -lblas
#PCA
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/PCA_IT.d" -MT"src/PCA_IT.d" -o "src/PCA_IT.o" "PCA_IT.cpp"
g++  -o "../bin/2_PCA"  ./src/PCA_IT.o   -llapack -lblas
#PCA CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/PCA.o" "../src/PCA.cu"
nvcc --cudart static -link -o  "../bin/2_PCA_CUDA"  ./src/PCA.o   -llapack -lblas -lcublas
#SPP
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/SPP_IT.d" -MT"src/SPP_IT.d" -o "src/SPP_IT.o" "SPP_IT.cpp"
g++  -o "../bin/2_SPP"  ./src/SPP_IT.o
#SPP_GLOBAL CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/SPP_GLOBAL.o" "../src/SPP_GLOBAL.cu"
nvcc --cudart static -link -o  "../bin/2_SPP_GLOBAL_CUDA"  ./src/SPP_GLOBAL.o   -llapack -lblas -lcublas
#SPP_PIXBLOCK CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/SPP_PIXBLOCK.o" "../src/SPP_PIXBLOCK.cu"
nvcc --cudart static -link -o  "../bin/2_SPP_PIXBLOCK_CUDA"  ./src/SPP_PIXBLOCK.o   -llapack -lblas -lcublas
#OSP
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/OSP_IT.d" -MT"src/OSP_IT.d" -o "src/OSP_IT.o" "OSP_IT.cpp"
g++  -o "../bin/3_OSP"  ./src/OSP_IT.o   -llapack -lblas
#VCA
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/VCA_IT.d" -MT"src/VCA_IT.d" -o "src/VCA_IT.o" "VCA_IT.cpp"
g++  -o "../bin/3_VCA"  ./src/VCA_IT.o   -llapack -lblas
#VCA CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/VCA.o" "../src/VCA.cu"
nvcc --cudart static -link -o  "../bin/3_VCA_CUDA"  ./src/VCA.o   -llapack -lblas -lcublas
#IEA
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/IEA_IT.d" -MT"src/IEA_IT.d" -o "src/IEA_IT.o" "IEA_IT.cpp"
g++  -o "../bin/3_IEA"  ./src/IEA_IT.o   -llapack -lblas
#IEA CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/IEA.o" "../src/IEA.cu"
nvcc --cudart static -link -o  "../bin/3_IEA_CUDA"  ./src/IEA.o   -llapack -lblas -lcublas
#NFINDR
g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/NFINDR_IT.d" -MT"src/NFINDR_IT.d" -o "src/NFINDR_IT.o" "NFINDR_IT.cpp"
g++  -o "../bin/3_NFINDR"  ./src/NFINDR_IT.o -llapack -lblas
#NFINDR CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/NFINDR.o" "../src/NFINDR.cu"
nvcc --cudart static -link -o  "../bin/3_NFINDR_CUDA"  ./src/NFINDR.o   -llapack -lblas -lcublas
#LSU
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/LSU_IT.d" -MT"src/LSU_IT.d" -o "src/LSU_IT.o" "LSU_IT.cpp"
g++  -o "../bin/4_LSU"  ./src/LSU_IT.o   -llapack -lblas
#LSU CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/LSU.o" "../src/LSU.cu"
nvcc --cudart static -link -o  "../bin/4_LSU_CUDA"  ./src/LSU.o   -llapack -lblas -lcublas
#ISRA
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/ISRA_IT.d" -MT"src/ISRA_IT.d" -o "src/ISRA_IT.o" "ISRA_IT.cpp"
g++  -o "../bin/4_ISRA"  ./src/ISRA_IT.o   -llapack -lblas
#ISRA CUDA
nvcc --compile -G -O3 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/ISRA.o" "../src/ISRA.cu"
nvcc --cudart static -link -o  "../bin/4_ISRA_CUDA"  ./src/ISRA.o   -llapack -lblas -lcublas
#FCLSU
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/FCLSU_IT.d" -MT"src/FCLSU_IT.d" -o "src/FCLSU_IT.o" "FCLSU_IT.cpp"
g++  -o "../bin/4_FCLSU"  ./src/FCLSU_IT.o   -llapack -lblas
#NCLSU
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/NCLSU_IT.d" -MT"src/NCLSU_IT.d" -o "src/NCLSU_IT.o" "NCLSU_IT.cpp"
g++  -o "../bin/4_NCLSU"  ./src/NCLSU_IT.o   -llapack -lblas
#SCLSU
g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/SCLSU_IT.d" -MT"src/SCLSU_IT.d" -o "src/SCLSU_IT.o" "SCLSU_IT.cpp"
g++  -o "../bin/4_SCLSU"  ./src/SCLSU_IT.o   -llapack -lblas
