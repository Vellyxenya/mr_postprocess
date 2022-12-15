# Post-Processing Pipeline

## Requirements
* Open3D

## Run
```
mkdir build && cd build
cmake -DOpen3D_ROOT=path_to_open3d_install -DCMAKE_BUILD_TYPE=Release ..
(e.g.: cmake -DOpen3D_ROOT=${HOME}/open3d_install -DCMAKE_BUILD_TYPE=Release ..)
make
./postprocess ../finaldenoisedboxdb.xyz
```