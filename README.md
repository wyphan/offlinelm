### Setup

1. Create `conda` environment
```
conda create -n agent python=3.14
conda activate agent

conda install pip
python3 -m pip install google-antigravity
python3 -m pip install litert-lm
conda install openai
```

2. Download `llama.cpp` release 1.0.0 from https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0

3a. Build llama-cli and llama-server with Vulkan support
```
sudo apt update && sudo apt install -y build-essential cmake libvulkan-dev glslc spirv-headers libssl-dev
tar xf llama.cpp-1.0.0.tar.gz
cmake -B build -DGGML_VULKAN=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
cmake --build build --config Release --target llama-cli llama-server -j
```

3b. Build llama-cli and llama-server with CUDA support
```
sudo apt update && sudo apt install -y build-essential cmake libvulkan-dev glslc spirv-headers libssl-dev
tar xf llama.cpp-1.0.0.tar.gz
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
cmake --build build --config Release --target llama-cli llama-server -j
```

3. Install `llama.cpp` locally in the home directory
```
mkdir -p ~/sw/llama-cpp/bin ~/sw/llama-cpp/lib
cp -P build/bin/llama-cli build/bin/llama-server ~/sw/llama-cpp/bin/
cp -P build/bin/*.so* ~/sw/llama-cpp/lib/ 2>/dev/null || true
find build -name "*.so*" -exec cp -P {} ~/sw/llama-cpp/lib/ \;
cd ~/bin
ln -s ../sw/llama-cpp/bin/llama-cli llama-cli
ln -s ../sw/llama-cpp/bin/llama-server llama-server
```
