### Setup

```
# Create conda environment
conda create -n agent python=3.14
conda activate agent

conda install pip
python3 -m pip install google-antigravity
python3 -m pip install litert-lm
conda install openai

# Build llama-cpp-python, llama-cli, and llama-server with Vulkan support
sudo apt update && sudo apt install -y build-essential cmake libvulkan-dev glslc spirv-headers libssl-dev
CMAKE_ARGS="-DGGML_VULKAN=ON" python3 -m pip install llama-cpp-python --no-binary=llama-cpp-python --force-reinstall

tar xf llama.cpp-1.0.0.tar.gz
cmake -B build -DGGML_VULKAN=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
cmake --build build --config Release --target llama-cli llama-server -j

mkdir -p ~/sw/llama-cpp/bin ~/sw/llama-cpp/lib
cp -P build/bin/llama-cli build/bin/llama-server ~/sw/llama-cpp/bin/
cp -P build/bin/*.so* ~/sw/llama-cpp/lib/ 2>/dev/null || true
find build -name "*.so*" -exec cp -P {} ~/sw/llama-cpp/lib/ \;
cd ~/bin
ln -s ../sw/llama-cpp/bin/llama-cli llama-cli
ln -s ../sw/llama-cpp/bin/llama-server llama-server
```
