#include <torch/script.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./hw2-4-1 <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    int total_model_weight = 0;
    for (const auto& param : module.parameters()) {
        // 同 pytorch 用法
        total_model_weight += param.numel() * param.element_size();
    }
    std::cout << "Total model weights: " << total_model_weight << std::endl;
    
    return 0;
}