#include "include/Samples_Gen.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

struct CLI {
    std::string Tc_seed_string    = "--Tc_seed";
    std::string Num_Samples_string = "--Num_Samples";
    std::string L_string          = "--L";
    std::string T_ranges_string   = "--T_ranges";

    double Tc_seed     = -1.0;
    int    Num_Samples = -1;
    int    L           = -1;
    double T_min       = -1.0;
    double T_max       = -1.0;
    int    T_n         = -1;
};

void PrintUsage();

int main(int argc, char* argv[]) {
    if (argc == 1) {
        std::cerr << "Error: no arguments provided. Run with --help for usage.\n";
        return 1;
    }
    if (std::string(argv[1]) == "--h" || std::string(argv[1]) == "--help") {
        PrintUsage();
        return 0;
    }

    CLI args;
    int i = 1;
    while (i < argc) {
        std::string flag = argv[i];
        if (flag == args.Tc_seed_string) {
            if (i + 1 >= argc) { std::cerr << "Error: --Tc_seed requires a value.\n"; return 1; }
            args.Tc_seed = std::stod(argv[++i]);
        } else if (flag == args.Num_Samples_string) {
            if (i + 1 >= argc) { std::cerr << "Error: --Num_Samples requires a value.\n"; return 1; }
            args.Num_Samples = std::stoi(argv[++i]);
        } else if (flag == args.L_string) {
            if (i + 1 >= argc) { std::cerr << "Error: --L requires a value.\n"; return 1; }
            args.L = std::stoi(argv[++i]);
        } else if (flag == args.T_ranges_string) {
            if (i + 3 >= argc) { std::cerr << "Error: --T_ranges requires three values: T_min T_max T_n.\n"; return 1; }
            args.T_min = std::stod(argv[++i]);
            args.T_max = std::stod(argv[++i]);
            args.T_n   = std::stoi(argv[++i]);
        } else {
            std::cerr << "Error: unknown argument '" << flag << "'. Run with --help for usage.\n";
            return 1;
        }
        ++i;
    }

    // Validate all required arguments were provided
    if (args.Tc_seed < 0)   { std::cerr << "Error: --Tc_seed is required.\n";    return 1; }
    if (args.Num_Samples < 0){ std::cerr << "Error: --Num_Samples is required.\n"; return 1; }
    if (args.L < 0)         { std::cerr << "Error: --L is required.\n";           return 1; }
    if (args.T_min < 0)     { std::cerr << "Error: --T_ranges is required.\n";    return 1; }
    if (args.T_min <= 0)    { std::cerr << "Error: T_min must be > 0 (temperature must be positive).\n"; return 1; }
    if (args.T_n < 2)       { std::cerr << "Error: T_n must be >= 2.\n";          return 1; }
    if (args.T_min >= args.T_max) { std::cerr << "Error: T_min must be less than T_max.\n"; return 1; }

    // Clean up output directories from any previous run
    namespace fs = std::filesystem;
    for (const auto& dir : {"data", "plots"}) {
        if (fs::exists(dir)) {
            std::uintmax_t removed = fs::remove_all(dir);
            std::cout << "Cleaned " << dir << "/ (" << removed << " files removed)\n";
        }
        fs::create_directory(dir);
    }
    std::cout << "\n";
    std::cout.flush();

    // Build temperature vector (equivalent to np.linspace(T_min, T_max, T_n))
    std::vector<double> temperatures(args.T_n);
    for (int k = 0; k < args.T_n; ++k) {
        temperatures[k] = args.T_min + k * (args.T_max - args.T_min) / (args.T_n - 1);
    }

    // Write run parameters so notebooks and the dashboard can read the exact values
    {
        std::ofstream params("data/run_params.json");
        params << std::fixed << std::setprecision(10);
        params << "{\n";
        params << "  \"Tc_seed\":     " << args.Tc_seed     << ",\n";
        params << "  \"Num_Samples\": " << args.Num_Samples  << ",\n";
        params << "  \"L\":           " << args.L            << ",\n";
        params << "  \"T_min\":       " << args.T_min        << ",\n";
        params << "  \"T_max\":       " << args.T_max        << ",\n";
        params << "  \"T_n\":         " << args.T_n          << ",\n";
        params << "  \"temperatures\": [";
        for (int k = 0; k < args.T_n; ++k) {
            params << temperatures[k];
            if (k < args.T_n - 1) params << ", ";
        }
        params << "]\n}\n";
    }

    std::cout << "=== 2D Ising Model — Monte Carlo Sampler ===\n";
    std::cout << "  Tc_seed     = " << args.Tc_seed    << "\n";
    std::cout << "  Num_Samples = " << args.Num_Samples << "\n";
    std::cout << "  L           = " << args.L           << "\n";
    std::cout << "  T_ranges    = linspace(" << args.T_min << ", " << args.T_max << ", " << args.T_n << ")\n";
    std::cout << "  Temperatures: [ ";
    for (double T : temperatures) std::cout << T << " ";
    std::cout << "]\n\n";

    Samples simulator(args.L, args.Num_Samples);

    for (double T : temperatures) {
        simulator.Generate(T, args.Tc_seed);
        std::cout << "Samples generated for T = " << T << "\n";
    }

    std::cout << "\nAll samples written to data/\n";
    std::cout << "\nRunning analysis pipeline (notebooks + plots)...\n";
    std::cout.flush();

    int ret = system("python3 python/run_pipeline.py");
    if (ret != 0) {
        std::cerr << "Warning: pipeline script exited with code " << ret << ".\n";
        std::cerr << "You can run it manually: python3 python/run_pipeline.py\n";
    }

    std::cout << "\nDone. To launch the dashboard:\n";
    std::cout << "  streamlit run streamlit_app.py\n\n";

    return 0;
}

void PrintUsage() {
    std::cout << "\nUsage: ./main --Tc_seed <val> --Num_Samples <val> --L <val> --T_ranges <T_min> <T_max> <T_n>\n\n"
              << "  2D Ising Model end-to-end pipeline.\n"
              << "  Generates Monte Carlo spin configurations via Metropolis-Hastings,\n"
              << "  then executes analysis notebooks and populates the plots/ folder.\n\n"
              << "Arguments (all required, order-independent):\n"
              << "  --Tc_seed    <double>  Critical temperature estimate used to label microstates.\n"
              << "                         Configurations with T < Tc_seed are labeled 0 (ordered),\n"
              << "                         T >= Tc_seed are labeled 1 (disordered).\n"
              << "                         The exact 2D Ising Tc is ~2.269.\n\n"
              << "  --Num_Samples <int>   Number of Monte Carlo samples per temperature.\n\n"
              << "  --L          <int>    Linear lattice size. System contains L*L spins.\n\n"
              << "  --T_ranges   <T_min> <T_max> <T_n>\n"
              << "                         Temperature grid, equivalent to np.linspace(T_min, T_max, T_n).\n"
              << "                         T_min and T_max are doubles; T_n is an integer >= 2.\n\n"
              << "Example:\n"
              << "  ./main --Tc_seed 2.269 --Num_Samples 1000 --L 10 --T_ranges 0.1 4.0 8\n\n"
              << "Output:\n"
              << "  data/   — sampled spin configurations and labeled datasets per temperature\n"
              << "  plots/  — energy/magnetization distribution plots and ML training figures\n"
              << "  Run 'streamlit run streamlit_app.py' to view the interactive dashboard.\n\n";
}
