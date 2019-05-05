#include <iostream>
#include <ctime>
#include "single-layer-perceptron.hpp"
#include "multi-layer-perceptron.hpp"

using Vector = Eigen::VectorXd;

int main()
{
    constexpr int N = 4, H = 8, M = 4, T = 3;
//    MultiLayerPerceptron::initialiseRandomSeed(time(nullptr));
//    MultiLayerPerceptron mlp {{N, 4, M}};


    MultiLayerPerceptron::initialiseRandomSeed(static_cast<int>(time
            (nullptr)));
    MultiLayerPerceptron slp {{ N, H, M }};


    std::vector<TrainingExample> trainingData
            {
                    {
                            (Vector { N } << 1.0, 0.0, 0.0, 0.0).finished(),
                            (Vector { M } << 1.0, 0.0, 0.0, 0.0).finished()
                    },
                    {
                            (Vector { N } << 0.0, 1.0, 0.0, 0.0).finished(),
                            (Vector { M } << 0.0, 1.0, 0.0, 0.0).finished()
                    },
                    {
                            (Vector { N } << 0.0, 0.0, 1.0, 0.0).finished(),
                            (Vector { M } << 0.0, 0.0, 1.0, 0.0).finished()
                    },
                    {
                            (Vector { N } << 0.0, 0.0, 0.0, 1.0).finished(),
                            (Vector { M } << 0.0, 0.0, 0.0, 1.0).finished()
                    }
            };

    int const numberOfEpochs = 1000;
    double const errorGoal = 0.0;
    double const learningRate = 1.0;
    double const momentum = 1.0;
    slp.train(trainingData, numberOfEpochs, errorGoal, learningRate,
              momentum);

    using std::cout;
    using std::endl;
    cout << endl;
    cout << slp((Vector { N } << 1.0, 0.0, 0.0, 0.0).finished()) << "\n\n";
    cout << slp((Vector { N } << 0.0, 1.0, 0.0, 0.0).finished()) << "\n\n";
    cout << slp((Vector { N } << 0.0, 0.0, 1.0, 0.0).finished()) << "\n\n";
    cout << slp((Vector { N } << 0.0, 0.0, 0.0, 1.0).finished()) << "\n\n";


    return 0;
}
