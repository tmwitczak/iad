///////////////////////////////////////////////////////////////////// | Includes
#include "multi-layer-perceptron.hpp"
#include <iostream>
#include <ctime>

using namespace std;
using namespace NeuralNetworks;
using Vector = Eigen::VectorXd;

////////////////////////////////////////////////////////////// | Project: iad-2a
int main()
{
    constexpr int
            n = 4,
            h = 2,
            m = 4;

    MultiLayerPerceptron::initialiseRandomNumberGenerator(static_cast<int>
                                                          (time(nullptr)));
    MultiLayerPerceptron mlp {{ n, h, m },
                              std::vector<bool>(2, true) };

    std::vector<TrainingExample> trainingData
            {
                    {
                            (Vector { n } << 1.0, 0.0, 0.0, 0.0).finished(),
                            (Vector { m } << 1.0, 0.0, 0.0, 0.0).finished()
                    },
                    {
                            (Vector { n } << 0.0, 1.0, 0.0, 0.0).finished(),
                            (Vector { m } << 0.0, 1.0, 0.0, 0.0).finished()
                    },
                    {
                            (Vector { n } << 0.0, 0.0, 1.0, 0.0).finished(),
                            (Vector { m } << 0.0, 0.0, 1.0, 0.0).finished()
                    },
                    {
                            (Vector { n } << 0.0, 0.0, 0.0, 1.0).finished(),
                            (Vector { m } << 0.0, 0.0, 0.0, 1.0).finished()
                    }
            };

    int const numberOfEpochs = 10000;
    double const costGoal = 0.00001;
    double const learningCoefficient = 0.2;
    double const learningCoefficientChange = -0.1;
    double const momentumCoefficient = 0.8;
    bool const shuffleTrainingData = true;
    int const epochInterval = 500;

    MultiLayerPerceptron::TrainingResults trainingResults
            = mlp.train(trainingData,
                        numberOfEpochs, costGoal,
                        learningCoefficient, learningCoefficientChange,
                        momentumCoefficient,
                        shuffleTrainingData,
                        epochInterval);

    std::cout << "training results" << std::endl;
    std::cout << "epoch interval: " << trainingResults.epochInterval
              << std::endl;
    for (auto const &i : trainingResults.costPerEpochInterval)
        std::cout << i << std::endl;

    MultiLayerPerceptron::TestingResults testingResults
            = mlp.test(trainingData);

    std::cout << "testing results" << std::endl;
    std::cout << "global cost: " << testingResults.globalCost << std::endl;
    for (auto const &i : testingResults.testingResultsPerExample)
    {
        std::cout << "neurons\n";
        for (auto const &j : i.neurons)
            std::cout << ">" << std::endl
                      << j << std::endl;
        std::cout << "targets\n";
        std::cout << i.targets << std::endl;
        std::cout << "errors\n";
        for (auto const &j : i.errors)
            std::cout << ">" << std::endl
                      << j << std::endl;
        std::cout << "cost\n";
        std::cout << i.cost << std::endl;
    }

    mlp.saveToFile("multi-layer-perceptron.mlp");
    MultiLayerPerceptron loadedFromFile { "multi-layer-perceptron.mlp" };

    using std::cout;
    using std::endl;
    cout << endl;
    cout << mlp((Vector { n } << 1.0, 0.0, 0.0, 0.0).finished()) << "\n\n";
    cout << loadedFromFile((Vector { n } << 1.0, 0.0, 0.0, 0.0).finished()) <<
         "\n\n";
//    cout << mlp((Vector { n } << 0.0, 1.0, 0.0, 0.0).finished()) << "\n\n";
//    cout << mlp((Vector { n } << 0.0, 0.0, 1.0, 0.0).finished()) << "\n\n";
//    cout << mlp((Vector { n } << 0.0, 0.0, 0.0, 1.0).finished()) << "\n\n";


//    using namespace NeuralNetworks;
//    Sigmoid sigmoid;
//    std::unique_ptr<Cloneable<ActivationFunction>> cloneable
//            = std::make_unique<Sigmoid>();
//    Sigmoid sigmoidCopyAssignment = sigmoid;
//    PerceptronLayer layer1 { 3, 4, Sigmoid {}};
//    PerceptronLayer layer2 { 3, 4, RectifiedLinearUnit {}};
//    PerceptronLayer layer3 { 3, 4 };


    return 0;
}


//    PerceptronLayer::initialiseRandomSeed(time(nullptr));
//    std::vector<PerceptronLayer> layers;
//    layers.emplace_back(4, 5, Sigmoid {});
//    layers.emplace_back(5, 4, Sigmoid {});
//    TrainingExample example1
//            { (Vector { 4 } << 1.0, 0.0, 0.0, 0.0).finished(),
//              (Vector { 4 } << 1.0, 0.0, 0.0, 0.0).finished() };
//    TrainingExample example2
//            { (Vector { 4 } << 0.0, 1.0, 0.0, 0.0).finished(),
//              (Vector { 4 } << 0.0, 1.0, 0.0, 0.0).finished() };
//    std::vector<Vector> outputsPerLayer { 2 };
//    std::vector<Vector> errorsPerLayer { 2 };
//    for (int i = 0; i < 10000; i++)
//    {
//        outputsPerLayer.at(0) = layers.at(0).feedForward(example1.inputs);
//        outputsPerLayer.at(1) = layers.at(1).feedForward(outputsPerLayer.at(0));
//        errorsPerLayer.at(1) = example1.outputs - outputsPerLayer.at(1);
//        errorsPerLayer.at(0) = layers.at(1).backpropagate
//                (outputsPerLayer.at(0), errorsPerLayer.at(1));
//        layers.at(0).update(example1.inputs, errorsPerLayer.at(0),
//                            outputsPerLayer.at(0), 1.0);
//        layers.at(1).update(outputsPerLayer.at(0), errorsPerLayer.at(1),
//                            outputsPerLayer.at(1), 1.0);
//
//        double cost = errorsPerLayer.at(1).array().square().sum();
//
//        outputsPerLayer.at(0) = layers.at(0).feedForward(example2.inputs);
//        outputsPerLayer.at(1) = layers.at(1).feedForward(outputsPerLayer.at(0));
//        errorsPerLayer.at(1) = example2.outputs - outputsPerLayer.at(1);
//        errorsPerLayer.at(0) = layers.at(1).backpropagate
//                (outputsPerLayer.at(0), errorsPerLayer.at(1));
//        layers.at(0).update(example2.inputs, errorsPerLayer.at(0),
//                            outputsPerLayer.at(0), 1.0);
//        layers.at(1).update(outputsPerLayer.at(0), errorsPerLayer.at(1),
//                            outputsPerLayer.at(1), 1.0);
//
//        cost += errorsPerLayer.at(1).array().square().sum();
//        cost /= 2.0;
//        if (cost < 1e-12)
//            break;
//    }
//    cout << "inputs:\n\n" << example2.inputs << "\n\n";
//    cout << "outputs:\n\n" << outputsPerLayer.at(1) << "\n\n";
//    cout << "errors:\n\n" << errorsPerLayer.at(1) << "\n\n";
//    cout << "targets:\n\n" << example2.outputs << "\n\n";
//-------------------------------------------------------------
