///////////////////////////////////////////////////////////////////// | Includes
#include "multi-layer-perceptron.hpp"
#include <iostream>
#include <ctime>

using namespace std;
using namespace NeuralNetworks;

#include "training-example.hpp"
#include <fstream>

using Vector = Eigen::VectorXd;

struct TrainingExampleClass
{
    std::string className;
    std::vector<TrainingExample> trainingExamples;
};

#include <sstream>
#include <string_view>

std::vector<std::string_view> split
        (std::string_view stringView,
         std::string_view delimiters = " ")
{
    std::vector<std::string_view> output;

    for (auto first = stringView.data(),
                 second = stringView.data(),
                 last = first + stringView.size();
         second != last && first != last;
         first = second + 1)
    {
        second = std::find_first_of(first,
                                    last,
                                    std::cbegin(delimiters),
                                    std::cend(delimiters));

        if (first != second)
            output.emplace_back(first,
                                second - first);
    }

    return output;
}

std::vector<TrainingExampleClass> readTrainingExamplesFromCsvFile
        (std::string filename, int const numberOfInputs, int const
        numberOfClasses)
{
    std::vector<TrainingExampleClass> trainingExampleClasses;

    std::ifstream file(filename, std::ios::in);
    {
        std::string line;
        while (std::getline(file, line) && !line.empty())
        {
            std::vector<std::string_view> tokens = split(line, ",");

            TrainingExample trainingExample;
            trainingExample.inputs = Vector { numberOfInputs };
            for (int i = 0; i < numberOfInputs; i++)
            {
                // TODO: Rewrite that monster!
                double inputNumber;
                std::stringstream str;
                str << tokens.at(i);
                str >> inputNumber;
                trainingExample.inputs(i) = inputNumber;
            }

            bool isClassAlreadyIn = false;
            for (auto const &i : trainingExampleClasses)
            {
                if (i.className == tokens.back())
                {
                    isClassAlreadyIn = true;
                    break;
                }
            }

            trainingExample.outputs = Vector::Zero(numberOfClasses);

            if (isClassAlreadyIn)
            {
                for (int i = 0; i < trainingExampleClasses.size(); i++)
                    if (trainingExampleClasses[i].className == tokens.back())
                    {
                        trainingExample.outputs(i) = 1.0;
                        trainingExampleClasses[i].trainingExamples.push_back
                                (trainingExample);
                    }
            } else
            {
                TrainingExampleClass newClass;
                trainingExample.outputs(trainingExampleClasses.size()) = 1.0;
                newClass.className = tokens.back();
                newClass.trainingExamples.push_back(trainingExample);
                trainingExampleClasses.push_back(newClass);
            }
        }
    }
    return trainingExampleClasses;
}

////////////////////////////////////////////////////////////// | Project: iad-2a
int main()
{



    std::vector<TrainingExampleClass> trainingExampleClasses
            = readTrainingExamplesFromCsvFile
                    ("E:\\Studia\\semestr-4\\IAD\\iad-2a\\src\\data\\iris"
                     ".data", 4, 3);

//    for (auto const &i : trainingExampleClasses)
//    {
//        for (auto const &j : i.trainingExamples)
//        {
//            cout << "> inputs\n" << j.inputs
//                 << "> outputs\n" << j.outputs
//                 << endl;
//        }
//    }

    double ratio = 0.8;
    std::vector<TrainingExample> trainingExamples;
    std::vector<TrainingExample> testingExamples;
    for (auto const &trainingExampleClass : trainingExampleClasses)
        for (int i = 0;
             i < trainingExampleClass.trainingExamples.size();
             i++)
            if (i < trainingExampleClass.trainingExamples.size() * ratio)
                trainingExamples.push_back
                        (trainingExampleClass.trainingExamples.at(i));
            else
                testingExamples.push_back
                        (trainingExampleClass.trainingExamples.at(i));

    //-------

    constexpr int
            n = 4,
            h = 2,
            m = 4;

    MultiLayerPerceptron::initialiseRandomNumberGenerator(static_cast<int>
                                                          (time(nullptr)));
    MultiLayerPerceptron mlp {{ 4, 16, 16, 3 },
                              std::vector<bool>(3, true) };

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

    int const numberOfEpochs = 500;
    double const costGoal = 0.01;
    double const learningCoefficient = 0.01;
    double const learningCoefficientChange = -0.0075;
    double const momentumCoefficient = 0.75;
    bool const shuffleTrainingData = true;
    int const epochInterval = 50;

    MultiLayerPerceptron::TrainingResults trainingResults
            = mlp.train(trainingExamples,
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
            = mlp.test(trainingExamples);

    std::cout << "testing results" << std::endl;
    std::cout << "global cost: " << testingResults.globalCost << std::endl;
//    for (auto const &i : testingResults.testingResultsPerExample)
//    {
//        std::cout << "neurons\n";
//        for (auto const &j : i.neurons)
//            std::cout << ">" << std::endl
//                      << j << std::endl;
//        std::cout << "targets\n";
//        std::cout << i.targets << std::endl;
//        std::cout << "errors\n";
//        for (auto const &j : i.errors)
//            std::cout << ">" << std::endl
//                      << j << std::endl;
//        std::cout << "cost\n";
//        std::cout << i.cost << std::endl;
//    }

    int avg = 0;
    for (auto const &i : testingResults.testingResultsPerExample)
    {
        Vector::Index classNumber;
        i.neurons.back().array().maxCoeff(&classNumber);
        avg += (int) i.targets(classNumber);
    }
    cout << "dokl: " << (double) avg / trainingExamples.size() * 100 << " %" <<
         endl;

    mlp.saveToFile("multi-layer-perceptron.mlp");
    MultiLayerPerceptron loadedFromFile { "multi-layer-perceptron.mlp" };

    using std::cout;
    using std::endl;
    cout << endl;
    cout << mlp((Vector { n } << 5.0, 3.3, 1.4, 0.2).finished()) << "\n\n";
//    cout << loadedFromFile((Vector { n } << 1.0, 0.0, 0.0, 0.0).finished()) <<
//         "\n\n";
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
