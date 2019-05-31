///////////////////////////////////////////////////////////////////// | Includes
#include "neural-network.hpp"
#include "k-nearest-neighbours.hpp"
#include "training-example.hpp"
#include "identity.hpp"
#include "radial-basis-function-layer.hpp"
#include <iostream>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <string_view>
#include <string>
#include <limits>
#include <iomanip>
#include <fstream>
#include <utility>
#include <list>

using namespace std;
using namespace NeuralNetworks;
using Vector = Eigen::VectorXd;

constexpr int IOMANIP_WIDTH = 38;


std::list<std::string_view> split
        (std::string_view stringView,
         std::string_view delimiters)
{
    std::list<std::string_view> output;

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

std::pair<std::vector<TrainingExample>, std::vector<std::string>>
readTrainingExamplesFromCsvFile
        (std::string const &filename)
{
    std::ios::sync_with_stdio(false);

    std::list<std::string> classLabels;
    std::list<TrainingExample> trainingExamples;

    std::ifstream file(filename, std::ios::in);
    {
        int lineNumber = 1;
        std::string line;
        bool firstLine = true;
        int numberOfOutputs = 0;
        int numberOfInputs = 0;

        while (std::getline(file, line)
               && !line.empty())
        {
            std::cout << "line: " << lineNumber++ << "\r";

            std::list<std::string_view> tokens
                    = split(line, ",");

            if (firstLine)
            {
                for (auto const &token
                        : tokens)
                    if (token != " ")
                        classLabels.emplace_back(token);

                firstLine = false;
                numberOfInputs = tokens.size() - classLabels.size();
                numberOfOutputs = classLabels.size();
            }
            else
            {
                // Load training example
                TrainingExample trainingExample { Vector { numberOfInputs },
                                                  Vector { numberOfOutputs }};

                // Read inputs
                auto token = tokens.cbegin();

                for (int i = 0, j = 0;
                     i < numberOfInputs;
                     i++, j++, ++token)
                {
                    trainingExample.inputs(i) = std::stod(token->data());
                }

                // Read outputs
                for (int i = 0, j = numberOfInputs;
                     i < numberOfOutputs;
                     i++, j++, ++token)
                {
                    trainingExample.outputs(i) = std::stod(token->data());
                }

                // Save the training example
                trainingExamples.emplace_back(trainingExample);
            }
        }
    }

    return std::make_pair<std::vector<TrainingExample>, std::vector<std::string>>
            ({ std::make_move_iterator(std::begin(trainingExamples)),
               std::make_move_iterator(std::end(trainingExamples)) },
             { std::make_move_iterator(std::begin(classLabels)),
               std::make_move_iterator(std::end(classLabels)) });
}


std::vector<std::string>::const_iterator askUserForInput
        (std::string_view const &question,
         std::vector<std::string> const &options)
{
    int optionIndex = 1;
    for (auto const &option
            : options)
    {
        std::cout << setw(IOMANIP_WIDTH)
                  << optionIndex << " " << '|' << " " << option << "\n";
        ++optionIndex;
    }

    std::cout << setw(IOMANIP_WIDTH) << question << " " << '|' << " ";
    int userInput;
    std::cin >> userInput;

    for (int i = 0; i < options.size(); i++)
        if (i == userInput)
            return options.cbegin() + i;

    std::cout << "Wrong input!\n";
    exit(1);
}

void saveErrorToFile
        (std::string const &filename,
         NeuralNetwork::TrainingResults const &trainingResults)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        file << i * trainingResults.epochInterval << ","
             << trainingResults.costPerEpochInterval.at(i) << "\n";
}

void saveTrainingAccuracyToFile
        (std::string const &filename,
         NeuralNetwork::TrainingResults const &trainingResults)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        file << i * trainingResults.epochInterval << ","
             << trainingResults.accuracyTraining.at(i) << "\n";
}

void saveTestingAccuracyToFile
        (std::string const &filename,
         NeuralNetwork::TrainingResults const &trainingResults)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        file << i * trainingResults.epochInterval << ","
             << trainingResults.accuracyTesting.at(i) << "\n";
}

std::string braces
        (std::string const &str)
{
    return "[" + str + "]";
}

template <typename TestingResults>
void printAccuracy
        (std::ostream &stream,
         std::vector<TrainingExample> const &testingExamples,
         std::vector<std::string> const &testingClassLabels,
         std::string const &testFilename,
         TestingResults const &testingResults)
{
    // Accuracy
    int globalNumberOfAccurateClassifications = 0;
    std::vector<int> accurateClassificationsPerClass
            (testingClassLabels.size(), 0);
    std::vector<int> classCount
            (testingClassLabels.size(), 0);

    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero
            (testingClassLabels.size(),
             testingClassLabels.size());
    std::vector<Eigen::MatrixXd> confusionPerClass
            (testingClassLabels.size(), Eigen::MatrixXd::Zero(2, 2));

    for (auto const &testingResultsPerExample
            : testingResults.testingResultsPerExample)
    {
        Vector::Index predictedClass, actualClass;
        testingResultsPerExample.neurons
                .back().array().maxCoeff(&predictedClass);
        testingResultsPerExample.targets.array().maxCoeff(&actualClass);

        confusionMatrix(predictedClass, actualClass)++;

        for (int i = 0; i < confusionPerClass.size(); i++)
        {
            // True positive
            if (actualClass == i && predictedClass == i)
                confusionPerClass.at(i)(0, 0)++;
                // True negative
            else if (actualClass != i && predictedClass != i)
                confusionPerClass.at(i)(1, 1)++;
                // False positive (type I error)
            else if (actualClass != i && predictedClass == i)
                confusionPerClass.at(i)(0, 1)++;
                // False negative (type II error)
            else if (actualClass == i && predictedClass != i)
                confusionPerClass.at(i)(1, 0)++;
        }

        globalNumberOfAccurateClassifications
                += (int) (predictedClass == actualClass);
        accurateClassificationsPerClass.at((int) actualClass)
                += (int) (predictedClass == actualClass);
        classCount.at((int) actualClass)++;
    }
    double globalAccuracy = (double) globalNumberOfAccurateClassifications
                            / testingExamples.size();

    stream << ">> " << testFilename << "\n";

    stream << "\n > Global confusion matrix: " << "\n";
    for (int i = 0; i < confusionMatrix.rows(); i++)
    {
        if (i == 0)
        {
            stream << setw(20) << " ";
            for (auto const &testingClassLabel
                    : testingClassLabels)
                stream
                        << setw(braces(testingClassLabel)
                                        .length() + 5)
                        << braces(testingClassLabel) << " ";
            stream << "\n";
        }
        stream << setw(20)
               << braces(testingClassLabels.at(i));

        for (int j = 0; j < confusionMatrix.cols(); j++)
        {
            stream
                    << setw(braces(testingClassLabels.at(j))
                                    .length() + 5) << confusionMatrix(i, j)
                    << " ";
        }

        stream << "\n";
    }

    stream << "\n\n" << setw(IOMANIP_WIDTH) << "Total population " << " " << '|'
           << " "
           << testingExamples.size()
           << "\n"
           << "\n" << setw(IOMANIP_WIDTH) << "Accuracy " << " " << '|' << " "
           << globalAccuracy * 100 << " %" << "\n";


    // Confusion matrices per class
    for (int k = 0; k < confusionPerClass.size(); k++)
    {
        stream << "\n\n\n";
        stream << "> " << braces(testingClassLabels.at(k)) << "\n";

        for (int i = 0; i < confusionPerClass.at(k).rows(); i++)
        {
            if (i == 0)
            {
                stream << setw(20) << " "
                       << setw(8 + 5) << "[Positive]" << " "
                       << setw(8 + 5) << "[Negative]"
                       << "\n";
            }
            stream << setw(20) << (i == 0 ? "[Positive]" : "[Negative]");

            for (int j = 0; j < confusionPerClass.at(k).cols(); j++)
                stream
                        << setw(8 + 5) << confusionPerClass.at(k)(i, j)
                        << " ";
            stream << "\n";
        }

        stream << "\n";

        // Statistics
        int const totalPopulation = confusionPerClass.at(k).sum();

        int const &truePositive = confusionPerClass.at(k)(0, 0);
        int const &trueNegative = confusionPerClass.at(k)(1, 1);
        int const &falsePositive = confusionPerClass.at(k)(0, 1);
        int const &falseNegative = confusionPerClass.at(k)(1, 0);

        int const predictedPositive = truePositive + falsePositive;
        int const predictedNegative = trueNegative + falseNegative;
        int const actualPositive = truePositive + falseNegative;
        int const actualNegative = trueNegative + falsePositive;

        double const positivePredictiveValue
                = double(truePositive) / predictedPositive;
        double const falseDiscoveryRate
                = double(falsePositive) / predictedPositive;
        double const falseOmissionRate
                = double(falseNegative) / predictedNegative;
        double const negativePredictiveValue
                = double(trueNegative) / predictedNegative;

        double const truePositiveRate
                = double(truePositive) / actualPositive;
        double const falsePositiveRate
                = double(falsePositive) / actualNegative;
        double const falseNegativeRate
                = double(falseNegative) / actualPositive;
        double const trueNegativeRate
                = double(trueNegative) / actualNegative;

        double const accuracy
                = double(truePositive + trueNegative) / totalPopulation;

        stream << "\n" << setw(IOMANIP_WIDTH)
               << "Total population " << '|' << " " << totalPopulation;
        stream << "\n";
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "True positive " << '|' << " " << truePositive;
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "True negative " << '|' << " " << trueNegative;
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "False positive (type I error) " << '|' << " "
               << falsePositive;
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "False negative (type II error) " << '|' << " "
               << falseNegative;
        stream << "\n";
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "Correct positive predictions " << '|' << " "
               << positivePredictiveValue * 100 << " %";
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "Correct negative predictions " << '|' << " "
               << negativePredictiveValue * 100 << " %";
        stream << "\n";
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "Correct positive classifications " << '|' << " "
               << truePositiveRate * 100 << " %";
        stream << "\n" << setw(IOMANIP_WIDTH)
               << "Correct negative classifications " << '|' << " "
               << trueNegativeRate * 100 << " %";
        stream << "\n"
               << "\n" << setw(IOMANIP_WIDTH) << "Accuracy " << '|' << " "
               << accuracy * 100 << " %"
               << std::endl;
    }
}

void printTestingResults
        (NeuralNetwork const &multiLayerPerceptron,
         std::vector<TrainingExample> const &testingExamples,
         std::vector<std::string> const &testingClassLabels,
         std::string const &testFilename,
         std::string const &perceptronFilename,
         bool const additionalTestingDump)
{
    NeuralNetwork::TestingResults testingResults
            = multiLayerPerceptron.test(testingExamples);

    std::cout << "\r" << std::string(80, ' ') << "\r";

    printAccuracy(std::cout, testingExamples, testingClassLabels,
                  testFilename, testingResults);

    std::string dirName = "testing-results_" + perceptronFilename;
    {
        //system(("rmdir \"" + dirName + "\" /s /q").data());
        system(("if not exist \"" + dirName + "\" mkdir \"" + dirName + "\"")
                       .data());
        std::ofstream file(dirName + "/" + perceptronFilename
                           + ".analysis", std::ios::out | std::ios::trunc);
        printAccuracy(file, testingExamples, testingClassLabels,
                      testFilename, testingResults);
    }

    // Additional info to file
    if (additionalTestingDump)
    {
        std::ofstream file(dirName + "/" + perceptronFilename
                           + ".information",
                           std::ios::out | std::ios::trunc);

        file << "> [Global cost]: " << testingResults.globalCost << "\n";
        for (auto const &i : testingResults.testingResultsPerExample)
        {
            file << "\n\n\n>>>>>>>>>>>>>>>>>>>>>>" << "\n";

            file << "> [Neurons]\n";
            for (auto const &j : i.neurons)
                file << "\n"
                     << j << "\n";

            file << "\n> [Targets]\n";
            file << "\n"
                 << i.targets << "\n";

            file << "\n> [Errors]\n";
            for (auto j = i.errors.crbegin(); j != i.errors.crend(); ++j)
                file << "\n"
                     << *j << "\n";

            file << "\n> [Cost]\n";
            file << "\n"
                 << i.cost << "\n";
        }
    }
}

void printTestingResults(KNearestNeighbours const &kNearestNeighbours,
                         std::vector<TrainingExample> const &testingExamples,
                         std::vector<std::string> const &
                         testingClassLabels,
                         std::string const &testFilename,
                         std::string const &neighboursFilename,
                         bool const additionalTestingDump)
{
    KNearestNeighbours::TestingResults testingResults
            = kNearestNeighbours.test(testingExamples);

    std::cout << "\r" << std::string(80, ' ') << "\r";

//    NeuralNetwork::TestingResults convertedTestingResults;
//    convertedTestingResults.globalCost = testingResults.globalCost;
//    for (auto const &i : testingResults.testingResultsPerExample)
//    {
//        convertedTestingResults
//                .testingResultsPerExample
//                .push_back({ i.neurons,
//                             i.targets,
//                             { i.errors },
//                             i.cost });
//    }

    printAccuracy(std::cout, testingExamples, testingClassLabels,
                  testFilename, testingResults);

    std::string dirName = "testing-results_" + neighboursFilename;
    {
        //system(("rmdir \"" + dirName + "\" /s /q").data());
        system(("if not exist \"" + dirName + "\" mkdir \"" + dirName + "\"")
                       .data());
        std::ofstream file(dirName + "/" + neighboursFilename
                           + ".analysis", std::ios::out | std::ios::trunc);
        printAccuracy(file, testingExamples, testingClassLabels,
                      testFilename, testingResults);
    }

    // Additional info to file
    if (additionalTestingDump)
    {
        std::ofstream file(dirName + "/" + neighboursFilename
                           + ".information", std::ios::out | std::ios::trunc);

        file << "> [Global cost]: " << testingResults.globalCost << "\n";
        for (auto const &i : testingResults.testingResultsPerExample)
        {
            file << "\n\n\n>>>>>>>>>>>>>>>>>>>>>>" << "\n";

            file << "> [Neurons]\n";
            for (auto const &j : i.neurons)
                file << "\n"
                     << j << "\n";

            file << "\n> [Targets]\n";
            file << "\n"
                 << i.targets << "\n";

            file << "\n> [Errors]\n";
            file << "\n"
                 << i.errors << "\n";

            file << "\n> [Cost]\n";
            file << "\n"
                 << i.cost << "\n";
        }
    }
}
//class SigmoidActivation
//{
//public:
//    SigmoidActivation()
//        : ptr {std::make_unique<Sigmoid>()}
//    {
//    }
//    operator std::unique_ptr<ActivationFunction>()
//    {
//        return std::move(ptr);
//    }
//private:
//    std::unique_ptr<ActivationFunction> ptr;
//};



////////////////////////////////////////////////////////////// | Project: iad-2a
int main
        ()
{
    NeuralNetwork::initialiseRandomNumberGenerator(time(nullptr));

    std::vector<std::unique_ptr<NeuralNetworkLayer>> layers;
    layers.emplace_back(RadialBasisFunctionLayer { 1, 100 });
    layers.emplace_back(AffineLayerWithBias { 100, 100,
                                              ParametricRectifiedLinearUnit
                                                      { 0.01 }});
    layers.emplace_back(AffineLayerWithBias { 100, 1 });
    NeuralNetwork neuralNetwork(std::move(layers));

    std::vector<TrainingExample> trainingExamples;
    for (int i = 0; i < 1000; i++)
    {
        Vector input { 1 };
        input << (double(i) / 1000.0);

        Vector output { 1 };
        output << std::sqrt(input(0));

        TrainingExample tr = { input, output };
        trainingExamples.push_back(tr);
    }

    std::cout << trainingExamples[500].inputs << "\n\n"
              << trainingExamples[500].outputs << "\n\n";

    NeuralNetwork::TrainingResults trainingResults =
            neuralNetwork.train
                    (trainingExamples,
                     trainingExamples,
                     1000,
                     0.0, 0.01, 0.0, 0.8, true, 10);

    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        std::cout << i * 10 << " | " << trainingResults
                .costPerEpochInterval[i] << "\n";
    //std::cout << "\n\n" << trainingResults.costPerEpochInterval.back() <<
    // "\n";

    Vector test { 1 };
    test << 0.5;
    std::cout << "\n\n" << neuralNetwork(test) << "\n";
//
//    Vector input { 4 };
//    Vector target { 4 };
//    Vector output1 { 7 };
//    Vector output2 { 4 };
//    Vector error1 { 7 };
//    Vector error2 { 4 };
//    for (int i = 0; i < 10000; i++)
//    {
//        input << 1.0, 0.5, 0.25, 0.125;
////        std::cout << "input\n\n" << input << "\n\n";
//        target << 0.1, 0.2, 0.3, 0.4;
////        std::cout << "target\n\n" << target << "\n\n";
//        output1 = radialBasisFunctionLayer1(input);
////        std::cout << "output1\n\n" << output1 << "\n\n";
//        output2 = radialBasisFunctionLayer2(output1);
////        std::cout << "output2\n\n" << output2 << "\n\n";
//        error2 = target - output2;
////        std::cout << "error2\n\n" << error2 << "\n\n";
//        error1 = radialBasisFunctionLayer2.backpropagate(output1,
//                                                         error2, output2,
//                                                         radialBasisFunctionLayer2
//                                                                 .calculateOutputsDerivative(
//                                                                         output2));
////        std::cout << "error1\n\n" << error1 << "\n\n";
//
//        std::cout << "cost: " << 0.5 * (error2.array() * error2.array()).sum()
//                  << "\n";
//
//        radialBasisFunctionLayer1.calculateNextStep(input, error1, output1,
//                                                    radialBasisFunctionLayer1
//                                                            .calculateOutputsDerivative(
//                                                                    output1));
//        radialBasisFunctionLayer1.update(0.01, 0.9);
//        radialBasisFunctionLayer2.calculateNextStep(output1, error2, output2,
//                                                    radialBasisFunctionLayer2
//                                                            .calculateOutputsDerivative(
//                                                                    output2));
//        radialBasisFunctionLayer2.update(0.01, 0.9);
//    }
//
//    std::cout << "\ninput\n\n" << input << "\n";
//    std::cout << "\noutput\n\n" << output2 << "\n";
//    std::cout << "\nerror\n\n" << error2 << "\n";

    return 0;
    //------------------------------
    std::cout << std::string(79, '/') << std::endl;
    std::vector<std::string> functions
            { "sqrt(x)",
              "sin(x)",
              "sin(x1 * x2) + cos(3*(x1 - x2))" };
    auto const chosenFunction
            = askUserForInput("Choose function", functions);

    std::cout << "\n";
    std::cout << std::string(79, '-') << std::endl;
    std::vector<std::string> architectures
            { std::string("Affine (w/bias, sigmoid)\n")
              + std::string(IOMANIP_WIDTH + 4, ' ')
              + std::string(" -> Affine (w/bias, identity)"),
              std::string("RBF\n")
              + std::string(IOMANIP_WIDTH + 4, ' ')
              + std::string(" -> Affine (w/bias, identity)"),
              std::string("RBF\n")
              + std::string(IOMANIP_WIDTH + 4, ' ')
              + std::string(" -> Affine (w/bias, sigmoid)\n")
              + std::string(IOMANIP_WIDTH + 4, ' ')
              + std::string(" -> Affine (w/bias, identity)") };
    auto const architecture
            = askUserForInput("Choose network architecture", architectures);

    std::cout << std::endl;
    std::cout << std::string(79, '-') << std::endl;
    std::vector<std::string> modes
            { "Training",
              "Testing" };
    auto const mode
            = askUserForInput("Choose mode", modes);

    std::string neuralNetworkFilename;
    std::cout << "\n";
    std::cout << std::string(79, '-') << std::endl;
    std::cout << setw(IOMANIP_WIDTH) << "Neural network filename"
              << '|' << " ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(cin, neuralNetworkFilename);

    // Create training and testing examples for chosen function
//        auto const[trainingExamples, trainingClassLabels]
//        = readTrainingExamplesFromCsvFile
//                (dataSetTrainingFilenames[dataSet]);
//
//        auto const[testingExamples, testingClassLabels]
//        = readTrainingExamplesFromCsvFile
//                (dataSetTestingFilenames[dataSet]);

    // Prepare perceptron
    NeuralNetwork::initialiseRandomNumberGenerator
            (static_cast<int>(time(nullptr)));

    // Do the math :)
    if (mode == modes.cbegin())
    {
        // Neuron number
        std::string hiddenLayerNeuronNumber;
        std::cout << setw(IOMANIP_WIDTH) << "Neurons in hidden layers"
                  << '|' << " ";
        std::getline(std::cin, hiddenLayerNeuronNumber);
    }
//        std::vector<int> layersNeurons;
//        layersNeurons.push_back
//                (static_cast<int>(trainingExamples.at(0).inputs.size()));
//
//        for (auto const &neurons : split(hiddenLayerNeuronNumber, " "))
//            layersNeurons.push_back(std::stoi(neurons.data()));
//
//        layersNeurons.push_back
//                (static_cast<int>(trainingExamples.at(0).outputs.size()));


//        NeuralNetwork multiLayerPerceptron
//                { layersNeurons,
//                  biasesPerLayer };
//
//            // Get parameters
//            int numberOfEpochs;
//            double costGoal;
//            double learningCoefficientStart;
//            double learningCoefficientEnd;
//            double momentumCoefficient;
//            bool shuffleTrainingData;
//            int epochInterval;
//
//            std::cout << setw(IOMANIP_WIDTH) << "Number of epochs" << " " << '|' << " ";
//            std::cin >> numberOfEpochs;
//            std::cout << setw(IOMANIP_WIDTH) << "Cost goal" << " " << '|' << " ";
//            std::cin >> costGoal;
//            std::cout << setw(IOMANIP_WIDTH) << "Learning coefficient (start)"
//                      << " " << '|' << " ";
//            std::cin >> learningCoefficientStart;
//            std::cout << setw(IOMANIP_WIDTH) << "Learning coefficient (end)"
//                      << " " << '|' << " ";
//            std::cin >> learningCoefficientEnd;
//            std::cout << setw(IOMANIP_WIDTH) << "Momentum coefficient" << " " << '|' << " ";
//            std::cin >> momentumCoefficient;
//            std::cout << setw(IOMANIP_WIDTH) << "Shuffle training data"
//                      << " " << '|' << " ";
//            std::cin >> shuffleTrainingData;
//            std::cout << setw(IOMANIP_WIDTH) << "Epoch interval" << " " << '|' << " ";
//            std::cin >> epochInterval;
//
//            // Train
//            NeuralNetwork::TrainingResults trainingResults
//                    = multiLayerPerceptron.train(trainingExamples,
//                                                 testingExamples,
//                                                 numberOfEpochs,
//                                                 costGoal,
//                                                 learningCoefficientStart,
//                                                 learningCoefficientEnd
//                                                 - learningCoefficientStart,
//                                                 momentumCoefficient,
//                                                 shuffleTrainingData,
//                                                 epochInterval);
//
//            // Document learning
//            multiLayerPerceptron.saveToFile(perceptronFilename);
//
//            std::string dirName = "training-results_" + perceptronFilename;
//            std::string plotErrorName = dirName + "/"
//                                        + perceptronFilename + ".cost";
//            std::string plotTrainingAccuracyName = dirName + "/"
//                                                   + perceptronFilename +
//                                                   ".training-accuracy";
//            std::string plotTestingAccuracyName = dirName + "/"
//                                                  + perceptronFilename
//                                                  + ".testing-accuracy";
//            system(("rmdir \"" + dirName + "\" /s /q").data());
//            system(("mkdir \"" + dirName + "\"").data());
//            saveErrorToFile(plotErrorName, trainingResults);
//            saveTrainingAccuracyToFile(plotTrainingAccuracyName,
//                                       trainingResults);
//            saveTestingAccuracyToFile(plotTestingAccuracyName, trainingResults);
//            {
//                std::ofstream file(dirName + "/" + perceptronFilename
//                                   + ".parameters", std::ios::trunc);
//                file << "Number of epochs: " << numberOfEpochs
//                     << "\nCost goal: " << costGoal
//                     << "\nLearning coefficient (start): "
//                     << learningCoefficientStart
//                     << "\nLearning coefficient (end): "
//                     << learningCoefficientEnd
//                     << "\nMomentum coefficient: " << momentumCoefficient
//                     << "\nShuffle training data: " << bool(shuffleTrainingData)
//                     << "\nEpoch interval: " << epochInterval;
//            }
//            system(("plot-cost-function.py " + plotErrorName).data());
//            system(("plot-training-accuracy.py " +
//                    plotTrainingAccuracyName).data());
//            system(("plot-testing-accuracy.py " +
//                    plotTestingAccuracyName).data());
//        }
//        else
//        {
//            NeuralNetwork multiLayerPerceptron(perceptronFilename);
//
//            std::cout << "\n" << std::string(79, '-') << std::endl;
//            printTestingResults(multiLayerPerceptron,
//                                trainingExamples, trainingClassLabels,
//                                dataSetTrainingFilenames[dataSet],
//                                perceptronFilename, false);
//
//            std::cout << "\n" << std::string(79, '-') << std::endl;
//            printTestingResults(multiLayerPerceptron,
//                                testingExamples, testingClassLabels,
//                                dataSetTestingFilenames[dataSet],
//                                perceptronFilename, true);
//        }
//    }
//    else if (classifier == "K-nearest neighbours")
//    {
//        // Load training and testing examples from file
//        auto const[trainingExamples, trainingClassLabels]
//        = readTrainingExamplesFromCsvFile
//                (dataSetTrainingFilenames[dataSet]);
//
//        auto const[testingExamples, testingClassLabels]
//        = readTrainingExamplesFromCsvFile
//                (dataSetTestingFilenames[dataSet]);
//
//        std::string neighboursFilename;
//        std::cout << std::string(60, ' ') << std::endl;
//        std::cout << std::string(79, '-') << std::endl;
//        std::cout << setw(IOMANIP_WIDTH) << "K-nearest neighbours filename"
//                  << " " << '|' << " ";
//        //std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//        std::getline(std::cin, neighboursFilename);
//
//        int k;
//        std::cout << setw(IOMANIP_WIDTH) << "K" << " " << '|' << " ";
//        std::cin >> k;
//        NeuralNetworks::KNearestNeighbours kNearestNeigbours
//                { k,
//                  trainingExamples };
//
////        printTestingResults(kNearestNeigbours,
////                            trainingExamples, trainingClassLabels,
////                            dataSetTrainingFilenames[dataSet],
////                            neighboursFilename);
//        std::cout << "\n" << std::string(79, '-') << std::endl;
//        printTestingResults(kNearestNeigbours,
//                            testingExamples, testingClassLabels,
//                            dataSetTestingFilenames[dataSet],
//                            neighboursFilename, false);
//
//    }
//
//    std::cout << "\n\n";
//    std::cout << std::string(79, '/') << std::endl;
//    return 0;
}
