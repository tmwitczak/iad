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
#include <random>

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
        if (i == userInput - 1)
            return options.cbegin() + i;

    std::cout << "Wrong input!\n";
    exit(1);
}

void saveErrorToFile
        (std::string const &filename,
         std::vector<double> const &costPerEpochInterval,
         int const epochInterval)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < costPerEpochInterval.size(); i++)
        file << i * epochInterval << ","
             << costPerEpochInterval.at(i) << "\n";
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
         std::string const &neuralNetworkFilename,
         bool const additionalTestingDump)
{
    NeuralNetwork::TestingResults testingResults
            = multiLayerPerceptron.test(testingExamples);

    std::cout << "\r" << std::string(80, ' ') << "\r";

    printAccuracy(std::cout, testingExamples, testingClassLabels,
                  testFilename, testingResults);

    std::string dirName = "testing-results_" + neuralNetworkFilename;
    {
        //system(("rmdir \"" + dirName + "\" /s /q").data());
        system(("if not exist \"" + dirName + "\" mkdir \"" + dirName + "\"")
                       .data());
        std::ofstream file(dirName + "/" + neuralNetworkFilename
                           + ".analysis", std::ios::out | std::ios::trunc);
        printAccuracy(file, testingExamples, testingClassLabels,
                      testFilename, testingResults);
    }

    // Additional info to file
    if (additionalTestingDump)
    {
        std::ofstream file(dirName + "/" + neuralNetworkFilename
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

void createPlotForThirdFunction(NeuralNetwork const &neuralNetwork, std::string
const &filenameNet, std::string const &filenameActual)
{
    {
        std::vector<double> x;
        std::vector<double> yNet;
        std::vector<double> yActual;
        int const resolution = 1024;
        std::ofstream fileNet(filenameNet + "1", std::ios::trunc);
        std::ofstream fileActual(filenameActual + "1", std::ios::trunc);

        for (int i = 0; i < resolution; ++i)
        {
            double const interval = (3.0 - (-3.0)) /
                                    (double) resolution;
            Vector xVec { 2 };
            xVec(0) = -3.0 + interval * i;
            xVec(1) = 0.0;

            x.push_back(xVec(0));
            yNet.push_back(neuralNetwork(xVec)(0));
            yActual.push_back(std::sin(xVec(0) * xVec(1)
                                       + std::cos(3.0 * (xVec(0) - xVec(1)))));

            fileNet << x.back() << ","
                    << yNet.back() << "\n";

            fileActual << x.back() << ","
                       << yActual.back() << "\n";
        }
    }
    {
        std::vector<double> x;
        std::vector<double> yNet;
        std::vector<double> yActual;
        int const resolution = 1024;
        std::ofstream fileNet(filenameNet + "2", std::ios::trunc);
        std::ofstream fileActual(filenameActual + "2", std::ios::trunc);

        for (int i = 0; i < resolution; ++i)
        {
            double const interval = (3.0 - (-3.0)) /
                                    (double) resolution;
            Vector xVec { 2 };
            xVec(0) = 0.0;
            xVec(1) = -3.0 + interval * i;

            x.push_back(xVec(1));
            yNet.push_back(neuralNetwork(xVec)(0));
            yActual.push_back(std::sin(xVec(0) * xVec(1)
                                       + std::cos(3.0 * (xVec(0) - xVec(1)))));

            fileNet << x.back() << ","
                    << yNet.back() << "\n";

            fileActual << x.back() << ","
                       << yActual.back() << "\n";
        }
    }
}
void createPlotForSin(NeuralNetwork const &neuralNetwork, std::string const
&filenameNet, std::string const &filenameActual)
{
    std::vector<double> x;
    std::vector<double> yNet;
    std::vector<double> yActual;
    int const resolution = 1024;
    std::ofstream fileNet(filenameNet, std::ios::trunc);
    std::ofstream fileActual(filenameActual, std::ios::trunc);

    for (int i = 0; i < resolution; ++i)
    {
        double const interval = (10.0 - (-10.0)) /
                                (double) resolution;
        Vector xVec { 1 };
        xVec(0) = -10.0 + interval * i;

        x.push_back(xVec(0));
        yNet.push_back(neuralNetwork(xVec)(0));
        yActual.push_back(std::sin(xVec(0)));

        fileNet << x.back() << ","
                << yNet.back() << "\n";

        fileActual << x.back() << ","
                   << yActual.back() << "\n";
    }
}
void createPlotForSqrt(NeuralNetwork const &neuralNetwork, std::string const
&filenameNet, std::string const &filenameActual)
{
    std::vector<double> x;
    std::vector<double> yNet;
    std::vector<double> yActual;
    int const resolution = 1024;
    std::ofstream fileNet(filenameNet, std::ios::trunc);
    std::ofstream fileActual(filenameActual, std::ios::trunc);

    for (int i = 0; i < resolution; ++i)
    {
        double const interval = (10.0 - 0.0) /
                                (double) resolution;
        Vector xVec { 1 };
        xVec(0) = 0.0 + interval * i;

        x.push_back(xVec(0));
        yNet.push_back(neuralNetwork(xVec)(0));
        yActual.push_back(std::sqrt(xVec(0)));

        fileNet << x.back() << ","
             << yNet.back() << "\n";

        fileActual << x.back() << ","
                   << yActual.back() << "\n";
    }
}
void testFun()
{
    NeuralNetwork::initialiseRandomNumberGenerator(time(nullptr));

    std::vector<std::unique_ptr<NeuralNetworkLayer>> layers;
    layers.emplace_back(RadialBasisFunctionLayer { 2, 10 });
    layers.emplace_back(AffineLayerWithBias { 10, 10,
                                              ParametricRectifiedLinearUnit
                                                      {0.01}});
    layers.emplace_back(AffineLayerWithBias { 10, 1 });
    NeuralNetwork neuralNetwork(std::move(layers));

    std::vector<TrainingExample> trainingExamples;
    for (int i = 0; i < 1000; i++)
    {
        Vector input { 2 };
        input << (double(i) / 1000.0), (double(i) / 1000.0);

        Vector output { 1 };
        output << std::sqrt(input(0) + input(1));

        TrainingExample tr = { input, output };
        trainingExamples.push_back(tr);
    }

    std::cout << trainingExamples[500].inputs << "\n\n"
              << trainingExamples[500].outputs << "\n\n";

    NeuralNetwork::TrainingResults trainingResults =
            neuralNetwork.train
                    (trainingExamples,
                     trainingExamples,
                     trainingExamples,
                     10000,
                     0.0, 0.0001, 0.0, 0.8, true, 10);

    for (int i = 0; i < trainingResults.costPerEpochIntervalTraining.size(); i++)
        std::cout << i * 10 << " | " << trainingResults
                .costPerEpochIntervalTraining[i] << "\n";
    //std::cout << "\n\n" << trainingResults.costPerEpochInterval.back() <<
    // "\n";

    Vector test { 2 };
    test << 0.5, 0.5;
    std::cout << "\n\n" << neuralNetwork(test) << "\n";
}
////////////////////////////////////////////////////////////// | Project: iad-2a
int main
        ()
{
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
    //------------------------------
    // Choose function to approximate
    std::cout << std::string(79, '/') << std::endl;
    std::vector<std::string> functions
            { "sqrt(x)",
              "sin(x)",
              "sin(x1 * x2) + cos(3*(x1 - x2))" };
    auto const chosenFunction
            = askUserForInput("Choose function", functions);

    // Choose network architecture
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

    // Choose mode
    //std::cout << std::endl;
    //std::cout << std::string(79, '-') << std::endl;
    std::vector<std::string> modes
            { "Training",
              "Testing" };
    auto const mode = modes.cbegin();
            //= askUserForInput("Choose mode", modes);

    // Enter number of points for training and testing
    int numberOfTrainingPoints;
    int numberOfTestingPoints;
    std::cout << "\n";
    std::cout << std::string(79, '-') << std::endl;
    std::cout << setw(IOMANIP_WIDTH) << "Number of training examples"
              << '|' << " ";
    std::cin >> numberOfTrainingPoints;
    std::cout << setw(IOMANIP_WIDTH) << "Number of testing examples"
              << '|' << " ";
    std::cin >> numberOfTestingPoints;

    // Enter network filename
    std::string neuralNetworkFilename;
    std::cout << "\n";
    std::cout << std::string(79, '-') << std::endl;
    std::cout << setw(IOMANIP_WIDTH) << "Neural network filename"
              << '|' << " ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(cin, neuralNetworkFilename);

    // Create training and testing examples for chosen function
    std::vector<TrainingExample> trainingExamples;
    std::vector<TrainingExample> testingExamples;

    auto randomNumberGenerator
            = std::mt19937 { std::random_device {}() };

    if (*chosenFunction == "sqrt(x)")
    {
        for (int i = 0; i < numberOfTrainingPoints; ++i)
        {
            double const interval = (10.0 - 0.0) /
                    (double) numberOfTrainingPoints;
            double const low = 0.0 + interval * i;
            double const high = 0.0 + interval * (i + 1);

            std::uniform_real_distribution<double>
                    uniformRealDistribution(low, high);


            TrainingExample trainingExample { Vector { 1 }, Vector { 1 }};
            trainingExample.inputs(0)
                    = uniformRealDistribution(randomNumberGenerator);
            trainingExample.outputs(0) = std::sqrt(
                    trainingExample.inputs(0));
            trainingExamples.push_back(trainingExample);
        }
        for (int i = 0; i < numberOfTestingPoints; ++i)
        {
            double const interval = (10.0 - 0.0) / (double)
                    numberOfTestingPoints;
            double const low = 0.0 + interval * i;
            double const high = 0.0 + interval * (i + 1);

            std::uniform_real_distribution<double>
                    uniformRealDistribution(low, high);

            TrainingExample testingExample { Vector { 1 }, Vector { 1 }};
            testingExample.inputs(0)
                    = uniformRealDistribution(randomNumberGenerator);
            testingExample.outputs(0) = std::sqrt(
                    testingExample.inputs(0));
            testingExamples.push_back(testingExample);
        }
    }
    else if (*chosenFunction == "sin(x)")
    {
        for (int i = 0; i < numberOfTrainingPoints; ++i)
        {
            double const interval = (10.0 - (-10.0)) /
                    (double) numberOfTrainingPoints;
            double const low = -10.0 + interval * i;
            double const high = -10.0 + interval * (i + 1);

            std::uniform_real_distribution<double>
                    uniformRealDistribution(low, high);

            TrainingExample trainingExample { Vector { 1 }, Vector { 1 }};
            trainingExample.inputs(0)
                    = uniformRealDistribution(randomNumberGenerator);
            trainingExample.outputs(0) = std::sin(
                    trainingExample.inputs(0));
            trainingExamples.push_back(trainingExample);
        }
        for (int i = 0; i < numberOfTestingPoints; ++i)
        {
            double const interval = (10.0 - (-10.0)) /
                    (double) numberOfTestingPoints;
            double const low = -10.0 + interval * i;
            double const high = -10.0 + interval * (i + 1);

            std::uniform_real_distribution<double>
                    uniformRealDistribution(low, high);

            TrainingExample testingExample { Vector { 1 }, Vector { 1 }};
            testingExample.inputs(0)
                    = uniformRealDistribution(randomNumberGenerator);
            testingExample.outputs(0) = std::sin(
                    testingExample.inputs(0));
            testingExamples.push_back(testingExample);
        }
    }
    else if (*chosenFunction == "sin(x1 * x2) + cos(3*(x1 - x2))")
    {
        for (int i = 0; i < std::sqrt(numberOfTrainingPoints); ++i)
        {
            double const interval = (3.0 - (-3.0)) /
                    (double) std::sqrt(numberOfTrainingPoints);
            double const low1 = -3.0 + interval * i;
            double const high1 = -3.0 + interval * (i + 1);

            std::uniform_real_distribution<double>
                    uniformRealDistribution1(low1, high1);

            for (int j = 0; j < std::sqrt(numberOfTrainingPoints); ++j)
            {
                double const low2 = -3.0 + interval * j;
                double const high2 = -3.0 + interval * (j + 1);

                std::uniform_real_distribution<double>
                        uniformRealDistribution2(low2, high2);

                TrainingExample trainingExample { Vector { 2 }, Vector { 1 }};
                trainingExample.inputs(0)
                        = uniformRealDistribution1(randomNumberGenerator);
                trainingExample.inputs(1)
                        = uniformRealDistribution2(randomNumberGenerator);
                trainingExample.outputs(0)
                    = std::sin(trainingExample.inputs(0) *
                            trainingExample.inputs(1)) + std::cos(
                                    3.0 * (trainingExample.inputs(0) -
                                    trainingExample.inputs(1))
                                    );
                trainingExamples.push_back(trainingExample);
            }
        }
        for (int i = 0; i < std::sqrt(numberOfTestingPoints); ++i)
        {
            double const interval = (3.0 - (-3.0)) /
                    (double) numberOfTestingPoints;
            double const low1 = -3.0 + interval * i;
            double const high1 = -3.0 + interval * (i + 1);

            std::uniform_real_distribution<double>
                    uniformRealDistribution1(low1, high1);

            for (int j = 0; j < std::sqrt(numberOfTestingPoints); ++j)
            {
                double const low2 = -3.0 + interval * j;
                double const high2 = -3.0 + interval * (j + 1);

                std::uniform_real_distribution<double>
                        uniformRealDistribution2(low2, high2);

                TrainingExample testingExample { Vector { 2 }, Vector { 1 }};
                testingExample.inputs(0)
                        = uniformRealDistribution1(randomNumberGenerator);
                testingExample.inputs(1)
                        = uniformRealDistribution2(randomNumberGenerator);
                testingExample.outputs(0)
                        = std::sin(testingExample.inputs(0) *
                                   testingExample.inputs(1)) + std::cos(
                        3.0 * (testingExample.inputs(0) -
                               testingExample.inputs(1))
                );
                testingExamples.push_back(testingExample);
            }
        }
    }

    // Prepare MLP
    NeuralNetwork::initialiseRandomNumberGenerator(static_cast<int>(time(nullptr)));

    // Do the math :)
    std::vector<std::unique_ptr<NeuralNetworkLayer>> layers;

    if (architecture == architectures.cbegin())
    {
        // Neuron number
        std::string hiddenLayerNeuronNumber;
        std::cout << setw(IOMANIP_WIDTH) << "Neurons in hidden layers"
                  << '|' << " ";
        std::getline(std::cin, hiddenLayerNeuronNumber);


        layers.emplace_back(AffineLayerWithBias { (int)trainingExamples.back()
        .inputs.size(),
                                                  std::stoi
        (hiddenLayerNeuronNumber),
                                                  Sigmoid {} });
        layers.emplace_back(AffineLayerWithBias { std::stoi(hiddenLayerNeuronNumber), 1 });
    }
    if (architecture == architectures.cbegin() + 1)
    {
        // Neuron number
        std::string hiddenLayerNeuronNumber;
        std::cout << setw(IOMANIP_WIDTH) << "Neurons in hidden layers"
                  << '|' << " ";
        std::getline(std::cin, hiddenLayerNeuronNumber);


        layers.emplace_back(RadialBasisFunctionLayer{ int(trainingExamples
        .back().inputs.size()), std::stoi
        (hiddenLayerNeuronNumber)});
        layers.emplace_back(AffineLayerWithBias { std::stoi(hiddenLayerNeuronNumber), 1 });
    }
    if (architecture == architectures.cbegin() + 2)
    {
        // Neuron number
        std::string hiddenLayerNeuronNumber;
        std::cout << setw(IOMANIP_WIDTH) << "Neurons in hidden layers"
                  << '|' << " ";
        std::getline(std::cin, hiddenLayerNeuronNumber);

        std::vector<int> layersNeurons;
        for (auto const &neurons : split(hiddenLayerNeuronNumber, " "))
            layersNeurons.push_back(std::stoi(neurons.data()));

        layers.emplace_back(RadialBasisFunctionLayer{ (int)trainingExamples
        .back().inputs.size(),
                                                      layersNeurons.at
        (0)});
        layers.emplace_back(AffineLayerWithBias { layersNeurons.at(0),
                                                  layersNeurons.at(1),
                                                          Sigmoid {}});
        layers.emplace_back(AffineLayerWithBias { layersNeurons.at(1), 1});
    }

    NeuralNetwork neuralNetwork(std::move(layers));

    // Get parameters
    int numberOfEpochs;
    double costGoal;
    double learningCoefficientStart;
    double learningCoefficientEnd;
    double momentumCoefficient;
    bool shuffleTrainingData;
    int epochInterval;

    std::cout << setw(IOMANIP_WIDTH) << "Number of epochs" << " " << '|' << " ";
    std::cin >> numberOfEpochs;
    std::cout << setw(IOMANIP_WIDTH) << "Cost goal" << " " << '|' << " ";
    std::cin >> costGoal;
    std::cout << setw(IOMANIP_WIDTH) << "Learning coefficient (start)"
              << " " << '|' << " ";
    std::cin >> learningCoefficientStart;
    std::cout << setw(IOMANIP_WIDTH) << "Learning coefficient (end)"
              << " " << '|' << " ";
    std::cin >> learningCoefficientEnd;
    std::cout << setw(IOMANIP_WIDTH) << "Momentum coefficient" << " " << '|' << " ";
    std::cin >> momentumCoefficient;
    std::cout << setw(IOMANIP_WIDTH) << "Shuffle training data"
              << " " << '|' << " ";
    std::cin >> shuffleTrainingData;
    std::cout << setw(IOMANIP_WIDTH) << "Epoch interval" << " " << '|' << " ";
    std::cin >> epochInterval;

    // Train
    NeuralNetwork::TrainingResults trainingResults
            = neuralNetwork.train(trainingExamples,
                                         testingExamples,
                                         testingExamples, // TODO:
                                         // EXTRAPOLATION!
                                         numberOfEpochs,
                                         costGoal,
                                         learningCoefficientStart,
                                         learningCoefficientEnd
                                         - learningCoefficientStart,
                                         momentumCoefficient,
                                         shuffleTrainingData,
                                         epochInterval);

    // Document learning
    //neuralNetwork.saveToFile(neuralNetworkFilename);

    std::string dirName = "training-results_" + neuralNetworkFilename;
    std::string plotFunction = dirName + "/"
                                       + neuralNetworkFilename + ".plot";
    std::string plotCostNameTraining = dirName + "/"
                                + neuralNetworkFilename + ".training-cost";
    std::string plotCostNameTesting = dirName + "/"
                                      + neuralNetworkFilename + ".testing-cost";
    std::string plotCostNameTestingExtrapolation
    = dirName + "/" + neuralNetworkFilename + ".testing-extrapolation-cost";

    system(("rmdir \"" + dirName + "\" /s /q").data());
    system(("mkdir \"" + dirName + "\"").data());

    if (chosenFunction == functions.cbegin())
    {
        createPlotForSqrt(neuralNetwork, plotFunction + ".net",
                plotFunction + ".actual");
        system(("python plot-sqrt.py " + plotFunction).data());
    }
    if (chosenFunction == functions.cbegin()+1)
    {
        createPlotForSin(neuralNetwork, plotFunction + ".net",
                          plotFunction + ".actual");
        system(("python plot-sin.py " + plotFunction).data());
    }
    if (chosenFunction == functions.cbegin()+2)
    {
        createPlotForThirdFunction(neuralNetwork, plotFunction + ".net",
                         plotFunction + ".actual");
        system(("python plot-thirdfunction.py " + plotFunction).data());
    }

    saveErrorToFile(plotCostNameTraining, trainingResults
    .costPerEpochIntervalTraining, trainingResults.epochInterval);
    saveErrorToFile(plotCostNameTesting, trainingResults
    .costPerEpochIntervalTesting, trainingResults.epochInterval);
    saveErrorToFile(plotCostNameTestingExtrapolation, trainingResults
    .costPerEpochIntervalTestingExtrapolation, trainingResults.epochInterval);
    {
        std::ofstream file(dirName + "/" + neuralNetworkFilename
                           + ".parameters", std::ios::trunc);
        file << "Number of epochs: " << numberOfEpochs
             << "\nCost goal: " << costGoal
             << "\nLearning coefficient (start): "
             << learningCoefficientStart
             << "\nLearning coefficient (end): "
             << learningCoefficientEnd
             << "\nMomentum coefficient: " << momentumCoefficient
             << "\nShuffle training data: " << bool(shuffleTrainingData)
             << "\nEpoch interval: " << epochInterval;
    }


    system(("python plot-cost-function.py " + plotCostNameTraining).data());
    system(("python plot-cost-function.py " + plotCostNameTesting).data());
    system(("python plot-cost-function.py " + plotCostNameTestingExtrapolation).data());
//        }
//        else
//        {
//            NeuralNetwork multiLayerPerceptron(neuralNetworkFilename);
//
//            std::cout << "\n" << std::string(79, '-') << std::endl;
//            printTestingResults(multiLayerPerceptron,
//                                trainingExamples, trainingClassLabels,
//                                dataSetTrainingFilenames[dataSet],
//                                neuralNetworkFilename, false);
//
//            std::cout << "\n" << std::string(79, '-') << std::endl;
//            printTestingResults(multiLayerPerceptron,
//                                testingExamples, testingClassLabels,
//                                dataSetTestingFilenames[dataSet],
//                                neuralNetworkFilename, true);
//        }
//    }
//
    std::cout << "\n\n";
    std::cout << std::string(79, '/') << std::endl;

    return 0;
}
