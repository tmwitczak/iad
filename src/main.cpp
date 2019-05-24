///////////////////////////////////////////////////////////////////// | Includes
#include "multi-layer-perceptron.hpp"
#include "training-example.hpp"
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

using namespace std;
using namespace NeuralNetworks;
using Vector = Eigen::VectorXd;

constexpr int IOMANIP_WIDTH = 40;


std::vector<std::string_view> split
        (std::string_view stringView,
         std::string_view delimiters)
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

std::pair<std::vector<TrainingExample>, std::vector<std::string>>
readTrainingExamplesFromCsvFile
        (std::string const &filename)
{
    std::vector<std::string> classLabels;
    std::vector<TrainingExample> trainingExamples;

    std::ifstream file(filename, std::ios::in);
    {
        std::string line;
        bool firstLine = true;
        int numberOfOutputs = 0;
        int numberOfInputs = 0;

        while (std::getline(file, line)
               && !line.empty())
        {
            std::vector<std::string_view> tokens
                    = split(line, ",");

            if (firstLine)
            {
                int numberOfClasses = 0;
                for (auto const &i : tokens)
                    numberOfClasses += (int) (i != " ");

                classLabels.assign(tokens.end() - numberOfClasses,
                                   tokens.end());

                firstLine = false;
                numberOfInputs = tokens.size() - numberOfClasses;
                numberOfOutputs = numberOfClasses;
            }
            else
            {
                // Load training example
                TrainingExample trainingExample { Vector { numberOfInputs },
                                                  Vector { numberOfOutputs }};

                // Read inputs
                for (int i = 0, j = 0;
                     i < numberOfInputs;
                     i++, j++)
                {
                    trainingExample.inputs(i) = atof(tokens.at(j).data());
                }

                // Read outputs
                for (int i = 0, j = numberOfInputs;
                     i < numberOfOutputs;
                     i++, j++)
                {
                    trainingExample.outputs(i) = atof(tokens.at(j).data());
                }

                // Save the training example
                trainingExamples.push_back(trainingExample);
            }
        }
    }

    return std::make_pair(trainingExamples, classLabels);
}


std::string askUserForInput
        (std::string_view const &question,
         std::vector<std::pair<int, std::string>> options)
{
    for (auto const &option
            : options)
        std::cout << setw(IOMANIP_WIDTH)
                  << option.first << " | " << option.second << std::endl;

    std::cout << setw(IOMANIP_WIDTH) << question << " | ";
    int userInput;
    std::cin >> userInput;

    for (int i = 0; i < options.size(); i++)
        if (options[i].first == userInput)
            return options[i].second;

    exit(1);
}

void saveErrorToFile
        (std::string const &filename,
         MultiLayerPerceptron::TrainingResults const &trainingResults)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        file << i * trainingResults.epochInterval << ","
             << trainingResults.costPerEpochInterval.at(i) << std::endl;
}

void saveTrainingAccuracyToFile
        (std::string const &filename,
         MultiLayerPerceptron::TrainingResults const &trainingResults)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        file << i * trainingResults.epochInterval << ","
             << trainingResults.accuracyTraining.at(i) << std::endl;
}

void saveTestingAccuracyToFile
        (std::string const &filename,
         MultiLayerPerceptron::TrainingResults const &trainingResults)
{
    std::ofstream file(filename, std::ios::trunc);
    for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
        file << i * trainingResults.epochInterval << ","
             << trainingResults.accuracyTesting.at(i) << std::endl;
}

std::string braces
        (std::string const &str)
{
    return "[" + str + "]";
}

void printTestingResults(MultiLayerPerceptron const &multiLayerPerceptron,
                         std::vector<TrainingExample> const &testingExamples,
                         std::vector<std::string> const &
                         testingClassLabels,
                         std::string const &testFilename,
                         std::string const &perceptronFilename)
{
    MultiLayerPerceptron::TestingResults testingResults
            = multiLayerPerceptron.test(testingExamples);

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

    std::cout << "\n\n>> " << testFilename << std::endl;

    std::cout << "\n > Global confusion matrix: " << std::endl;
    for (int i = 0; i < confusionMatrix.rows(); i++)
    {
        if (i == 0)
        {
            std::cout << setw(20) << " ";
            for (auto const &testingClassLabel
                    : testingClassLabels)
                std::cout
                        << setw(braces(testingClassLabel)
                                        .length() + 5)
                        << braces(testingClassLabel) << " ";
            std::cout << std::endl;
        }
        std::cout << setw(20)
                  << braces(testingClassLabels.at(i));

        for (int j = 0; j < confusionMatrix.cols(); j++)
        {
            std::cout
                    << setw(braces(testingClassLabels.at(j))
                                    .length() + 5) << confusionMatrix(i, j)
                    << " ";
        }

        std::cout << std::endl;
    }

    std::cout << "\n\n" << setw(IOMANIP_WIDTH) << "Total population | "
              << testingExamples.size()
              << "\n"
              << "\n" << setw(IOMANIP_WIDTH) << "Accuracy | "
              << globalAccuracy * 100 << " %" << std::endl;


// Confusion matrices per class
    for (int k = 0; k < confusionPerClass.size(); k++)
    {
        std::cout << "\n\n\n";
        std::cout << "> " << braces(testingClassLabels.at(k)) << std::endl;

        for (int i = 0; i < confusionPerClass.at(k).rows(); i++)
        {
            if (i == 0)
            {
                std::cout << setw(20) << " "
                          << setw(8 + 5) << "[Positive]" << " "
                          << setw(8 + 5) << "[Negative]"
                          << std::endl;
            }
            std::cout << setw(20) << (i == 0 ? "[Positive]" : "[Negative]");

            for (int j = 0; j < confusionPerClass.at(k).cols(); j++)
                std::cout
                        << setw(8 + 5) << confusionPerClass.at(k)(i, j)
                        << " ";
            std::cout << std::endl;
        }

        std::cout << std::endl;

// Statistics
        int const totalPopulation = confusionPerClass.at(k).sum();

        int const &truePositive = confusionPerClass.at(k)(0, 0);
        int const &trueNegative = confusionPerClass.at(k)(1, 1);
        int const &falsePositive = confusionPerClass.at(k)(0, 1);
        int const &falseNegative = confusionPerClass.at(k)(1, 0);

        int const predictedPositive = truePositive + falsePositive;
        int const predictedNegative = falseNegative + trueNegative;
        int const actualPositive = truePositive + falseNegative;
        int const actualNegative = falsePositive + trueNegative;

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
                = double(falsePositive) / actualPositive;
        double const falseNegativeRate
                = double(falseNegative) / actualNegative;
        double const trueNegativeRate
                = double(trueNegative) / actualNegative;

        double const accuracy
                = double(truePositive + trueNegative) / totalPopulation;

        std::cout << "\n" << setw(IOMANIP_WIDTH) << "Total population | " <<
                  totalPopulation
                  << "\n"
                  << "\n" << setw(IOMANIP_WIDTH) << "True positive | "
                  << truePositive
                  << "\n" << setw(IOMANIP_WIDTH) << "True negative | "
                  << trueNegative
                  << "\n" << setw(IOMANIP_WIDTH)
                  << "False positive (type I error) | " << falsePositive
                  << "\n" << setw(IOMANIP_WIDTH)
                  << "False negative (type II error) | " << falseNegative
                  << "\n"
                  << "\n" << setw(IOMANIP_WIDTH) << "Predicted positive | "
                  << predictedPositive
                  << "\n" << setw(IOMANIP_WIDTH) << "Predicted negative | "
                  << predictedNegative
                  << "\n" << setw(IOMANIP_WIDTH) << "Actual positive | "
                  << actualPositive
                  << "\n" << setw(IOMANIP_WIDTH) << "Actual negative | "
                  << actualNegative
                  << "\n"
                  << "\n" << setw(IOMANIP_WIDTH)
                  << "Positive predictive value | "
                  << positivePredictiveValue * 100 << " %"
                  << "\n" << setw(IOMANIP_WIDTH)
                  << "False discovery rate | "
                  << falseDiscoveryRate * 100 << " %"
                  << "\n" << setw(IOMANIP_WIDTH) << "False omission rate | "
                  << falseOmissionRate * 100 << " %"
                  << "\n" << setw(IOMANIP_WIDTH)
                  << "Negative prediction value | "
                  << negativePredictiveValue * 100 << " %"
                  << "\n"
                  << "\n" << setw(IOMANIP_WIDTH) << "True positive rate | "
                  << truePositiveRate * 100 << " %"
                  << "\n" << setw(IOMANIP_WIDTH) << "False positive rate | "
                  << falsePositiveRate * 100 << " %"
                  << "\n" << setw(IOMANIP_WIDTH) << "False negative rate | "
                  << falseNegativeRate * 100 << " %"
                  << "\n" << setw(IOMANIP_WIDTH) << "True negative rate | "
                  << trueNegativeRate * 100 << " %"
                  << "\n"
                  << "\n" << setw(IOMANIP_WIDTH) << "Accuracy | "
                  << accuracy * 100 << " %"
                  << std::endl;
    }

    // Additional info to file
    {
        std::string dirName = "testing-results_" + perceptronFilename;
        system(("rmdir \"" + dirName + "\" /s /q").data());
        system(("mkdir \"" + dirName + "\"").data());
        std::ofstream file(dirName + "/" + perceptronFilename
                           + ".information", std::ios::trunc);

        file << "> [Global cost]: " << testingResults.globalCost << std::endl;
        for (auto const &i : testingResults.testingResultsPerExample)
        {
            file << "\n\n\n>>>>>>>>>>>>>>>>>>>>>>" << std::endl;

            file << "> [Neurons]\n";
            for (auto const &j : i.neurons)
                file << "\n"
                     << j << std::endl;

            file << "\n> [Targets]\n";
            file << "\n"
                 << i.targets << std::endl;

            file << "\n> [Errors]\n";
            for (auto j = i.errors.crbegin(); j != i.errors.crend(); ++j)
                file << "\n"
                     << *j << std::endl;

            file << "\n> [Cost]\n";
            file << "\n"
                 << i.cost << std::endl;
        }
    }
}

////////////////////////////////////////////////////////////// | Project: iad-2a
int main()
{
    std::string dataSet = askUserForInput("Choose data set",
                                          {{ 1, "Identity" },
                                           { 2, "Iris" },
                                           { 3, "Iris (normalised)" },
                                           { 4, "Iris (standardised)" },
                                           { 5, "Seeds" },
                                           { 6, "Seeds (normalised)" },
                                           { 7, "Seeds (standardised)" },
                                           { 8, "Digits" },
                                           { 9, "Digits (normalised)" }});

    // TODO: Klasyfikator

    std::map<std::string, std::string> dataSetTrainingFilenames
            {{ "Identity",             "./data/identity-train.csv" },
             { "Iris",                 "./data/iris-train.csv" },
             { "Iris (normalised)",    "./data/iris-normalised-train.csv" },
             { "Iris (standardised)",  "./data/iris-standardised-train.csv" },
             { "Seeds",                "./data/seeds-train.csv" },
             { "Seeds (normalised)",   "./data/seeds-normalised-train.csv" },
             { "Seeds (standardised)", "./data/seeds-standardised-train.csv" },
             { "Digits",               "./data/digits-train.csv" },
             { "Digits (normalised)",  "./data/digits-normalised-train.csv" }};

    std::map<std::string, std::string> dataSetTestingFilenames
            {{ "Identity",             "./data/identity-test.csv" },
             { "Iris",                 "./data/iris-test.csv" },
             { "Iris (normalised)",    "./data/iris-normalised-test.csv" },
             { "Iris (standardised)",  "./data/iris-standardised-test.csv" },
             { "Seeds",                "./data/seeds-test.csv" },
             { "Seeds (normalised)",   "./data/seeds-normalised-test.csv" },
             { "Seeds (standardised)", "./data/seeds-standardised-test.csv" },
             { "Digits",               "./data/digits-test.csv" },
             { "Digits (normalised)",  "./data/digits-normalised-test.csv" }};


    std::cout << endl;
    std::string mode = askUserForInput("Choose mode",
                                       {{ 1, "Training" },
                                        { 2, "Testing" }});

    std::string perceptronFilename;
    std::cout << std::endl;
    std::cout << setw(IOMANIP_WIDTH) << "Multi-layer perceptron filename"
              << " | ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(cin, perceptronFilename);

    // Load training and testing examples from file
    auto const[trainingExamples, trainingClassLabels]
    = readTrainingExamplesFromCsvFile
            (dataSetTrainingFilenames[dataSet]);

    auto const[testingExamples, testingClassLabels]
    = readTrainingExamplesFromCsvFile
            (dataSetTestingFilenames[dataSet]);

    // Do the math :)
    if (mode == "Training")
    {
        // Prepare perceptron
        MultiLayerPerceptron::initialiseRandomNumberGenerator
                (static_cast<int>(time(nullptr)));

        // Neuron number
        std::string hiddenLayerNeuronNumber;
        std::cout << setw(IOMANIP_WIDTH) << "Neurons in hidden layers" << " | ";
        std::getline(std::cin, hiddenLayerNeuronNumber);

        std::vector<int> layersNeurons;
        layersNeurons.push_back
                (static_cast<int>(trainingExamples.at(0).inputs.size()));

        for (auto const &neurons : split(hiddenLayerNeuronNumber, " "))
            layersNeurons.push_back(atoi(neurons.data()));

        layersNeurons.push_back
                (static_cast<int>(trainingExamples.at(0).outputs.size()));

        // Biases
        std::string biases;
        std::cout << setw(IOMANIP_WIDTH) << "Enable biases per layer" << " | ";
        std::getline(std::cin, biases);

        std::vector<bool> biasesPerLayer;

        for (auto const &bias : split(biases, " "))
            biasesPerLayer.push_back((bool) atoi(bias.data()));

        if (biasesPerLayer.empty())
            biasesPerLayer = std::vector<bool>(layersNeurons.size() - 1, true);

        MultiLayerPerceptron multiLayerPerceptron
                { layersNeurons,
                  biasesPerLayer };

        // Get parameters
        int numberOfEpochs;
        double costGoal;
        double learningCoefficientStart;
        double learningCoefficientEnd;
        double momentumCoefficient;
        bool shuffleTrainingData;
        int epochInterval;

        std::cout << setw(IOMANIP_WIDTH) << "Number of epochs" << " | ";
        std::cin >> numberOfEpochs;
        std::cout << setw(IOMANIP_WIDTH) << "Cost goal" << " | ";
        std::cin >> costGoal;
        std::cout << setw(IOMANIP_WIDTH) << "Learning coefficient (start)"
                  << " | ";
        std::cin >> learningCoefficientStart;
        std::cout << setw(IOMANIP_WIDTH) << "Learning coefficient (end)"
                  << " | ";
        std::cin >> learningCoefficientEnd;
        std::cout << setw(IOMANIP_WIDTH) << "Momentum coefficient" << " | ";
        std::cin >> momentumCoefficient;
        std::cout << setw(IOMANIP_WIDTH) << "Shuffle training data" << " | ";
        std::cin >> shuffleTrainingData;
        std::cout << setw(IOMANIP_WIDTH) << "Epoch interval" << " | ";
        std::cin >> epochInterval;

        // Train
        MultiLayerPerceptron::TrainingResults trainingResults
                = multiLayerPerceptron.train(trainingExamples,
                                             testingExamples,
                                             numberOfEpochs,
                                             costGoal,
                                             learningCoefficientStart,
                                             learningCoefficientEnd
                                             - learningCoefficientStart,
                                             momentumCoefficient,
                                             shuffleTrainingData,
                                             epochInterval);

        // Document learning
        multiLayerPerceptron.saveToFile(perceptronFilename);

        std::string dirName = "training-results_" + perceptronFilename;
        std::string plotErrorName = dirName + "/"
                                    + perceptronFilename + ".cost";
        std::string plotTrainingAccuracyName = dirName + "/"
                                               + perceptronFilename +
                                               ".training-accuracy";
        std::string plotTestingAccuracyName = dirName + "/"
                                              + perceptronFilename
                                              + ".testing-accuracy";
        system(("rmdir \"" + dirName + "\" /s /q").data());
        system(("mkdir \"" + dirName + "\"").data());
        saveErrorToFile(plotErrorName, trainingResults);
        saveTrainingAccuracyToFile(plotTrainingAccuracyName, trainingResults);
        saveTestingAccuracyToFile(plotTestingAccuracyName, trainingResults);
        {
            std::ofstream file(dirName + "/" + perceptronFilename
                               + ".parameters", std::ios::trunc);
            file << "Number of epochs: " << numberOfEpochs
                 << "\nCost goal: " << costGoal
                 << "\nLearning coefficient (start): "
                 << learningCoefficientStart
                 << "\nLearning coefficient (end): " << learningCoefficientEnd
                 << "\nMomentum coefficient: " << momentumCoefficient
                 << "\nShuffle training data: " << bool(shuffleTrainingData)
                 << "\nEpoch interval: " << epochInterval;
        }
        system(("plot-cost-function.py " + plotErrorName).data());
        system(("plot-training-accuracy.py " +
                plotTrainingAccuracyName).data());
        system(("plot-testing-accuracy.py " + plotTestingAccuracyName).data());
    }
    else
    {
        MultiLayerPerceptron multiLayerPerceptron(perceptronFilename);

        printTestingResults(multiLayerPerceptron,
                            trainingExamples, trainingClassLabels,
                            dataSetTrainingFilenames[dataSet],
                            perceptronFilename);
        printTestingResults(multiLayerPerceptron,
                            testingExamples, testingClassLabels,
                            dataSetTestingFilenames[dataSet],
                            perceptronFilename);
    }

    std::cout << "\n\n";
    system("pause");
    return 0;
}
