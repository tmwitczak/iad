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

struct TrainingExampleClass
{
    std::string className;
    std::vector<TrainingExample> trainingExamples;
};


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
        (std::string const &filename)
{
    std::vector<TrainingExampleClass> trainingExampleClasses;

    std::ifstream file(filename, std::ios::in);
    {
        std::string line;
        bool firstLine = true;
        int numberOfOutputs = 0;
        int numberOfInputs = 0;
        std::vector<std::string> classes;

        while (std::getline(file, line) && !line.empty())
        {
            std::vector<std::string_view> tokens = split(line, ",");

            if (firstLine)
            {
                int numberOfClasses = 0;
                for (auto const &i : tokens)
                    numberOfClasses += (int) (i != " ");
                classes.assign(tokens.end() - numberOfClasses, tokens.end());

                firstLine = false;
                numberOfInputs = tokens.size() - numberOfClasses;
                numberOfOutputs = numberOfClasses;
            }
            else
            {
                TrainingExample trainingExample;
                trainingExample.inputs = Vector { numberOfInputs };
                trainingExample.outputs = Vector { numberOfOutputs };

                for (int i = 0; i < numberOfInputs; i++)
                {
                    // TODO: Rewrite that monster!
                    double inputNumber;
                    std::stringstream str;
                    str << tokens.at(i);
                    str >> inputNumber;
                    trainingExample.inputs(i) = inputNumber;
                }

                for (int i = 0; i < numberOfOutputs; i++)
                {
                    // TODO: Rewrite that monster!
                    double inputNumber;
                    std::stringstream str;
                    str << tokens.at(i + numberOfInputs);
                    str >> inputNumber;
                    trainingExample.outputs(i) = inputNumber;
                }

                bool isClassAlreadyIn = false;
                int classNumber = -1;
                trainingExample.outputs.array().maxCoeff(&classNumber);
                for (auto const &i : trainingExampleClasses)
                {
                    if (i.className == classes.at(classNumber))
                    {
                        isClassAlreadyIn = true;
                        break;
                    }
                }

                if (isClassAlreadyIn)
                {
                    for (auto &i : trainingExampleClasses)
                    {
//                        trainingExample.outputs.array().maxCoeff(&classNumber);
                        if (i.className == classes.at(classNumber))
                            i.trainingExamples.push_back(trainingExample);
                    }
//                    for (int i = 0; i < trainingExampleClasses.size(); i++)
//                        if (trainingExampleClasses[i].className ==
//                            tokens.back())
//                        {
//                            trainingExample.outputs(i) = 1.0;
//                            trainingExampleClasses[i].trainingExamples.push_back
//                                    (trainingExample);
//                        }
                }
                else
                {
                    TrainingExampleClass newClass;
//                    trainingExample.outputs(
//                            trainingExampleClasses.size()) = 1.0;
                    newClass.className = classes.at(classNumber);
                    newClass.trainingExamples.push_back(trainingExample);
                    trainingExampleClasses.push_back(newClass);
                }
            }
        }
    }
    return trainingExampleClasses;
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
                         std::vector<TrainingExampleClass> const &
                         testingExampleClasses,
                         std::string const &testFilename,
                         std::string const &perceptronFilename)
{
    MultiLayerPerceptron::TestingResults testingResults
            = multiLayerPerceptron.test(testingExamples);

    // Accuracy
    int globalNumberOfAccurateClassifications = 0;
    std::vector<int> accurateClassificationsPerClass
            (testingExampleClasses.size(), 0);
    std::vector<int> classCount
            (testingExampleClasses.size(), 0);

    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero
            (testingExampleClasses.size(),
             testingExampleClasses.size());
    std::vector<Eigen::MatrixXd> confusionPerClass
            (testingExampleClasses.size(), Eigen::MatrixXd::Zero(2, 2));

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
            for (auto const &testingExampleClass
                    : testingExampleClasses)
                std::cout
                        << setw(braces(testingExampleClass.className)
                                        .length() + 5)
                        << braces(testingExampleClass.className) << " ";
            std::cout << std::endl;
        }
        std::cout << setw(20)
                  << braces(testingExampleClasses.at(i).className);

        for (int j = 0; j < confusionMatrix.cols(); j++)
        {
            std::cout
                    << setw(braces(testingExampleClasses.at(j).className)
                                    .length() + 5) << confusionMatrix(i, j)
                    << " ";
        }

//            std::cout << " | "
//                      << confusionMatrix(i, i) / confusionMatrix.row(i).sum() *
//                         100;
        std::cout << std::endl;
    }
//        std::cout << setw(20) << " ";
//        for (int i = 0; i < confusionMatrix.cols(); i++)
//            std::cout
//                    << setw(testingExampleClasses.at(i).className.length() + 5)
//                    << confusionMatrix(i, i) / confusionMatrix.col(i).sum()
//                       * 100 << " ";

    std::cout << "\n\n" << setw(IOMANIP_WIDTH) << "Total population | "
              << testingExamples.size()
              << "\n"
              << "\n" << setw(IOMANIP_WIDTH) << "Accuracy | "
              << globalAccuracy * 100 << " %" << std::endl;

//        for (int i = 0; i < testingExampleClasses.size(); i++)
//            std::cout << "  -- " << testingExampleCla7sses.at(i).className
//                      << " | " << (double) accurateClassificationsPerClass.at(i)
//                                  / classCount.at(i) * 100 << " %" << std::endl;

// Confusion matrices per class
    for (int k = 0; k < confusionPerClass.size(); k++)
    {
        std::cout << "\n\n\n";
        std::cout << "> " << braces(testingExampleClasses.at(k)
                                            .className) << std::endl;

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

//                std::cout << " | "
//                          << confusionPerClass.at(k)(i, i) /
//                             confusionPerClass.at(k).row(i).sum() *
//                             100;
            std::cout << std::endl;
        }
//            std::cout << setw(20) << " ";
//            for (int i = 0; i < confusionPerClass.at(k).cols(); i++)
//                std::cout
//                        << setw(8 + 5)
//                        << confusionPerClass.at(k)(i, i) /
//                           confusionPerClass.at(k).col(i).sum()
//                           * 100 << " ";
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
                                           { 8, "Digits" }});

    // TODO: Klasyfikator

    std::map<std::string, std::string> dataSetTrainingFilenames
            {{ "Identity",             "./data/identity-train.csv" },
             { "Iris",                 "./data/iris-train.csv" },
             { "Iris (normalised)",    "./data/iris-normalised-train.csv" },
             { "Iris (standardised)",  "./data/iris-standardised-train.csv" },
             { "Seeds",                "./data/seeds-train.csv" },
             { "Seeds (normalised)",   "./data/seeds-normalised-train.csv" },
             { "Seeds (standardised)", "./data/seeds-standardised-train.csv" },
             { "Digits",               "./data/digits-train.csv" }};

    std::map<std::string, std::string> dataSetTestingFilenames
            {{ "Identity",             "./data/identity-test.csv" },
             { "Iris",                 "./data/iris-test.csv" },
             { "Iris (normalised)",    "./data/iris-normalised-test.csv" },
             { "Iris (standardised)",  "./data/iris-standardised-test.csv" },
             { "Seeds",                "./data/seeds-test.csv" },
             { "Seeds (normalised)",   "./data/seeds-normalised-test.csv" },
             { "Seeds (standardised)", "./data/seeds-standardised-test.csv" },
             { "Digits",               "./data/digits-test.csv" }};


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
    std::vector<TrainingExampleClass> trainingExampleClasses
            = readTrainingExamplesFromCsvFile
                    (dataSetTrainingFilenames[dataSet]);

    std::vector<TrainingExample> trainingExamples;
    for (auto const &trainingExampleClass
            : trainingExampleClasses)
        trainingExamples.insert(trainingExamples.end(),
                                trainingExampleClass.trainingExamples.begin(),
                                trainingExampleClass.trainingExamples.end());

    std::vector<TrainingExampleClass> testingExampleClasses
            = readTrainingExamplesFromCsvFile
                    (dataSetTestingFilenames[dataSet]);

    std::vector<TrainingExample> testingExamples;
    for (auto const &testingExampleClass
            : testingExampleClasses)
        testingExamples.insert(testingExamples.end(),
                               testingExampleClass.trainingExamples.begin(),
                               testingExampleClass.trainingExamples.end());


    // Do the math
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

        for (auto const &neurons : split(hiddenLayerNeuronNumber))
            layersNeurons.push_back(atoi(neurons.data()));

        layersNeurons.push_back
                (static_cast<int>(trainingExamples.at(0).outputs.size()));

        // Biases
        std::string biases;
        std::cout << setw(IOMANIP_WIDTH) << "Enable biases per layer" << " | ";
        std::getline(std::cin, biases);

        std::vector<bool> biasesPerLayer;

        for (auto const &bias : split(biases))
            biasesPerLayer.push_back((bool)atoi(bias.data()));

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
                            trainingExamples, trainingExampleClasses,
                            dataSetTrainingFilenames[dataSet],
                            perceptronFilename);
        printTestingResults(multiLayerPerceptron,
                            testingExamples, testingExampleClasses,
                            dataSetTestingFilenames[dataSet],
                            perceptronFilename);
    }

    std::cout << "\n\n";
    system("pause");
    return 0;
}

//    std::vector<TrainingExampleClass> trainingExampleClasses
//            = readTrainingExamplesFromCsvFile
//                    ("E:\\Studia\\semestr-4\\IAD\\iad-2a\\src\\data"
//                     "\\iris\\iris-test-standardised.csv");
//
////    for (auto const &i : trainingExampleClasses)
////    {
////        for (auto const &j : i.trainingExamples)
////        {
////            cout << "> inputs\n" << j.inputs
////                 << "> outputs\n" << j.outputs
////                 << endl;
////        }
////    }
//
//    double ratio = 1.0;
//    std::vector<TrainingExample> trainingExamples;
//    std::vector<TrainingExample> testingExamples;
//    for (auto const &trainingExampleClass : trainingExampleClasses)
//        for (int i = 0;
//             i < trainingExampleClass.trainingExamples.size();
//             i++)
//            if (i < trainingExampleClass.trainingExamples.size() * ratio)
//                trainingExamples.push_back
//                        (trainingExampleClass.trainingExamples.at(i));
//            else
//                testingExamples.push_back
//                        (trainingExampleClass.trainingExamples.at(i));
//
//    //-------
//
//    constexpr int
//            n = 4,
//            h = 2,
//            m = 4;
//
//    int seed = 0;
////    int seed = static_cast<int>(time(nullptr));
//    MultiLayerPerceptron::initialiseRandomNumberGenerator(seed);
//    MultiLayerPerceptron mlp {{ (int) trainingExamples[0].inputs.size(), 32, 32,
//                                (int) trainingExamples[0].outputs.size() },
//                              std::vector<bool>(3, true) };
//
//    std::vector<TrainingExample> trainingData
//            {
//                    {
//                            (Vector { n } << 1.0, 0.0, 0.0, 0.0).finished(),
//                            (Vector { m } << 1.0, 0.0, 0.0, 0.0).finished()
//                    },
//                    {
//                            (Vector { n } << 0.0, 1.0, 0.0, 0.0).finished(),
//                            (Vector { m } << 0.0, 1.0, 0.0, 0.0).finished()
//                    },
//                    {
//                            (Vector { n } << 0.0, 0.0, 1.0, 0.0).finished(),
//                            (Vector { m } << 0.0, 0.0, 1.0, 0.0).finished()
//                    },
//                    {
//                            (Vector { n } << 0.0, 0.0, 0.0, 1.0).finished(),
//                            (Vector { m } << 0.0, 0.0, 0.0, 1.0).finished()
//                    }
//            };
//
//    int const numberOfEpochs = 10'000;//100
//    double const costGoal = 0.00001;
//    double const learningCoefficient = 0.01;//0.1
//    double const learningCoefficientChange = -0.005;//-0.08
//    double const momentumCoefficient = 0.75;
//    bool const shuffleTrainingData = true;//false
//    int const epochInterval = 100;
//
//    MultiLayerPerceptron::TrainingResults trainingResults
//            = mlp.train(trainingExamples,
//                        numberOfEpochs, costGoal,
//                        learningCoefficient, learningCoefficientChange,
//                        momentumCoefficient,
//                        shuffleTrainingData,
//                        epochInterval);
//
//    {
//        std::ofstream file("training-result-error", std::ios::trunc);
//        for (int i = 0; i < trainingResults.costPerEpochInterval.size(); i++)
//            file << i * trainingResults.epochInterval << ","
//                 << trainingResults.costPerEpochInterval.at(i) << std::endl;
//    }
//
//    std::cout << "training results" << std::endl;
//    std::cout << "epoch interval: " << trainingResults.epochInterval
//              << std::endl;
//    for (auto const &i : trainingResults.costPerEpochInterval)
//        std::cout << i << std::endl;
//
//    MultiLayerPerceptron::TestingResults testingResults
//            = mlp.test(trainingExamples);
//
//    std::cout << "testing results" << std::endl;
//    std::cout << "global cost: " << testingResults.globalCost << std::endl;
////    for (auto const &i : testingResults.testingResultsPerExample)
////    {
////        std::cout << "neurons\n";
////        for (auto const &j : i.neurons)
////            std::cout << ">" << std::endl
////                      << j << std::endl;
////        std::cout << "targets\n";
////        std::cout << i.targets << std::endl;
////        std::cout << "errors\n";
////        for (auto const &j : i.errors)
////            std::cout << ">" << std::endl
////                      << j << std::endl;
////        std::cout << "cost\n";
////        std::cout << i.cost << std::endl;
////    }
//
//    int avg = 0;
//    for (auto const &i : testingResults.testingResultsPerExample)
//    {
//        Vector::Index classNumber;
//        i.neurons.back().array().maxCoeff(&classNumber);
//        avg += (int) i.targets(classNumber);
//    }
//    cout << "dokl: " << (double) avg / trainingExamples.size() * 100 << " %" <<
//         endl;
//
//    mlp.saveToFile("multi-layer-perceptron.mlp");
//    MultiLayerPerceptron loadedFromFile { "multi-layer-perceptron.mlp" };
//
////    using std::cout;
////    using std::endl;
////    cout << endl;
////    cout << mlp((Vector { n } << 1.0, 0.0, 0.0, 0.0).finished()) << "\n\n";
////    cout << loadedFromFile((Vector { n } << 1.0, 0.0, 0.0, 0.0).finished()) <<
////         "\n\n";
////    cout << mlp((Vector { n } << 0.0, 1.0, 0.0, 0.0).finished()) << "\n\n";
////    cout << mlp((Vector { n } << 0.0, 0.0, 1.0, 0.0).finished()) << "\n\n";
////    cout << mlp((Vector { n } << 0.0, 0.0, 0.0, 1.0).finished()) << "\n\n";
//
//
////    using namespace NeuralNetworks;
////    Sigmoid sigmoid;
////    std::unique_ptr<Cloneable<ActivationFunction>> cloneable
////            = std::make_unique<Sigmoid>();
////    Sigmoid sigmoidCopyAssignment = sigmoid;
////    PerceptronLayer layer1 { 3, 4, Sigmoid {}};
////    PerceptronLayer layer2 { 3, 4, RectifiedLinearUnit {}};
////    PerceptronLayer layer3 { 3, 4 };
//
//
//    return 0;
//}


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
