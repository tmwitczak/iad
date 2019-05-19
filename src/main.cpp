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
//            std::cout << ".";
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

#include <utility>

std::string askUserForInput
        (std::string_view const &question,
         std::vector<std::pair<int, std::string>> options)
{
    for (auto const &option
            : options)
        std::cout << option.first << " | " << option.second << std::endl;

    std::cout << question << " | ";
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
    std::cout << "Multi-layer perceptron filename | ";
    cin.ignore();
    std::getline(cin, perceptronFilename);

    if (mode == "Training")
    {
        std::vector<TrainingExampleClass> trainingExampleClasses
                = readTrainingExamplesFromCsvFile
                        (dataSetTrainingFilenames[dataSet]);

        std::vector<TrainingExample> trainingExamples;
        for (auto const &trainingExampleClass
                : trainingExampleClasses)
            trainingExamples.insert(trainingExamples.end(),
                                    trainingExampleClass.trainingExamples.begin(),
                                    trainingExampleClass.trainingExamples.end());

        MultiLayerPerceptron::initialiseRandomNumberGenerator
                (static_cast<int>(time(nullptr)));
        MultiLayerPerceptron multiLayerPerceptron
                {{ static_cast<int>(trainingExamples.at(0).inputs.size()),
                         64,
                         static_cast<int>(trainingExamples.at(
                                 0).outputs.size()), },
                 { true, true }};

        int numberOfEpochs;
        double costGoal;
        double learningCoefficientStart;
        double learningCoefficientEnd;
        double momentumCoefficient;
        bool shuffleTrainingData;
        int epochInterval;

        std::cout << "Number of epochs | ";
        std::cin >> numberOfEpochs;
        std::cout << "Cost goal | ";
        std::cin >> costGoal;
        std::cout << "Learning coefficient (start) | ";
        std::cin >> learningCoefficientStart;
        std::cout << "Learning coefficient (end) | ";
        std::cin >> learningCoefficientEnd;
        std::cout << "Momentum coefficient | ";
        std::cin >> momentumCoefficient;
        std::cout << "Shuffle training data | ";
        std::cin >> shuffleTrainingData;
        std::cout << "Epoch interval | ";
        std::cin >> epochInterval;

        MultiLayerPerceptron::TrainingResults trainingResults
                = multiLayerPerceptron.train(trainingExamples,
                                             numberOfEpochs,
                                             costGoal,
                                             learningCoefficientStart,
                                             learningCoefficientEnd
                                             - learningCoefficientStart,
                                             momentumCoefficient,
                                             shuffleTrainingData,
                                             epochInterval);
        saveErrorToFile("iad2-cwiczenie-przypadek1-funkcja-kosztu",
                        trainingResults);

        multiLayerPerceptron.saveToFile(perceptronFilename);
    }
    else
    {
        std::vector<TrainingExampleClass> testingExampleClasses
                = readTrainingExamplesFromCsvFile
                        (dataSetTestingFilenames[dataSet]);

        std::vector<TrainingExample> testingExamples;
        for (auto const &testingExampleClass
                : testingExampleClasses)
            testingExamples.insert(testingExamples.end(),
                                   testingExampleClass.trainingExamples.begin(),
                                   testingExampleClass.trainingExamples.end());

        MultiLayerPerceptron multiLayerPerceptron(perceptronFilename);

        MultiLayerPerceptron::TestingResults testingResults
                = multiLayerPerceptron.test(testingExamples);

        int avg = 0;
        for (auto const &i : testingResults.testingResultsPerExample)
        {
            Vector::Index classNumber;
            i.neurons.back().array().maxCoeff(&classNumber);
            avg += (int) (i.targets(classNumber) == 1.0);
        }
        cout << "dokl: " << (double) avg / testingExamples.size() * 100 << " %"
             <<
             endl;
    }

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
