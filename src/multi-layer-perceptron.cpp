///////////////////////////////////////////////////////////////////// | Includes
#include "multi-layer-perceptron.hpp"

#include <algorithm>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

#include <cereal/types/vector.hpp>

///////////////////////////////////////////////////// | TODO: Name this section.
using Array = Eigen::ArrayXd;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////// | Namespace: HelperFunctions <
    namespace HelperFunctions
    {
        template <typename T>
        T &shuffle
                (T &container)
        {
            static auto randomNumberGenerator
                    = std::default_random_engine { std::random_device {}() };

            //T shuffledContainer { container };
            std::shuffle(std::begin(container),
                         std::end(container),
                         randomNumberGenerator);

            return container;
        }

        template <typename T>
        T reverse
                (T const &container)
        {
            T reversedContainer { std::size(container) };

            std::reverse_copy(std::begin(container),
                              std::end(container),
                              std::begin(reversedContainer));

            return reversedContainer;
        }
    }

    double getAccuracy
            (MultiLayerPerceptron const &multiLayerPerceptron,
             std::vector<TrainingExample> const
             &testingExamples)
    {
        int globalNumberOfAccurateClassifications = 0;

        MultiLayerPerceptron::TestingResults testingResults =
                multiLayerPerceptron.test(
                testingExamples);

        for (auto const &testingResultsPerExample
                : testingResults.testingResultsPerExample)
        {
            Vector::Index predictedClass, actualClass;
            testingResultsPerExample.neurons
                    .back().array().maxCoeff(&predictedClass);
            testingResultsPerExample.targets.array().maxCoeff(&actualClass);

            globalNumberOfAccurateClassifications
                    += (int) (predictedClass == actualClass);
        }
        double globalAccuracy = (double) globalNumberOfAccurateClassifications
                                / testingExamples.size();
        return globalAccuracy;
    }

    //////////////////////////////////////////// | Class: MultiLayerPerceptron <
    //============================================================= | Methods <<
    //----------------------------------------------------- | Static methods <<<
    void MultiLayerPerceptron::initialiseRandomNumberGenerator
            (int const &seed)
    {
        PerceptronLayer::initialiseRandomNumberGenerator(seed);
    }

    //------------------------------------------------------- | Constructors <<<
    MultiLayerPerceptron::MultiLayerPerceptron
            (std::vector<int> const &numberOfNeurons,
             std::vector<bool> const &enableBiasPerLayer)
            :
            layers { createLayers(numberOfNeurons,
                                  enableBiasPerLayer) }
    {
    }

    MultiLayerPerceptron::MultiLayerPerceptron
            (std::string const &filename)
//            :
//            MultiLayerPerceptron({1, 1}, {true, true})
    {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        {
            cereal::BinaryInputArchive binaryInputArchive(file);
            binaryInputArchive(*this);
        }
        file.close();
    }

    //---------------------------------------------------------- | Operators <<<
    Vector MultiLayerPerceptron::operator()
            (Vector const &inputs) const
    {
        return feedForward(inputs);
    }

    //----------------------------------------------------- | Main behaviour <<<
    Vector MultiLayerPerceptron::feedForward
            (Vector const &inputs) const
    {
        Vector neurons = inputs;

        for (auto const &layer : layers)
            neurons = layer.feedForward(neurons);

        return neurons;
    }

    MultiLayerPerceptron::TrainingResults MultiLayerPerceptron::train
            (std::vector<TrainingExample> const &trainingExamples,
             std::vector<TrainingExample> const &testingExamples,
             int const numberOfEpochs,
             double const costGoal,
             double learningCoefficient,
             double const learningCoefficientChange,
             double const momentumCoefficient,
             bool const shuffleTrainingData,
             int const epochInterval)
    {
        // Prepare results
        TrainingResults trainingResults;
        trainingResults.epochInterval = epochInterval;

        // Create container of training examples' iterators
        std::vector<decltype(trainingExamples.cbegin())>
                trainingExamplesIterators;

        for (auto trainingExample = trainingExamples.cbegin();
             trainingExample != trainingExamples.cend();
             ++trainingExample)
        {
            trainingExamplesIterators.emplace_back(trainingExample);
        }

        // Create container of layers' iterators
        std::vector<decltype(layers.begin())>
                layersIterators;

        for (auto layer = layers.begin();
             layer != layers.end();
             ++layer)
        {
            layersIterators.emplace_back(layer);
        }

        auto layersIteratorsReversed
                { HelperFunctions::reverse(layersIterators) };


        // Train the network
        for (int epoch = 0;
             epoch < numberOfEpochs;
             epoch++)
        {
            double costPerEpoch = 0.0;

            for (auto const &trainingExamplesIterator
                    : (shuffleTrainingData
                       ? HelperFunctions::shuffle(trainingExamplesIterators)
                       : trainingExamplesIterators))
            {
                auto const &firstLayerInputs
                        = trainingExamplesIterator->inputs;

                std::vector<Vector> neurons
                        { firstLayerInputs };

                std::vector<Vector> outputsDerivatives;

                for (auto const &layer
                        : layersIterators)
                {
                    Vector outputsDry = layer->calculateOutputs(neurons.back());
                    outputsDerivatives.emplace_back
                            (layer->calculateOutputsDerivative(outputsDry));
                    neurons.emplace_back
                            (layer->activate(outputsDry));
                }

                auto const &lastLayerOutputs
                        = neurons.back();

                auto const &lastLayerTargets
                        = trainingExamplesIterator->outputs;

                auto const lastLayerErrors
                        = lastLayerTargets - lastLayerOutputs;

                std::vector<Vector> errors
                        { lastLayerErrors };

                {
                    auto derivatives = outputsDerivatives.crbegin();
                    for (auto const &layer
                            : layersIteratorsReversed)
                    {
                        errors.emplace_back
                                (layer->backpropagate
                                        (*derivatives, errors.back()));
                        ++derivatives;
                    }
                    errors = HelperFunctions::reverse(errors);
                }

                // Increment epoch's total cost
                costPerEpoch += errors.back().array().square().sum();

                // Calculate steps for weights and biases
                {
                    auto inputsIterator = neurons.cbegin();
                    auto derivativesIterator = outputsDerivatives.cbegin();
                    auto errorsIterator = errors.cbegin() + 1;

                    for (auto &layer
                            : layersIterators)
                    {
                        layer->calculateNextStep(*inputsIterator,
                                                 *errorsIterator,
                                                 *derivativesIterator);
                        ++inputsIterator;
                        ++derivativesIterator;
                        ++errorsIterator;
                    }
                }

                // Update layers
                for (auto &layer
                        : layersIterators)
                    layer->update(learningCoefficient, momentumCoefficient);
            }

            // Check if goal total error across all
            // training examples was achieved
            costPerEpoch /= trainingExamples.size();

            if (epoch % trainingResults.epochInterval == 0
                || epoch == 0 || epoch == numberOfEpochs - 1)
            {
                trainingResults.costPerEpochInterval.emplace_back(costPerEpoch);
                std::cout << "\r epoch: " << epoch;

                trainingResults.accuracyTraining
                        .emplace_back(getAccuracy(*this,
                                                  trainingExamples));
                trainingResults.accuracyTesting
                        .emplace_back(getAccuracy(*this,
                                                  testingExamples));
            }

            if (costPerEpoch < costGoal)
                break;

            // Reduce learning coefficient with every epoch
            learningCoefficient -= (learningCoefficientChange / numberOfEpochs);
        }

        return trainingResults;
    }

    MultiLayerPerceptron::TestingResults MultiLayerPerceptron::test
            (std::vector<TrainingExample> const &testingExamples) const
    {
        // Prepare results
        TestingResults testingResults;
        testingResults.globalCost = 0.0;

        // Create container of layers' iterators
        std::vector<decltype(layers.begin())>
                layersIterators;

        for (auto layer = layers.begin();
             layer != layers.end();
             ++layer)
        {
            layersIterators.emplace_back(layer);
        }

        auto layersIteratorsReversed
                { HelperFunctions::reverse(layersIterators) };

        // Test the network
        for (auto const &testingExample
                : testingExamples)
        {
            auto const &firstLayerInputs
                    = testingExample.inputs;

            std::vector<Vector> neurons
                    { firstLayerInputs };

            std::vector<Vector> outputsDerivatives;

            for (auto const &layer
                    : layersIterators)
            {
                Vector outputsDry = layer->calculateOutputs(neurons.back());
                outputsDerivatives.emplace_back
                        (layer->calculateOutputsDerivative(outputsDry));
                neurons.emplace_back
                        (layer->activate(outputsDry));
            }

            auto const &lastLayerOutputs
                    = neurons.back();

            auto const &lastLayerTargets
                    = testingExample.outputs;

            auto const lastLayerErrors
                    = lastLayerTargets - lastLayerOutputs;

            std::vector<Vector> errors
                    { lastLayerErrors };

            {
                auto derivatives = outputsDerivatives.crbegin();
                for (auto const &layer
                        : layersIteratorsReversed)
                {
                    errors.emplace_back
                            (layer->backpropagate
                                    (*derivatives, errors.back()));
                    ++derivatives;
                }
                errors = HelperFunctions::reverse(errors);
            }

            // Calculate cost
            double cost = errors.back().array().square().sum();
            testingResults.globalCost += cost;

            // Save testing results per example
            testingResults.testingResultsPerExample.push_back
                    ({ neurons,
                       lastLayerTargets,
                       errors,
                       cost });
        }

        // Average out global error
        testingResults.globalCost /= testingExamples.size();

        // Return testing report
        return testingResults;
    }

    void MultiLayerPerceptron::saveToFile
            (std::string const &filename) const
    {
        std::ofstream file;
        file.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        {
            cereal::BinaryOutputArchive binaryOutputArchive(file);
            binaryOutputArchive(*this);
        }
        file.close();
    }

    template <typename Archive>
    void MultiLayerPerceptron::save
            (Archive &archive) const
    {
        archive(layers);
    }

    template <typename Archive>
    void MultiLayerPerceptron::load
            (Archive &archive)
    {
        archive(layers);
    }

    void MultiLayerPerceptron::readFromFile
            (std::string const &filename)
    {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        {
            cereal::BinaryInputArchive binaryInputArchive(file);
            binaryInputArchive(*this);
        }
        file.close();
//        std::ifstream file;
//        file.open(filename);
//        {
//            std::string line;
//            while (std::getline(file, line))
//            {
//                std::istringstream inputStringStream { line };
//
//                std::string token;
//                if (!(inputStringStream >> token))
//                    break;
//
//                if (token == ">>")
//                    continue;
//
//                if (token == ">")
//                {
//
//                }
//            }
//        }
//        file.close();
    }

    //----------------------------------------------------- | Helper methods <<<
    std::vector<PerceptronLayer> MultiLayerPerceptron::createLayers
            (std::vector<int> const &numberOfNeuronsPerLayer,
             std::vector<bool> enableBiasPerLayer) const
    {
        std::vector<PerceptronLayer> layers;

//        if (enableBiasPerLayer.empty())
//            enableBiasPerLayer = std::vector<bool>
//                    (numberOfNeuronsPerLayer.size() - 1, true);
        auto enableBias = enableBiasPerLayer.cbegin();

        for (auto numberOfNeurons = numberOfNeuronsPerLayer.cbegin();
             numberOfNeurons != numberOfNeuronsPerLayer.cend() - 1;
             ++numberOfNeurons)
        {
            layers.emplace_back(*numberOfNeurons,
                                *(numberOfNeurons + 1),
                                Sigmoid {},
                                *enableBias);
            ++enableBias;
        }

        return layers;
    }

    std::vector<Vector> MultiLayerPerceptron::feedForwardPerLayer
            (Vector const &inputs) const
    {
        std::vector<Vector> outputs { inputs };

        for (auto const &layer : layers)
            outputs.emplace_back(layer.feedForward(outputs.back()));

        outputs.erase(outputs.begin());
        //std::reverse(outputs.begin(), outputs.end());

        return outputs;
    }

    std::vector<Vector> MultiLayerPerceptron::backpropagateErrorsPerLayer
            (std::vector<Vector> const &inputsPerLayer,
             Vector const &errors) const
    {
//            std::vector<Eigen::VectorXd> errors
//                        { trainingExample.outputs - outputsPerLayer.back() };
//
//                int x = layers.size() - 1;
//                for (auto layer = layers.end() - 1;
//                     layer != layers.begin(); layer--)
//                {
//
//                    errors.push_back
//                            (layer->backpropagate(errors.back(),
//                                                  outputsPerLayer.at(x)));
//
//                    x--;
//
//                    //VectorXd target{ VectorXd::Zero(layer->numberOfInputs()) };
//
//                    /*for (int i = 0; i < layer->numberOfOutputs(); i++)
//                    {
//                        for (int j = 0; j < layer->numberOfInputs(); j++)
//                        {
//                            target(j) +=
//                                (outputs.back().array() * (VectorXd::Ones(outputs.back().size()).array() - outputs.back().array())).matrix()
//                                targets.back()(i);
//                        }
//                    }
//
//                    targets.push_back(target);*/
//                }
//                std::reverse(errors.begin(), errors.end());
//            errorsSum += errors;
        std::vector<Eigen::VectorXd> propagatedErrors { errors };

        auto inputPerLayer = inputsPerLayer.rbegin();
        for (auto layer = layers.rbegin();
             layer != layers.rend() - 1; ++layer)
        {
            propagatedErrors.emplace_back
                    (layer->backpropagate(*inputPerLayer,
                                          propagatedErrors.back()));
            ++inputPerLayer;
        }

        std::reverse(propagatedErrors.begin(), propagatedErrors.end());

        return propagatedErrors;
    }
}

////////////////////////////////////////////////////////////////////////////////
