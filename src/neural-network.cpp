///////////////////////////////////////////////////////////////////// | Includes
#include "neural-network.hpp"

#include <algorithm>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <tuple>



/////////////////////////////////////////////////////////// | Using declarations
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

    std::vector<AffineLayer> createLayers
            (std::vector<int> const &numberOfNeuronsPerLayer,
             std::vector<bool> const &enableBiasPerLayer)
    {
        std::vector<AffineLayer> layers;

        for (auto[numberOfNeurons, enableBias]
             = std::make_tuple(numberOfNeuronsPerLayer.cbegin(),
                               enableBiasPerLayer.cbegin());
             numberOfNeurons != numberOfNeuronsPerLayer.cend() - 1;
             ++numberOfNeurons, ++enableBias)
        {
            layers.emplace_back(*numberOfNeurons,
                                *(numberOfNeurons + 1),
                                Sigmoid {},
                                *enableBias);
        }

        return layers;
    }

    double getAccuracy
            (NeuralNetwork const &multiLayerPerceptron,
             std::vector<TrainingExample> const
             &testingExamples)
    {
        int globalNumberOfAccurateClassifications = 0;

        NeuralNetwork::TestingResults testingResults =
                multiLayerPerceptron.test(testingExamples);

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

    /////////////////////////////////////////////////// | Class: NeuralNetwork <
    //============================================================= | Methods <<
    //----------------------------------------------------- | Static methods <<<
    void NeuralNetwork::initialiseRandomNumberGenerator
            (int const &seed)
    {
        AffineLayer::initialiseRandomNumberGenerator(seed);
    }

    //------------------------------------------------------- | Constructors <<<
//    NeuralNetwork::NeuralNetwork
//            (std::vector<int> const &numberOfNeurons,
//             std::vector<bool> const &enableBiasPerLayer)
//            :
//            layers { createLayers(numberOfNeurons,
//                                  enableBiasPerLayer) }
//    {
//    }

    NeuralNetwork::NeuralNetwork
            (std::vector<std::unique_ptr<NeuralNetworkLayer>> layers)
            :
            layers { std::move(layers) }
    {
    }

    NeuralNetwork::NeuralNetwork
            (std::string const &filename)
    {
        readFromFile(filename);
    }

    //---------------------------------------------------------- | Operators <<<
    Vector NeuralNetwork::operator()
            (Vector const &inputs) const
    {
        return feedForward(inputs);
    }

    //----------------------------------------------------- | Main behaviour <<<
    Vector NeuralNetwork::feedForward
            (Vector const &inputs) const
    {
        Vector neurons = inputs;

        for (auto const &layer : layers)
            neurons = layer->feedForward(neurons);

        return neurons;
    }

    NeuralNetwork::TrainingResults NeuralNetwork::train
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
                    Vector outputsDry = (*layer)->calculateOutputs(neurons.back
                            ());
                    outputsDerivatives.emplace_back
                            ((*layer)->calculateOutputsDerivative(outputsDry));
                    neurons.emplace_back
                            ((*layer)->activate(outputsDry));
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
                    auto inputsIterator = neurons.crbegin() + 1;
                    auto outputsIterator = neurons.crbegin();
                    auto derivativesIterator = outputsDerivatives.crbegin();

                    for (auto const &layer
                            : layersIteratorsReversed)
                    {
                        errors.emplace_back
                                ((*layer)->backpropagate
                                        (*inputsIterator,
                                         errors.back(),
                                         *outputsIterator,
                                         *derivativesIterator));
                        ++inputsIterator;
                        ++outputsIterator;
                        ++derivativesIterator;
                    }
                    errors = HelperFunctions::reverse(errors);
                }

                // Increment epoch's total cost
                costPerEpoch += errors.back().array().square().sum();

                // Calculate steps for weights and biases
                {
                    auto inputsIterator = neurons.cbegin();
                    auto outputsIterator = neurons.cbegin() + 1;
                    auto derivativesIterator = outputsDerivatives.cbegin();
                    auto errorsIterator = errors.cbegin() + 1;

                    for (auto &layer
                            : layersIterators)
                    {
                        (*layer)->calculateNextStep(*inputsIterator,
                                                 *errorsIterator,
                                                 *outputsIterator,
                                                 *derivativesIterator);
                        ++inputsIterator;
                        ++outputsIterator;
                        ++derivativesIterator;
                        ++errorsIterator;
                    }
                }

                // Update layers
                for (auto &layer
                        : layersIterators)
                    (*layer)->update(learningCoefficient, momentumCoefficient);
            }

            // Check if goal total error across all
            // training examples was achieved
            costPerEpoch /= trainingExamples.size();

            if (epoch % trainingResults.epochInterval == 0
                || epoch == 0 || epoch == numberOfEpochs - 1)
            {
                trainingResults.costPerEpochInterval.emplace_back(costPerEpoch);
                std::cout << "\r epoch: " << epoch << " | cost: " <<
                costPerEpoch;
                std::cout.flush();

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

    NeuralNetwork::TestingResults NeuralNetwork::test
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
                Vector outputsDry = (*layer)->calculateOutputs(neurons.back());
                outputsDerivatives.emplace_back
                        ((*layer)->calculateOutputsDerivative(outputsDry));
                neurons.emplace_back
                        ((*layer)->activate(outputsDry));
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
                auto inputsIterator = neurons.crbegin() + 1;
                auto outputsIterator = neurons.crbegin();
                auto derivativesIterator = outputsDerivatives.crbegin();

                for (auto const &layer
                        : layersIteratorsReversed)
                {
                    errors.emplace_back
                            ((*layer)->backpropagate
                                    (*inputsIterator,
                                     errors.back(),
                                     *outputsIterator,
                                     *derivativesIterator));
                    ++inputsIterator;
                    ++outputsIterator;
                    ++derivativesIterator;
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

    void NeuralNetwork::saveToFile
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

    void NeuralNetwork::readFromFile
            (std::string const &filename)
    {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        {
            cereal::BinaryInputArchive binaryInputArchive(file);
            binaryInputArchive(*this);
        }
        file.close();
    }

    //----------------------------------------------------- | Helper methods <<<
//    std::vector<Vector> NeuralNetwork::feedForwardPerLayer
//            (Vector const &inputs) const
//    {
//        std::vector<Vector> outputs { inputs };
//
//        for (auto const &layer : layers)
//            outputs.emplace_back(layer.feedForward(outputs.back()));
//
//        outputs.erase(outputs.begin());
//
//        return outputs;
//    }
//
//    std::vector<Vector> NeuralNetwork::backpropagateErrorsPerLayer
//            (std::vector<Vector> const &inputsPerLayer,
//             Vector const &errors) const
//    {
//        std::vector<Eigen::VectorXd> propagatedErrors { errors };
//
//        auto inputPerLayer = inputsPerLayer.rbegin();
//        for (auto layer = layers.rbegin();
//             layer != layers.rend() - 1; ++layer)
//        {
//            propagatedErrors.emplace_back
//                    (layer->backpropagate(*inputPerLayer,
//                                          propagatedErrors.back()));
//            ++inputPerLayer;
//        }
//
//        std::reverse(propagatedErrors.begin(), propagatedErrors.end());
//
//        return propagatedErrors;
//    }
}

////////////////////////////////////////////////////////////////////////////////
