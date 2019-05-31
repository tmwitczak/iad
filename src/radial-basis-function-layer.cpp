///////////////////////////////////////////////////////////////////// | Includes
#include "radial-basis-function-layer.hpp"
#include "identity.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

/////////////////////////////////////////////////////////// | Using declarations
using Array = Eigen::ArrayXd;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////// | Class: RadialBasisFunctionLayer <
    //============================================================= | Methods <<
    //----------------------------------------------------- | Static methods <<<
    void RadialBasisFunctionLayer::initialiseRandomNumberGenerator
            (int const seed)
    {
        srand(static_cast<unsigned int>(seed));
    }

    //------------------------------------------------------- | Constructors <<<
    RadialBasisFunctionLayer::RadialBasisFunctionLayer
            ()
            :
            RadialBasisFunctionLayer(1, 1, Identity {})
    {
    }

    RadialBasisFunctionLayer::RadialBasisFunctionLayer
            (int const numberOfInputs,
             int const numberOfOutputs,
             ActivationFunction const &activationFunction)
            :
            weights { std::sqrt(2.0 / (numberOfInputs + numberOfOutputs))
                      * Matrix::Random(numberOfOutputs, numberOfInputs) },
            deltaWeights { Matrix::Zero(numberOfOutputs, numberOfInputs) },
            momentumWeights { Matrix::Zero(numberOfOutputs, numberOfInputs) },

            biases { std::sqrt(2.0 / (numberOfInputs + numberOfOutputs))
                     * ((Vector::Random(numberOfOutputs).array() + 1.0)
                        / 2.0).matrix() },
            deltaBiases { Vector::Zero(numberOfOutputs) },
            momentumBiases { Vector::Zero(numberOfOutputs) },

            activationFunction { activationFunction.clone() },
            currentNumberOfSteps { 0 }
    {
    }

    RadialBasisFunctionLayer::RadialBasisFunctionLayer
            (std::string const &filename)
    {
        std::ifstream file;
        file.open(filename, std::ios::binary);
        {
            cereal::BinaryInputArchive binaryInputArchive(file);
            binaryInputArchive(*this);
        }
        file.close();
    }

    RadialBasisFunctionLayer::RadialBasisFunctionLayer
            (RadialBasisFunctionLayer const &layer)
            :
            weights { layer.weights },
            deltaWeights { layer.deltaWeights },
            momentumWeights { layer.momentumWeights },
            biases { layer.biases },
            deltaBiases { layer.deltaBiases },
            momentumBiases { layer.momentumBiases },
            activationFunction { layer.activationFunction->clone() },
            currentNumberOfSteps { layer.currentNumberOfSteps }
    {
    }

    //---------------------------------------------------------- | Operators <<<
    Vector RadialBasisFunctionLayer::operator()
            (Vector const &inputs) const
    {
        return feedForward(inputs);
    }

    //------------------------------ | Interface: Cloneable | Implementation <<<
    std::unique_ptr<NeuralNetworkLayer> RadialBasisFunctionLayer::clone
            () const
    {
        return std::make_unique<RadialBasisFunctionLayer>(*this);
    }

    //----------------------------------------------------- | Main behaviour <<<
    Vector RadialBasisFunctionLayer::calculateOutputs
            (Vector const &inputs) const
    {
        Vector outputs
                { numberOfOutputs() };

        for (int i = 0;
             i < numberOfOutputs();
             ++i)
        {
            outputs(i)
                    = std::exp(-biases(i)
                               * std::pow((inputs - weights.row(i)).norm(), 2));
        }

        return outputs;
    }

    Vector RadialBasisFunctionLayer::activate
            (Vector const &outputs) const
    {
        return (*activationFunction)(outputs);
    }

    Vector RadialBasisFunctionLayer::calculateOutputsDerivative
            (Vector const &outputs) const
    {
        return activationFunction->derivative(outputs);
    }

    Vector RadialBasisFunctionLayer::feedForward
            (Vector const &inputs) const
    {
        return calculateOutputs(inputs);
    }

    Vector RadialBasisFunctionLayer::backpropagate
            (Vector const &inputs,
             Vector const &errors,
             Vector const &outputs,
             Vector const &outputsDerivative) const
    {
        Vector propagatedError { numberOfInputs() };
        for (int x = 0; x < numberOfInputs(); ++x)
        {
            double sum1 = 0.0;
            for (int i = 0; i < numberOfOutputs(); ++i)
            {
                double sum2 = 0.0;
                sum2 += 2.0 * (inputs(x) - weights(i, x)) * 1.0;

                sum1 += (-errors(i)) * outputsDerivative(i) * outputs(i) *
                        (-biases(i)) * sum2;
            }
            propagatedError(x) = -sum1;
        }
        return propagatedError;
    }

    void RadialBasisFunctionLayer::calculateNextStep
            (Vector const &inputs,
             Vector const &errors,
             Vector const &outputs,
             Vector const &outputsDerivative)
    {
        for (int i = 0; i < numberOfOutputs(); ++i)
            for (int j = 0; j < numberOfInputs(); ++j)
            {
                // TODO: Include derivative of activation function
                deltaWeights(i, j) -= -errors(i)
                                      * outputsDerivative(i)
                                      * outputs(i)
                                      * (-biases(i))
                                      * 2.0 * (inputs(j) - weights(i, j)) *
                                      (-1.0);
            }

        for (int i = 0; i < numberOfOutputs(); ++i)
        {
            double sumOfSomething = 0.0;
            for (int j = 0; j < numberOfInputs(); ++j)
            {
                sumOfSomething += std::pow(inputs(j) - weights(i, j), 2);
            }
            // TODO: Include derivative of activation function
            deltaBiases(i) -= -errors(i)
                              * outputsDerivative(i)
                              * outputs(i)
                              * (-sumOfSomething);
        }

        ++currentNumberOfSteps;
    }

    void RadialBasisFunctionLayer::update
            (double const learningCoefficient,
             double const momentumCoefficient)
    {
        applyAverageOfDeltaStepsToMomentumStep(learningCoefficient,
                                               momentumCoefficient);
        applyMomentumStepToWeightsAndBiases();
        resetStepData();
    }

    void RadialBasisFunctionLayer::saveToFile
            (std::string const &filename) const
    {
        std::ofstream file;
        file.open(filename, std::ios::binary | std::ios::trunc);
        {
            cereal::BinaryOutputArchive binaryOutputArchive(file);
            binaryOutputArchive(*this);
        }
        file.close();
    }


    //------------------------------------------------------------- | Traits <<<
    int RadialBasisFunctionLayer::numberOfInputs
            () const
    {
        return weights.cols();
    }

    int RadialBasisFunctionLayer::numberOfOutputs
            () const
    {
        return weights.rows();
    }

    //--------------------------------------------------- | Helper functions <<<
    void RadialBasisFunctionLayer::applyAverageOfDeltaStepsToMomentumStep
            (double const learningCoefficient,
             double const momentumCoefficient)
    {
//        (momentumWeights *= momentumCoefficient)
//                += (learningCoefficient
//                    * (deltaWeights / currentNumberOfSteps));

        momentumWeights.noalias()
                = momentumCoefficient * momentumWeights
                  + (learningCoefficient / currentNumberOfSteps)
                    * deltaWeights;

//        if (isBiasEnabled)
//            (momentumBiases *= momentumCoefficient)
//                    += (learningCoefficient
//                        * (deltaBiases / currentNumberOfSteps));


        momentumBiases.noalias()
                = momentumCoefficient * momentumBiases
                  + (learningCoefficient / currentNumberOfSteps)
                    * deltaBiases;
    }

    void RadialBasisFunctionLayer::applyMomentumStepToWeightsAndBiases
            ()
    {
        weights.noalias() += momentumWeights;

        biases.noalias() += momentumBiases;
    }

    void RadialBasisFunctionLayer::resetStepData
            ()
    {
        currentNumberOfSteps = 0;

        deltaWeights.setZero();
        momentumWeights.setZero();

        deltaBiases.setZero();
        momentumBiases.setZero();
    }
}

////////////////////////////////////////////////////////////////////////////////
