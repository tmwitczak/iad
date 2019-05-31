///////////////////////////////////////////////////////////////////// | Includes
#include "affine-layer.hpp"

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
    ///////////////////////////////////////////////////// | Class: AffineLayer <
    //============================================================= | Methods <<
    //----------------------------------------------------- | Static methods <<<
    void AffineLayer::initialiseRandomNumberGenerator
            (int const seed)
    {
        srand(static_cast<unsigned int>(seed));
    }

    //------------------------------------------------------- | Constructors <<<
    AffineLayer::AffineLayer
            ()
            :
            AffineLayer(1, 1, Sigmoid {}, true)
    {
    }

    AffineLayer::AffineLayer
            (int const numberOfInputs,
             int const numberOfOutputs,
             ActivationFunction const &activationFunction,
             bool const enableBias)
            :
            NeuralNetworkLayer {},

            weights { std::sqrt(2.0 / (numberOfInputs + numberOfOutputs))
                      * Matrix::Random(numberOfOutputs, numberOfInputs) },
            deltaWeights { Matrix::Zero(numberOfOutputs, numberOfInputs) },
            momentumWeights { Matrix::Zero(numberOfOutputs, numberOfInputs) },

            biases { std::sqrt(2.0 / (numberOfInputs + numberOfOutputs))
                     * Vector::Random(numberOfOutputs) },
            deltaBiases { Vector::Zero(numberOfOutputs) },
            momentumBiases { Vector::Zero(numberOfOutputs) },

            activationFunction { activationFunction.clone() },
            currentNumberOfSteps { 0 },
            isBiasEnabled { enableBias }
    {
    }

    AffineLayer::AffineLayer
            (std::string const &filename)
            :
            NeuralNetworkLayer {}
    {
        std::ifstream file;
        file.open(filename, std::ios::binary);
        {
            cereal::BinaryInputArchive binaryInputArchive(file);
            binaryInputArchive(*this);
        }
        file.close();
    }

    AffineLayer::AffineLayer
            (AffineLayer const &affineLayer)
            :
            NeuralNetworkLayer { affineLayer },

            weights { affineLayer.weights },
            deltaWeights { affineLayer.deltaWeights },
            momentumWeights { affineLayer.momentumWeights },
            biases { affineLayer.biases },
            deltaBiases { affineLayer.deltaBiases },
            momentumBiases { affineLayer.momentumBiases },
            activationFunction { affineLayer.activationFunction->clone() },
            currentNumberOfSteps { affineLayer.currentNumberOfSteps },
            isBiasEnabled { affineLayer.isBiasEnabled }
    {
    }

    //---------------------------------------------------------- | Operators <<<
    Vector AffineLayer::operator()
            (Vector const &inputs) const
    {
        return feedForward(inputs);
    }

    //------------------------------ | Interface: Cloneable | Implementation <<<
    std::unique_ptr<NeuralNetworkLayer> AffineLayer::clone
            () const
    {
        return std::make_unique<AffineLayer>(*this);
    }

    //----------------------------------------------------- | Main behaviour <<<
    Vector AffineLayer::calculateOutputs
            (Vector const &inputs) const
    {
        return weights * inputs
               + (isBiasEnabled ? biases : Vector::Zero(biases.size()));
    }

    Vector AffineLayer::activate
            (Vector const &outputs) const
    {
        return (*activationFunction)(outputs);
    }

    Vector AffineLayer::calculateOutputsDerivative
            (Vector const &outputs) const
    {
        return activationFunction->derivative(outputs);
    }

    Vector AffineLayer::feedForward
            (Vector const &inputs) const
    {
        return activate(calculateOutputs(inputs));
    }

    Vector AffineLayer::backpropagate
            (Vector const &inputs,
             Vector const &errors,
             Vector const &outputs,
             Vector const &outputsDerivative) const
    {
        return weights.transpose()
               * (errors.array() * outputsDerivative.array()).matrix();
    }

    void AffineLayer::calculateNextStep
            (Vector const &inputs,
             Vector const &errors,
             Vector const &outputs,
             Vector const &outputsDerivative)
    {
        Vector derivative = -errors.array()
                            * outputsDerivative.array();

        deltaWeights.noalias() -= derivative * inputs.transpose();

        if (isBiasEnabled)
            deltaBiases.noalias() -= derivative;

        ++currentNumberOfSteps;
    }

    void AffineLayer::update
            (double const learningCoefficient,
             double const momentumCoefficient)
    {
        applyAverageOfDeltaStepsToMomentumStep(learningCoefficient,
                                               momentumCoefficient);
        applyMomentumStepToWeightsAndBiases();
        resetStepData();
    }

    void AffineLayer::saveToFile
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
    int AffineLayer::numberOfInputs
            () const
    {
        return weights.cols();
    }

    int AffineLayer::numberOfOutputs
            () const
    {
        return weights.rows();
    }

    //--------------------------------------------------- | Helper functions <<<
    void AffineLayer::applyAverageOfDeltaStepsToMomentumStep
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

        if (isBiasEnabled)
            momentumBiases.noalias()
                    = momentumCoefficient * momentumBiases
                      + (learningCoefficient / currentNumberOfSteps)
                        * deltaBiases;
    }

    void AffineLayer::applyMomentumStepToWeightsAndBiases
            ()
    {
        weights.noalias() += momentumWeights;

        if (isBiasEnabled)
            biases.noalias() += momentumBiases;
    }

    void AffineLayer::resetStepData
            ()
    {
        currentNumberOfSteps = 0;

        deltaWeights.setZero();
        momentumWeights.setZero();

        if (isBiasEnabled)
        {
            deltaBiases.setZero();
            momentumBiases.setZero();
        }
    }

    ///////////////////////////////////////////// | Class: AffineLayerWithBias <
    //============================================================= | Methods <<
    //------------------------------------------------------- | Constructors <<<
    AffineLayerWithBias::AffineLayerWithBias
            ()
            :
            AffineLayerWithBias(1, 1, Sigmoid {})
    {
    }

    AffineLayerWithBias::AffineLayerWithBias
            (int const numberOfInputs,
             int const numberOfOutputs,
             ActivationFunction const &activationFunction)
            :
            AffineLayer { numberOfInputs,
                          numberOfOutputs,
                          activationFunction,
                          true }
    {
    }

    AffineLayerWithBias::AffineLayerWithBias
            (std::string const &filename)
            :
            AffineLayer { filename }
    {
    }

    AffineLayerWithBias::AffineLayerWithBias
            (AffineLayerWithBias const &affineLayerWithBias)
            :
            AffineLayer { affineLayerWithBias }
    {
    }

    //------------------------------ | Interface: Cloneable | Implementation <<<
    std::unique_ptr<NeuralNetworkLayer> AffineLayerWithBias::clone
            () const
    {
        return std::make_unique<AffineLayerWithBias>(*this);
    }

    ////////////////////////////////////////// | Class: AffineLayerWithoutBias <
    //============================================================= | Methods <<
    //------------------------------------------------------- | Constructors <<<
    AffineLayerWithoutBias::AffineLayerWithoutBias
            ()
            :
            AffineLayerWithoutBias(1, 1, Sigmoid {})
    {
    }

    AffineLayerWithoutBias::AffineLayerWithoutBias
            (int const numberOfInputs,
             int const numberOfOutputs,
             ActivationFunction const &activationFunction)
            :
            AffineLayer { numberOfInputs,
                          numberOfOutputs,
                          activationFunction,
                          false }
    {
    }

    AffineLayerWithoutBias::AffineLayerWithoutBias
            (std::string const &filename)
            :
            AffineLayer { filename }
    {
    }

    AffineLayerWithoutBias::AffineLayerWithoutBias
            (AffineLayerWithoutBias const &affineLayerWithBias)
            :
            AffineLayer { affineLayerWithBias }
    {
    }

    //------------------------------ | Interface: Cloneable | Implementation <<<
    std::unique_ptr<NeuralNetworkLayer> AffineLayerWithoutBias::clone
            () const
    {
        return std::make_unique<AffineLayerWithoutBias>(*this);
    }


}

////////////////////////////////////////////////////////////////////////////////
