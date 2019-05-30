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

////////////////////////////////////////////// | cereal: Archive specialisations
namespace cereal
{
    template <typename Archive>
    void save
            (Archive &archive,
             Matrix const &matrix)
    {
        int matrixRows = matrix.rows();
        int matrixColumns = matrix.cols();

        archive(matrixRows);
        archive(matrixColumns);

        archive(binary_data(matrix.data(),
                            matrixRows * matrixColumns * sizeof(double)));
    }

    template <typename Archive>
    void load
            (Archive &archive,
             Matrix &matrix)
    {
        int matrixRows = matrix.rows();
        int matrixColumns = matrix.cols();

        archive(matrixRows);
        archive(matrixColumns);

        matrix.resize(matrixRows, matrixColumns);

        archive(binary_data(matrix.data(),
                            matrixRows * matrixColumns * sizeof(double)));
    }

    template <typename Archive>
    void save
            (Archive &archive,
             Vector const &vector)
    {
        save(archive, Matrix { vector });
    }

    template <typename Archive>
    void load
            (Archive &archive,
             Vector &vector)
    {
        Matrix vectorAsMatrix;
        load(archive, vectorAsMatrix);
        vector = vectorAsMatrix;
    }
}

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////////// | Class: PerceptronLayer <
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
            weights { Matrix::Random(numberOfOutputs, numberOfInputs) },
            deltaWeights { Matrix::Zero(numberOfOutputs, numberOfInputs) },
            momentumWeights { Matrix::Zero(numberOfOutputs, numberOfInputs) },

            biases { Vector::Random(numberOfOutputs) },
            deltaBiases { Vector::Zero(numberOfOutputs) },
            momentumBiases { Vector::Zero(numberOfOutputs) },

            activationFunction { activationFunction.clone() },
            currentNumberOfSteps { 0 },
            isBiasEnabled { enableBias }
    {
    }

    AffineLayer::AffineLayer
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

    AffineLayer::AffineLayer
            (AffineLayer const &affineLayer)
            :
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
            (Vector const &outputsDerivative,
             Vector const &errors) const
    {
        return weights.transpose()
               * (errors.array() * outputsDerivative.array()).matrix();
    }

    void AffineLayer::calculateNextStep
            (Vector const &inputs,
             Vector const &errors,
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

    template <typename Archive>
    void AffineLayer::save
            (Archive &archive) const
    {
        archive(weights, deltaWeights, momentumWeights,
                biases, deltaBiases, momentumBiases,
                activationFunction,
                currentNumberOfSteps,
                isBiasEnabled);
    }

    template <typename Archive>
    void AffineLayer::load
            (Archive &archive)
    {
        archive(weights, deltaWeights, momentumWeights,
                biases, deltaBiases, momentumBiases,
                activationFunction,
                currentNumberOfSteps,
                isBiasEnabled);
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

}

////////////////////////////////////////////////////////////////////////////////
