///////////////////////////////////////////////////////////////////// | Includes
#include "perceptron-layer.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

////////////////////////////////////////////////////// TODO: Name this section.
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
    void PerceptronLayer::initialiseRandomNumberGenerator
            (int const seed)
    {
        srand(static_cast<unsigned int>(seed));
    }

    //------------------------------------------------------- | Constructors <<<
    PerceptronLayer::PerceptronLayer
            ()
            :
            PerceptronLayer(1, 1)
    {
    }

    PerceptronLayer::PerceptronLayer
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

    PerceptronLayer::PerceptronLayer
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

    //---------------------------------------------------------- | Operators <<<
    Vector PerceptronLayer::operator()
            (Vector const &inputs) const
    {
        return feedForward(inputs);
    }

    //----------------------------------------------------- | Main behaviour <<<
    Vector PerceptronLayer::calculateOutputs
            (Vector const &inputs) const
    {
        return weights * inputs
               + (isBiasEnabled ? biases : Vector::Zero(biases.size()));
    }

    Vector PerceptronLayer::activate
            (Vector const &outputs) const
    {
        return (*activationFunction)(outputs);
    }

    Vector PerceptronLayer::calculateOutputsDerivative
            (Vector const &outputs) const
    {
        return activationFunction->derivative(outputs);
    }

    Vector PerceptronLayer::feedForward
            (Vector const &inputs) const
    {
        return activate(calculateOutputs(inputs));
    }

    Vector PerceptronLayer::backpropagate
            (Vector const &outputsDerivative,
             Vector const &errors) const
    {
        return weights.transpose()
               * (errors.array() * outputsDerivative.array()).matrix();
    }

    void PerceptronLayer::calculateNextStep
            (Vector const &inputs,
             Vector const &errors,
             Vector const &outputsDerivative)
    {
        Vector derivative = (-errors.array()
                             * outputsDerivative.array());

        deltaWeights -= (derivative * Eigen::RowVectorXd(inputs));

        if (isBiasEnabled)
            deltaBiases -= (derivative);

        ++currentNumberOfSteps;
    }

    void PerceptronLayer::update
            (double const learningCoefficient,
             double const momentumCoefficient)
    {
        applyAverageOfDeltaStepsToMomentumStep(learningCoefficient,
                                               momentumCoefficient);
        applyMomentumStepToWeightsAndBiases();
        resetStepData();
    }

    void PerceptronLayer::saveToFile
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
    void PerceptronLayer::save
            (Archive &archive) const
    {
        archive(weights, deltaWeights, momentumWeights,
                biases, deltaBiases, momentumBiases,
                activationFunction,
                currentNumberOfSteps,
                isBiasEnabled);
    }

    template <typename Archive>
    void PerceptronLayer::load
            (Archive &archive)
    {
        archive(weights, deltaWeights, momentumWeights,
                biases, deltaBiases, momentumBiases,
                activationFunction,
                currentNumberOfSteps,
                isBiasEnabled);
    }

    //------------------------------------------------------------- | Traits <<<
    int PerceptronLayer::numberOfInputs
            () const
    {
        return weights.cols();
    }

    int PerceptronLayer::numberOfOutputs
            () const
    {
        return weights.rows();
    }

    //--------------------------------------------------- | Helper functions <<<
    void PerceptronLayer::applyAverageOfDeltaStepsToMomentumStep
            (double const learningCoefficient,
             double const momentumCoefficient)
    {
        (momentumWeights *= momentumCoefficient)
                += (learningCoefficient
                    * (deltaWeights / currentNumberOfSteps));

        if (isBiasEnabled)
            (momentumBiases *= momentumCoefficient)
                    += (learningCoefficient
                        * (deltaBiases / currentNumberOfSteps));
    }

    void PerceptronLayer::applyMomentumStepToWeightsAndBiases
            ()
    {
        weights += momentumWeights;

        if (isBiasEnabled)
            biases += momentumBiases;
    }

    void PerceptronLayer::resetStepData
            ()
    {
        currentNumberOfSteps = 0;

        deltaWeights = Matrix::Zero(numberOfOutputs(), numberOfInputs());
        momentumWeights = Matrix::Zero(numberOfOutputs(), numberOfInputs());

        if (isBiasEnabled)
        {
            deltaBiases = Vector::Zero(numberOfOutputs());
            momentumBiases = Vector::Zero(numberOfOutputs());
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
