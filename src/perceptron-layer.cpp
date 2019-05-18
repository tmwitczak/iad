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
    Vector PerceptronLayer::feedForward
            (Vector const &inputs) const
    {
        return (*activationFunction)(weights * inputs
                                     + (isBiasEnabled
                                        ? biases
                                        : Vector::Zero(biases.size())));
    }

    Vector PerceptronLayer::backpropagate
            (Vector const &inputs,
             Vector const &errors) const
    {
//        Vector propagatedError
//                { weights.transpose()
//                  * Vector { outputs.array() * (1.0 - outputs.array()) }};
////        Vector { activationFunction->derivative(outputs.array())
////        }};
//
//        (propagatedError /= propagatedError.sum()) *= errors.sum();
//
//        return propagatedError;

        Vector propagatedError { Vector::Zero(numberOfInputs()) };
        Vector outputsDerivative
                = activationFunction->derivative
                        (weights * inputs + (isBiasEnabled
                                             ? biases
                                             : Vector::Zero(biases.size())));

        for (int i = 0; i < numberOfOutputs(); i++)
        {
            Vector gradientOfSingleOutput { numberOfInputs() };

            for (int j = 0; j < numberOfInputs(); j++)
                gradientOfSingleOutput(j) = outputsDerivative(i)
                                            * weights(i, j);


//            double std_dev
//                = std::sqrt((gradientOfSingleOutput.array()
//                        - gradientOfSingleOutput.mean()).square().sum()
//                                /(gradientOfSingleOutput.size() - 1));
//            gradientOfSingleOutput = (gradientOfSingleOutput.array()
//                     - gradientOfSingleOutput.array().mean()) / std_dev;

//            gradientOfSingleOutput
//                    /= gradientOfSingleOutput.array().abs().sum();


            // TODO: Normalise?
            gradientOfSingleOutput.normalize();
            gradientOfSingleOutput *= 1.25;

            propagatedError += (gradientOfSingleOutput * errors(i));
        }

        return propagatedError;

    }

    void PerceptronLayer::calculateNextStep
            (Vector const &inputs,
             Vector const &errors,
             Vector const &outputs)
    {
        Vector outputsDerivative
                = activationFunction->derivative
                        (weights * inputs + (isBiasEnabled
                                             ? biases
                                             : Vector::Zero(biases.size())));

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
