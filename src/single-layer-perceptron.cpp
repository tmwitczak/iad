///////////////////////////////////////////////////////////////////// | Includes
#include "single-layer-perceptron.hpp"

#include <ctime>
#include <utility>

////////////////////////////////////////////////////// TODO: Name this section.
using Array = Eigen::ArrayXd;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

//////////////////////////////////////////////////////// | SingleLayerPerceptron
//================================================================== | Methods <
//---------------------------------------------------------- | Static methods <<
void SingleLayerPerceptron::initialiseRandomSeed
        (int const &seed)
{
    srand(static_cast<unsigned int>(seed));
}

//------------------------------------------------------------ | Constructors <<
SingleLayerPerceptron::SingleLayerPerceptron
        (int const &numberOfInputs,
         int const &numberOfOutputs)
        :
        weights(Matrix::Random(numberOfOutputs, numberOfInputs)),
        biases(Vector::Random(numberOfOutputs)),
        activation { sigmoid },
        activationDerivative { sigmoidDerivative }
{
    // TODO: Add choosing of activation function.
}

//--------------------------------------------------------------- | Accessors <<
int SingleLayerPerceptron::numberOfInputs
        () const
{
    return weights.cols();
}

int SingleLayerPerceptron::numberOfOutputs
        () const
{
    return weights.rows();
}

Vector SingleLayerPerceptron::backpropagate
        (Vector const &errors,
         Vector const &outputs) const
{
    Vector propagatedError
            { weights.transpose()
              * Vector { activationDerivative(outputs.array()) }};

    (propagatedError /= propagatedError.sum()) *= errors.sum();

    return propagatedError;
}

Vector SingleLayerPerceptron::feedForward
        (Vector const &inputs) const
{
    return activation(weights * inputs + biases);
}

Vector SingleLayerPerceptron::operator()
        (Vector const &inputs) const
{
    return feedForward(inputs);
}

void SingleLayerPerceptron::train
        (std::vector<TrainingExample> const &trainingExamples,
         int const &numberOfEpochs,
         double const &errorGoal,
         double const &learningRate,
         double const &momentum)
{
    Matrix weightsMomentum { Matrix::Zero(weights.rows(), weights.cols()) };
    Vector biasesMomentum { Vector::Zero(biases.size()) };

    for (int epoch = 0; epoch < numberOfEpochs; epoch++)
    {
        Matrix deltaWeights { Matrix::Zero(weights.rows(), weights.cols()) };
        Vector deltaBiases { Vector::Zero(biases.size()) };
        Vector errorsSum { Vector::Zero(biases.size()) };

        for (auto const &trainingExample : trainingExamples)
        {
            auto const &inputs = trainingExample.inputs;
            auto const &targets = trainingExample.outputs;

            Vector outputs = feedForward(trainingExample.inputs);

            Vector errors = targets - outputs;
            errorsSum += errors;

            // Update weights and biases
            for (int i = 0; i < outputs.size(); i++)
                for (int j = 0; j < inputs.size(); j++)
                {
                    deltaWeights(i, j) += learningRate * inputs(j) * errors(i)
                                          * outputs(i) * (1.0 - outputs(i));
                }

            for (int i = 0; i < outputs.size(); i++)
            {
                deltaBiases(i) += learningRate * errors(i)
                                  * outputs(i) * (1.0 - outputs(i));
            }
        }

        (weightsMomentum *= momentum) += (deltaWeights / trainingExamples.size
                ());
        (biasesMomentum *= momentum) += (deltaBiases / trainingExamples.size());

        weights += weightsMomentum;
        biases += biasesMomentum;

        double cost = ((errorsSum.array() / trainingExamples.size())
                       * (errorsSum.array() / trainingExamples.size())).sum();

        if (cost < errorGoal)
            break;
    }
}


Array SingleLayerPerceptron::sigmoid
        (Array const &input)
{
    return 1.0 / (1.0 + Eigen::exp(-input));
}

Array SingleLayerPerceptron::sigmoidDerivative
        (Array const &input)
{
    Array sigmoidOutput = sigmoid(input);
    return sigmoidOutput * (1.0 - sigmoidOutput);
}

Array SingleLayerPerceptron::rectifiedLinearUnit
        (Array const &input)
{
    return sigmoid(input); // TODO: Write ReLU function.
}

Array SingleLayerPerceptron::rectifiedLinearUnitDerivative
        (Array const &input)
{
    return sigmoidDerivative(
            input); // TODO: Write ReLU derivative function.
}