#ifndef IAD_2A_SINGLE_LAYER_PERCEPTRON_HPP
#define IAD_2A_SINGLE_LAYER_PERCEPTRON_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "training-example.hpp"

#include <Eigen>
#include <vector>

///////////////////////////////////////////////////////////// | Class definition
class SingleLayerPerceptron
{
public:
    //============================================================== | Methods <
    //------------------------------------------------------ | Static methods <<
    static void initialiseRandomSeed
            (int const &seed);

    //-------------------------------------------------------- | Constructors <<
    SingleLayerPerceptron
            (int const &numberOfInputs,
             int const &numberOfOutputs);

    //----------------------------------------------- | TODO: Name this section.
    int numberOfInputs
            () const;

    int numberOfOutputs
            () const;

    void train
            (std::vector<TrainingExample> const &trainingExamples,
             int const &numberOfEpochs,
             double const &errorGoal,
             double const &learningRate,
             double const &momentum);

    // TODO: Write function with testing examples.

    Eigen::VectorXd backpropagate
            (Eigen::VectorXd const &errors,
             Eigen::VectorXd const &outputs) const;

    Eigen::VectorXd feedForward
            (Eigen::VectorXd const &inputs) const;

    //----------------------------------------------------------- | Operators <<
    Eigen::VectorXd operator()
            (Eigen::VectorXd const &inputs) const;

private:
    //============================================================== | Structs <
    struct Layer
    {
        Eigen::MatrixXd weights;
        Eigen::VectorXd biases;
    };

    //=============================================================== | Fields <
    std::vector<Layer> layers;

    Eigen::ArrayXd (*activation)
            (Eigen::ArrayXd const &input);

    Eigen::ArrayXd (*activationDerivative)
            (Eigen::ArrayXd const &input);

    //============================================================== | Methods <
    static Eigen::ArrayXd sigmoid
            (Eigen::ArrayXd const &input);

    static Eigen::ArrayXd sigmoidDerivative
            (Eigen::ArrayXd const &input);

    static Eigen::ArrayXd rectifiedLinearUnit
            (Eigen::ArrayXd const &input);

    static Eigen::ArrayXd rectifiedLinearUnitDerivative
            (Eigen::ArrayXd const &input);
};

////////////////////////////////////////////////////////////////////////////////
#endif //IAD_2A_SINGLE_LAYER_PERCEPTRON_HPP
