#ifndef IAD_2A_TRAINING_EXAMPLE_HPP
#define IAD_2A_TRAINING_EXAMPLE_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include <Eigen>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////////////// | Struct: TrainingExample <
    struct TrainingExample
    {
        //========================================================== | Fields <<
        //-------------------------------------------------------- | Vectors <<<
        Eigen::VectorXd inputs;
        Eigen::VectorXd outputs;
    };
}

////////////////////////////////////////////////////////////////////////////////
#endif //IAD_2A_TRAINING_EXAMPLE_HPP
