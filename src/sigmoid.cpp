///////////////////////////////////////////////////////////////////// | Includes
#include "sigmoid.hpp"

////////////////////////////////////////////////////// TODO: Name this section.
using Array = Eigen::ArrayXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////////////////// | Class: Sigmoid <
    //============================================================= | Methods <<
    //-------------------------------- | Interface implementation: Cloneable <<<
    std::unique_ptr<ActivationFunction> Sigmoid::clone
            () const
    {
        return std::make_unique<Sigmoid>(*this);
    }

    //----------------------- | Interface implementation: ActivationFunction <<<
    Array Sigmoid::operator()
            (Array const &input) const
    {
        return 1.0 / (1.0 + (-input).exp());
    }

    Array Sigmoid::derivative
            (Array const &input) const
    {
        Eigen::ArrayXd sigmoidOutput = (*this)(input);
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }
}

////////////////////////////////////////////////////////////////////////////////