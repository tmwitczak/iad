///////////////////////////////////////////////////////////////////// | Includes
#include "rectified-linear-unit.hpp"

////////////////////////////////////////////////////// TODO: Name this section.
using Array = Eigen::ArrayXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////// | Class: RectifiedLinearUnit <
    //============================================================= | Methods <<
    //-------------------------------- | Interface implementation: Cloneable <<<
    std::unique_ptr<ActivationFunction> RectifiedLinearUnit::clone
            () const
    {
        return std::make_unique<RectifiedLinearUnit>(*this);
    }

    //----------------------- | Interface implementation: ActivationFunction <<<
    Array RectifiedLinearUnit::operator()
            (Array const &inputs) const
    {
        return inputs.max(0.0);
    }

    Array RectifiedLinearUnit::derivative
            (Array const &inputs) const
    {
        return this->operator()(inputs).sign();
    }
}

////////////////////////////////////////////////////////////////////////////////
