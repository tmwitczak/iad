///////////////////////////////////////////////////////////////////// | Includes
#include "parametric-rectified-linear-unit.hpp"

/////////////////////////////////////////////////////////// | Using declarations
using Array = Eigen::ArrayXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    /////////////////////////////////// | Class: ParametricRectifiedLinearUnit <
    //============================================================= | Methods <<
    //------------------------------------------------------- | Constructors <<<
    ParametricRectifiedLinearUnit::ParametricRectifiedLinearUnit
            (double const &parameter)
            :
            parameter { parameter }
    {
    }

    //-------------------------------- | Interface implementation: Cloneable <<<
    std::unique_ptr<ActivationFunction> ParametricRectifiedLinearUnit::clone
            () const
    {
        return std::make_unique<ParametricRectifiedLinearUnit>(*this);
    }

    //----------------------- | Interface implementation: ActivationFunction <<<
    Array ParametricRectifiedLinearUnit::operator()
            (Array const &inputs) const
    {
        return inputs.max(0.0) + parameter * inputs.min(0.0);
    }

    Array ParametricRectifiedLinearUnit::derivative
            (Array const &inputs) const
    {
        return inputs.max(0.0).sign().abs()
               + parameter * inputs.min(0.0).sign().abs();
    }
}

////////////////////////////////////////////////////////////////////////////////
